import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from model.dimvton_model_noise_whole_lora import DiMVTONModel
from utils import compute_vae_encodings, prepare_image, prepare_mask_image, resize_and_crop, resize_and_padding
from diffusers.image_processor import VaeImageProcessor
from PIL import Image, ImageFilter
import wandb

class DiMVTONDataset(Dataset):
    def __init__(self, args, test=False):
        self.args = args
        self.test = test

        self.vae_processor = VaeImageProcessor(vae_scale_factor=8) 
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

        # Set correct pair file and root directory
        if self.test:
            pair_file = 'test_pairs_unpaired.txt' if not self.args.eval_pair else 'test_pairs.txt'
            data_subdir = 'test'
        else:
            pair_file = 'train_pairs.txt'
            data_subdir = 'train'

        pair_txt_path = os.path.join(self.args.data_root_path, pair_file)
        assert os.path.exists(pair_txt_path), f"Pair file {pair_txt_path} does not exist."

        self.pair_txt_path = pair_txt_path
        self.data_root = os.path.join(self.args.data_root_path, data_subdir)

        self.data = self.load_data()

    def load_data(self):
        with open(self.pair_txt_path, 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            person_img, cloth_img = line.strip().split(" ")

            # For paired eval, override cloth_img with person_img
            if self.args.eval_pair:
                cloth_img = person_img

            person_path = os.path.join(self.data_root, 'image', person_img)
            cloth_path = os.path.join(self.data_root, 'cloth', cloth_img)
            mask_path = os.path.join(self.data_root, 'agnostic-mask', person_img.replace('.jpg', '_mask.png'))

            data.append({
                'person_name': person_img,
                'person': person_path,
                'cloth': cloth_path,
                'mask': mask_path,
            })
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        # Load images
        person_img = Image.open(data['person']).convert('RGB')
        cloth_img = Image.open(data['cloth']).convert('RGB')
        mask_img = Image.open(data['mask']).convert('L')  # Mask is grayscale

        # Preprocess with VAE processors
        person_tensor = self.vae_processor.preprocess(person_img, self.args.height, self.args.width)[0]
        cloth_tensor = self.vae_processor.preprocess(cloth_img, self.args.height, self.args.width)[0]
        mask_tensor = self.mask_processor.preprocess(mask_img, self.args.height, self.args.width)[0]

        return {
            'index': idx,
            'person_name': data['person_name'],
            'person': person_tensor,
            'cloth': cloth_tensor,
            'mask': mask_tensor
        }

# === Training function ===
def train_dimvton(args):
    # Initialize pipeline
    model = DiMVTONModel(
        base_ckpt=args.base_ckpt,
        attn_ckpt=args.attn_ckpt,
        attn_ckpt_version=args.attn_ckpt_version,
        weight_dtype=torch.float32,
        device="cuda"
    )

    optimizer = torch.optim.AdamW(model.unet.parameters(), lr=args.lr)

    train_dataset = DiMVTONDataset(args, test = False)
    test_dataset = DiMVTONDataset(args, test = True)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    wandb.watch(model.unet, log="gradients", log_freq=args.gradient_log_freq)

    for epoch in range(args.epochs):
        model.unet.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")

        for batch in progress_bar:
            
            person = batch['person'].to(model.device)
            cloth = batch['cloth'].to(model.device)
            mask = batch['mask'].to(model.device)

            optimizer.zero_grad()
            
            scaler = torch.amp.GradScaler()
            # Forward pass
            with torch.amp.autocast(device_type="cuda"):

                pred_noise, noise, mask_latent_concat = model.forward(
                    image=person,
                    condition_image=cloth,
                    mask=mask,
                    height=args.height,
                    width=args.width,
                    conditional_dropout_prob=args.conditional_dropout_prob,
                    dream_mask_dropout_prob=args.dream_mask_dropout_prob
                )
                loss = ((pred_noise.sample - noise) ** 2) * mask_latent_concat
                loss = loss.mean()
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)            
            scaler.update()
            # Compute loss only on masked region
            

            # loss.backward()
            # optimizer.step()

            progress_bar.set_postfix(loss=loss.item())
            wandb.log({"train/loss": loss.item(), "epoch": epoch})
            # Save checkpoint
        
        model.unet.eval()
        test_losses = []

        with torch.no_grad():
            for batch in test_dataloader:
                person = batch['person'].to(model.device)
                cloth = batch['cloth'].to(model.device)
                mask = batch['mask'].to(model.device)

                pred_noise, noise, mask_latent_concat = model.forward(
                    image=person,
                    condition_image=cloth,
                    mask=mask,
                    height=args.height,
                    width=args.width,
                    conditional_dropout_prob=args.conditional_dropout_prob,
                    dream_mask_dropout_prob=args.dream_mask_dropout_prob
                )
                # CFG: Forward passes x 2 : one with the conditionning and one without
                # This allows the pred_noise to be anything within the non masked rgion :P lol
                # stable diff inpainting pipeline 1.5  (check how it is being trained)
                loss = ((pred_noise.sample - noise) ** 2)
                loss = loss.mean()
                test_losses.append(loss.item())

        avg_test_loss = sum(test_losses) / len(test_losses)
        wandb.log({"test/loss": avg_test_loss, "epoch": epoch})

        if( (epoch+1) % args.save_freq == 0):

            save_path = os.path.join(args.checkpoint_dir, f"unet_epoch_{epoch}.pt")
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save(model.unet.state_dict(), save_path)
            wandb.save(save_path)

class TrainConfig:

    base_ckpt = "booksforcharlie/stable-diffusion-inpainting"
    attn_ckpt = "zhengchong/CatVTON"
    attn_ckpt_version = "vitonhd"  # or 'mix', 'dresscode'
    data_root_path = "/home/gershom/Documents/Ro/zalando-hd-resized"
    batch_size = 2
    epochs = 10
    lr = 1e-6
    device = "cuda"
    height = 512
    width = 384
    conditional_dropout_prob = 0.1
    dream_mask_dropout_prob = 0.1
    output_dir = "./output"
    checkpoint_dir = "./checkpoint"
    save_freq = 10
    gradient_log_freq = 50
    eval_pair = False 

if __name__ == "__main__":

    args = TrainConfig()
    wandb.init(project="DiMVTON", config=args, name=f"run_{wandb.util.generate_id()}")
    train_dimvton(args)
    wandb.finish()

