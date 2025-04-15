import inspect
import os
from typing import Union

import PIL
import numpy as np
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from model.utils import get_trainable_module, init_adapter
from model.attn_processor import SkipAttnProcessor

from utils import (compute_vae_encodings, numpy_to_pil, prepare_image,
                   prepare_mask_image, resize_and_crop, resize_and_padding)
from accelerate import load_checkpoint_in_model
from huggingface_hub import snapshot_download
from diffusers.utils.torch_utils import randn_tensor


class DiMVTONModel:
    def __init__(
        self, 
        base_ckpt, 
        attn_ckpt, 
        attn_ckpt_version="mix",
        weight_dtype=torch.float32,
        device='cuda',
        compile=False,
        skip_safety_check=False,
        use_tf32=True,
    ):
        self.device = device
        self.weight_dtype = weight_dtype
        self.skip_safety_check = skip_safety_check

        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device, dtype=weight_dtype)
        self.unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(device, dtype=weight_dtype)

        init_adapter(self.unet, cross_attn_cls=SkipAttnProcessor)  # Skip Cross-Attention

        self.attn_modules = get_trainable_module(self.unet, "attention")
        
        self.auto_attn_ckpt_load(attn_ckpt, attn_ckpt_version)
        self.training = True

        self.vae.eval()

        for param in self.vae.parameters():
            param.requires_grad = False

        # # Pytorch 2.0 Compile
        # if compile:
        #     self.unet = torch.compile(self.unet)
        #     self.vae = torch.compile(self.vae, mode="reduce-overhead")
            
        # # Enable TF32 for faster training on Ampere GPUs (A100 and RTX 30 series).
        # if use_tf32:
        #     torch.set_float32_matmul_precision("high")
        #     torch.backends.cuda.matmul.allow_tf32 = True

    def auto_attn_ckpt_load(self, attn_ckpt, version):
        sub_folder = {
            "mix": "mix-48k-1024",
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512",
        }[version]
        if os.path.exists(attn_ckpt):
            load_checkpoint_in_model(self.attn_modules, os.path.join(attn_ckpt, sub_folder, 'attention'))
        else:
            repo_path = snapshot_download(repo_id=attn_ckpt)
            print(f"Downloaded {attn_ckpt} to {repo_path}")
            load_checkpoint_in_model(self.attn_modules, os.path.join(repo_path, sub_folder, 'attention'))
            
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.noise_scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    def forward(
            self,
            image: Union[PIL.Image.Image, torch.Tensor],
            condition_image: Union[PIL.Image.Image, torch.Tensor],
            mask: Union[PIL.Image.Image, torch.Tensor],
            height: int = 1024,
            width: int = 768,
            generator=None,
            eta=1.0,
            conditional_dropout_prob: float = 0.1,
            dream_mask_dropout_prob: float = 0.1,
            **kwargs
        ):

        concat_dim = -2
        image = prepare_image(image).to(self.device, dtype=self.weight_dtype)
        condition_image = prepare_image(condition_image).to(self.device, dtype=self.weight_dtype)
        mask = prepare_mask_image(mask).to(self.device, dtype=self.weight_dtype)

        # To:DO : DREAM has not been incorporated to the training code

        
        # dream_mask = torch.rand((mask.size(0),), device=self.device) < dream_mask_dropout_prob
        # dream_mask = dream_mask[:, None, None, None]  # [B, 1, 1, 1]
        # mask = mask * (~dream_mask) + dream_mask  

        # Mask image
        masked_image = image * (mask < 0.5)
        # VAE encoding
        masked_latent = compute_vae_encodings(masked_image, self.vae)
        condition_latent = compute_vae_encodings(condition_image, self.vae)
        mask_latent = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode="nearest")

        dropout_mask = torch.rand((condition_latent.size(0),), device=self.device) < conditional_dropout_prob
        dropout_mask = dropout_mask[:, None, None, None]  # Expand to [B,1,1,1]
        condition_latent = condition_latent * (~dropout_mask)

        # Concatenate latents
        masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
        mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)], dim=concat_dim)
        # Prepare noise only for the masked region
        noise = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=self.weight_dtype,
        )
        # latents = mask_latent_concat * noise + (1 - mask_latent_concat) * masked_latent_concat
        # mask_latent_unchanged = mask_latent_concat

        # Prepare timesteps

        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (masked_latent_concat.shape[0],), device=masked_latent_concat.device).long()
        noisy_latents_full = self.noise_scheduler.add_noise(masked_latent_concat, noise, timesteps)
        noisy_latents = mask_latent_concat * noisy_latents_full + (1 - mask_latent_concat) * masked_latent_concat

        model_input = torch.cat([noisy_latents, mask_latent_concat, masked_latent_concat], dim=1)

        pred_noise = self.unet(model_input, timesteps, encoder_hidden_states=None)

        return pred_noise, noise, mask_latent_concat
    

    
# # --- DREAM Implementation Begins ---

# # Get alphas for each timestep
# alpha_bar_t = self.noise_scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
# sqrt_alpha_bar_t = alpha_bar_t.sqrt()
# sqrt_one_minus_alpha_bar_t = (1 - alpha_bar_t).sqrt()

# # 1. Generate noisy image (yt) from ground truth latent and noise
# yt = sqrt_alpha_bar_t * masked_latent_concat + sqrt_one_minus_alpha_bar_t * noise

# # 2. Predict noise from yt
# model_input_yt = torch.cat([yt, mask_latent_concat, masked_latent_concat], dim=1)
# pred_noise_yt = self.unet(model_input_yt, timesteps, encoder_hidden_states=None)

# # 3. Compute Δεₜ,θ = εₜ - StopGrad(ε̂ₜ,θ)
# delta_epsilon = noise - pred_noise_yt.detach()

# # 4. Compute λₜ
# lambda_t = self.compute_lambda_t(timesteps).to(yt.device)  # [B, 1, 1, 1]

# # 5. Generate rectified noisy image: ŷₜ = yₜ + sqrt(1 - ᾱₜ) * λₜ * Δεₜ,θ
# y_hat_t = yt + sqrt_one_minus_alpha_bar_t * lambda_t * delta_epsilon

# # 6. Predict noise from rectified input ŷₜ
# model_input_y_hat_t = torch.cat([y_hat_t, mask_latent_concat, masked_latent_concat], dim=1)
# pred_noise_y_hat_t = self.unet(model_input_y_hat_t, timesteps, encoder_hidden_states=None)

# # 7. Target: εₜ + λₜ * Δεₜ,θ
# target_noise = noise + lambda_t * delta_epsilon

# # 8. Compute L1 loss between target and ε̂(ŷₜ)
# loss = torch.nn.functional.l1_loss(pred_noise_y_hat_t, target_noise)

# return loss
