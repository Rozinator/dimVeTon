# dimVeTon


CUDA_VISIBLE_DEVICES=0 python inference.py --dataset vitonhd --data_root_path ./zalando-hd-resized --output_dir ./output --dataloader_num_workers 8 --batch_size 8 --seed 555 --mixed_precision no --allow_tf32 --repaint --eval_pair