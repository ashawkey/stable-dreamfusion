#! /bin/bash

CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a DSLR photo of a lobster playing the saxophone" --workspace trial_vanilla_lobster --iters 10000 --backbone vanilla
CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a DSLR photo of a lobster playing the saxophone" --workspace trial_vanilla2_lobster --dmtet --iters 5000 --init_ckpt trial_vanilla_lobster/checkpoints/df.pth --backbone vanilla

CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a DSLR photo of a shiba inu playing golf wearing tartan golf clothes and hat" --workspace trial_vanilla_shiba --iters 10000 --backbone vanilla
CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a DSLR photo of a shiba inu playing golf wearing tartan golf clothes and hat" --workspace trial_vanilla2_shiba --dmtet --iters 5000 --init_ckpt trial_vanilla_shiba/checkpoints/df.pth --backbone vanilla

CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a banana peeling itself" --workspace trial_vanilla_banana --iters 10000 --backbone vanilla
CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a banana peeling itself" --workspace trial_vanilla2_banana --dmtet --iters 5000 --init_ckpt trial_vanilla_banana/checkpoints/df.pth --backbone vanilla

CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a capybara wearing a top hat, low poly" --workspace trial_vanilla_capybara --iters 10000 --backbone vanilla
CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a capybara wearing a top hat, low poly" --workspace trial_vanilla2_capybara --dmtet --iters 5000 --init_ckpt trial_vanilla_capybara/checkpoints/df.pth --backbone vanilla

CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a plush dragon toy" --workspace trial_vanilla_dragon --iters 10000 --backbone vanilla
CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a plush dragon toy" --workspace trial_vanilla2_dragon --dmtet --iters 5000 --init_ckpt trial_vanilla_dragon/checkpoints/df.pth --backbone vanilla

CUDA_VISIBLE_DEVICES=0 python main.py -O --text " a small saguaro cactus planted in a clay pot" --workspace trial_vanilla_cactus --iters 10000 --backbone vanilla
CUDA_VISIBLE_DEVICES=0 python main.py -O --text " a small saguaro cactus planted in a clay pot" --workspace trial_vanilla2_cactus --dmtet --iters 5000 --init_ckpt trial_vanilla_cactus/checkpoints/df.pth --backbone vanilla