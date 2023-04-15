#! /bin/bash

CUDA_VISIBLE_DEVICES=5 python main.py -O --text "a DSLR photo of a shiba inu playing golf wearing tartan golf clothes and hat" --workspace trial_shiba --iters 10000
CUDA_VISIBLE_DEVICES=5 python main.py -O --text "a DSLR photo of a shiba inu playing golf wearing tartan golf clothes and hat" --workspace trial2_shiba --dmtet --iters 5000 --init_ckpt trial_shiba/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=5 python main.py -O --text "a banana peeling itself" --workspace trial_banana --iters 10000
CUDA_VISIBLE_DEVICES=5 python main.py -O --text "a banana peeling itself" --workspace trial2_banana --dmtet --iters 5000 --init_ckpt trial_banana/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=5 python main.py -O --text "a capybara wearing a top hat, low poly" --workspace trial_capybara --iters 10000
CUDA_VISIBLE_DEVICES=5 python main.py -O --text "a capybara wearing a top hat, low poly" --workspace trial2_capybara --dmtet --iters 5000 --init_ckpt trial_capybara/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=5 python main.py -O --text " a small saguaro cactus planted in a clay pot" --workspace trial_cactus --iters 10000
CUDA_VISIBLE_DEVICES=5 python main.py -O --text " a small saguaro cactus planted in a clay pot" --workspace trial2_cactus --dmtet --iters 5000 --init_ckpt trial_cactus/checkpoints/df.pth