#! /bin/bash

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a delicious hamburger" --workspace trial_vanilla_hamburger --iters 10000 --backbone vanilla
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a delicious hamburger" --workspace trial_vanilla2_hamburger --dmtet --iters 5000 --init_ckpt trial_vanilla_hamburger/checkpoints/df.pth --backbone vanilla

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a peacock on a surfboard" --workspace trial_vanilla_peacock --iters 10000 --backbone vanilla
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a peacock on a surfboard" --workspace trial_vanilla2_peacock --dmtet --iters 5000 --init_ckpt trial_vanilla_peacock/checkpoints/df.pth --backbone vanilla

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "superman, full body" --workspace trial_vanilla_superman --iters 10000 --backbone vanilla
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "superman, full body" --workspace trial_vanilla2_superman --dmtet --iters 5000 --init_ckpt trial_vanilla_superman/checkpoints/df.pth --backbone vanilla

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a kingfisher bird" --workspace trial_vanilla_bird --iters 10000 --backbone vanilla
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a kingfisher bird" --workspace trial_vanilla2_bird --dmtet --iters 5000 --init_ckpt trial_vanilla_bird/checkpoints/df.pth --backbone vanilla