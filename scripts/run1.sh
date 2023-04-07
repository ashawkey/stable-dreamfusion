#! /bin/bash

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a delicious hamburger" --workspace trial_hamburger --iters 10000
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a delicious hamburger" --workspace trial2_hamburger --dmtet --iters 5000 --init_ckpt trial_hamburger/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a cat lying on its side batting at a ball of yarn" --workspace trial_cat_lying --iters 10000
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a cat lying on its side batting at a ball of yarn" --workspace trial2_cat_lying --dmtet --iters 5000 --init_ckpt trial_cat_lying/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a peacock on a surfboard" --workspace trial_peacock --iters 10000
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a peacock on a surfboard" --workspace trial2_peacock --dmtet --iters 5000 --init_ckpt trial_peacock/checkpoints/df.pth

# the dmtet editing
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a baby bunny sitting on top of a stack of pancakes" --workspace trial_rabbit_pancake --iters 10000
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a metal bunny sitting on top of a stack of chocolate cookies" --workspace trial2_rabbit_pancake --dmtet --iters 5000 --init_ckpt trial_rabbit_pancake/checkpoints/df.pth