#! /bin/bash

# CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a delicious hamburger" --workspace trial_hamburger --iters 10000
# CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a delicious hamburger" --workspace trial2_hamburger --dmtet --iters 15000 --ckpt trial_hamburger/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a cat lying on its side batting at a ball of yarn" --workspace trial_cat_lying --iters 10000
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a cat lying on its side batting at a ball of yarn" --workspace trial2_cat_lying --dmtet --iters 15000 --ckpt trial_cat_lying/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "the leaning tower of Pisa, aerial view" --workspace trial_tower --iters 10000
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "the leaning tower of Pisa, aerial view" --workspace trial2_tower --dmtet --iters 15000 --ckpt trial_tower/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a peacock on a surfboard" --workspace trial_peacock --iters 10000
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a peacock on a surfboard" --workspace trial2_peacock --dmtet --iters 15000 --ckpt trial_peacock/checkpoints/df.pth