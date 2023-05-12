#! /bin/bash
# CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a baby bunny sitting on top of a stack of pancakes" --workspace trial_if_rabbit_pancake --iters 5000 --IF --batch_size 2
# CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a metal bunny sitting on top of a stack of chocolate cookies" --workspace trial_if2_rabbit_pancake --dmtet --iters 5000 --init_with trial_if_rabbit_pancake/checkpoints/df.pth --batch_size 2

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a blue jay standing on a large basket of rainbow macarons" --workspace trial_if_jay --iters 5000 --IF --batch_size 2
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a blue jay standing on a large basket of rainbow macarons" --workspace trial_if2_jay --dmtet --iters 5000 --init_with trial_if_jay/checkpoints/df.pth --batch_size 2

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a fox taking a photograph using a DSLR" --workspace trial_if_fox --iters 5000 --IF --batch_size 2
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a fox taking a photograph using a DSLR" --workspace trial_if2_fox --dmtet --iters 5000 --init_with trial_if_fox/checkpoints/df.pth --batch_size 2

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a peacock on a surfboard" --workspace trial_if_peacock --iters 5000 --IF --batch_size 2
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a peacock on a surfboard" --workspace trial_if2_peacock --dmtet --iters 5000 --init_with trial_if_peacock/checkpoints/df.pth --batch_size 2

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a flower made out of metal" --workspace trial_if_metal_flower --iters 5000 --IF --batch_size 2
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a flower made out of metal" --workspace trial_if2_metal_flower --dmtet --iters 5000 --init_with trial_if_metal_flower/checkpoints/df.pth --batch_size 2

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a zoomed out DSLR photo of an egg cracked open with a newborn chick hatching out of it" --workspace trial_if_chicken --iters 5000 --IF --batch_size 2
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a zoomed out DSLR photo of an egg cracked open with a newborn chick hatching out of it" --workspace trial_if2_chicken --dmtet --iters 5000 --init_with trial_if_chicken/checkpoints/df.pth --batch_size 2