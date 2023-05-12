#! /bin/bash
# CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a baby bunny sitting on top of a stack of pancakes" --workspace trial_rabbit_pancake --iters 5000 --batch_size 1
# CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a metal bunny sitting on top of a stack of chocolate cookies" --workspace trial2_rabbit_pancake --dmtet --iters 5000 --init_with trial_rabbit_pancake/checkpoints/df.pth --batch_size 1

CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a DSLR photo of a blue jay standing on a large basket of rainbow macarons" --workspace trial_jay --iters 5000 --batch_size 1
CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a DSLR photo of a blue jay standing on a large basket of rainbow macarons" --workspace trial2_jay --dmtet --iters 5000 --init_with trial_jay/checkpoints/df.pth --batch_size 1

CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a DSLR photo of a fox taking a photograph using a DSLR" --workspace trial_fox --iters 5000 --batch_size 1
CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a DSLR photo of a fox taking a photograph using a DSLR" --workspace trial2_fox --dmtet --iters 5000 --init_with trial_fox/checkpoints/df.pth --batch_size 1

CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a DSLR photo of a peacock on a surfboard" --workspace trial_peacock --iters 5000 --batch_size 1
CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a DSLR photo of a peacock on a surfboard" --workspace trial2_peacock --dmtet --iters 5000 --init_with trial_peacock/checkpoints/df.pth --batch_size 1

CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a flower made out of metal" --workspace trial_metal_flower --iters 5000 --batch_size 1
CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a flower made out of metal" --workspace trial2_metal_flower --dmtet --iters 5000 --init_with trial_metal_flower/checkpoints/df.pth --batch_size 1

CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a zoomed out DSLR photo of an egg cracked open with a newborn chick hatching out of it" --workspace trial_chicken --iters 5000 --batch_size 1
CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a zoomed out DSLR photo of an egg cracked open with a newborn chick hatching out of it" --workspace trial2_chicken --dmtet --iters 5000 --init_with trial_chicken/checkpoints/df.pth --batch_size 1