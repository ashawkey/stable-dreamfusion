#! /bin/bash

# CUDA_VISIBLE_DEVICES=1 python main.py -O --text "an ice cream sundae" --workspace trial_icecream --iters 10000
# CUDA_VISIBLE_DEVICES=1 python main.py -O --text "an ice cream sundae" --workspace trial2_icecream --dmtet --iters 15000 --ckpt trial_icecream/checkpoints/df.pth

# CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a kingfisher bird" --workspace trial_bird --iters 10000
# CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a kingfisher bird" --workspace trial2_bird --dmtet --iters 15000 --ckpt trial_bird/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a car made of sushi" --workspace trial_sushi --iters 10000
CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a car made of sushi" --workspace trial2_sushi --dmtet --iters 15000 --ckpt trial_sushi/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a marble bust of a mouse" --workspace trial_mouse --iters 10000
CUDA_VISIBLE_DEVICES=0 python main.py -O --text "a marble bust of a mouse" --workspace trial2_mouse --dmtet --iters 15000 --ckpt trial_mouse/checkpoints/df.pth