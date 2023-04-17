#! /bin/bash

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "ironman, full body" --workspace trial_ironman --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "ironman, full body" --workspace trial2_ironman --dmtet --iters 5000 --init_ckpt trial_ironman/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of an ice cream sundae" --workspace trial_icecream --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of an ice cream sundae" --workspace trial2_icecream --dmtet --iters 5000 --init_ckpt trial_icecream/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a kingfisher bird" --workspace trial_bird --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a kingfisher bird" --workspace trial2_bird --dmtet --iters 5000 --init_ckpt trial_bird/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a car made of sushi" --workspace trial_sushi --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a car made of sushi" --workspace trial2_sushi --dmtet --iters 5000 --init_ckpt trial_sushi/checkpoints/df.pth
