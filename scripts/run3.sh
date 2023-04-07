#! /bin/bash

CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of an ice cream sundae" --workspace trial_icecream --iters 10000
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of an ice cream sundae" --workspace trial2_icecream --dmtet --iters 5000 --init_ckpt trial_icecream/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of a kingfisher bird" --workspace trial_bird --iters 10000
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a DSLR photo of a kingfisher bird" --workspace trial2_bird --dmtet --iters 5000 --init_ckpt trial_bird/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a car made of sushi" --workspace trial_sushi --iters 10000
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a car made of sushi" --workspace trial2_sushi --dmtet --iters 5000 --init_ckpt trial_sushi/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a highly detailed stone bust of Theodoros Kolokotronis" --workspace trial_stonehead --iters 10000
CUDA_VISIBLE_DEVICES=1 python main.py -O --text "a highly detailed stone bust of Theodoros Kolokotronis" --workspace trial2_stonehead --dmtet --iters 5000 --init_ckpt trial_stonehead/checkpoints/df.pth