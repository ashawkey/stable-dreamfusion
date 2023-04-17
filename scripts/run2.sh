#! /bin/bash

CUDA_VISIBLE_DEVICES=6 python main.py -O --text "a DSLR photo of a squirrel-octopus hybrid" --workspace trial_squrrel_octopus --iters 10000
CUDA_VISIBLE_DEVICES=6 python main.py -O --text "a DSLR photo of a squirrel-octopus hybrid" --workspace trial2_squrrel_octopus --dmtet --iters 5000 --init_ckpt trial_squrrel_octopus/checkpoints/df.pth

# the dmtet editing
CUDA_VISIBLE_DEVICES=6 python main.py -O --text "a baby bunny sitting on top of a stack of pancakes" --workspace trial_rabbit_pancake --iters 10000
CUDA_VISIBLE_DEVICES=6 python main.py -O --text "a metal bunny sitting on top of a stack of chocolate cookies" --workspace trial2_rabbit_pancake --dmtet --iters 5000 --init_ckpt trial_rabbit_pancake/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=6 python main.py -O --text "a highly detailed stone bust of Theodoros Kolokotronis" --workspace trial_stonehead --iters 10000
CUDA_VISIBLE_DEVICES=6 python main.py -O --text "a highly detailed stone bust of Theodoros Kolokotronis" --workspace trial2_stonehead --dmtet --iters 5000 --init_ckpt trial_stonehead/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=6 python main.py -O --text "an astronaut, full body" --workspace trial_astronaut --iters 10000
CUDA_VISIBLE_DEVICES=6 python main.py -O --text "an astronaut, full body" --workspace trial2_astronaut --dmtet --iters 5000 --init_ckpt trial_astronaut/checkpoints/df.pth