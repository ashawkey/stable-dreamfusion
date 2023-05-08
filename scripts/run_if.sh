#! /bin/bash
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a delicious hamburger" --workspace trial_if_hamburger --iters 5000 --IF
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a delicious hamburger" --workspace trial_if2_hamburger --dmtet --iters 5000 --init_with trial_if_hamburger/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a baby bunny sitting on top of a stack of pancakes" --workspace trial_if_rabbit_pancake --iters 5000 --IF
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a metal bunny sitting on top of a stack of chocolate cookies" --workspace trial_if2_rabbit_pancake --dmtet --iters 5000 --init_with trial_if_rabbit_pancake/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a highly detailed stone bust of Theodoros Kolokotronis" --workspace trial_if_stonehead --iters 5000 --IF
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a highly detailed stone bust of Theodoros Kolokotronis" --workspace trial_if2_stonehead --dmtet --iters 5000 --init_with trial_if_stonehead/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "an astronaut, full body" --workspace trial_if_astronaut --iters 5000 --IF
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "an astronaut, full body" --workspace trial_if2_astronaut --dmtet --iters 5000 --init_with trial_if_astronaut/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a squirrel-octopus hybrid" --workspace trial_if_squrrel_octopus --iters 5000 --IF
CUDA_VISIBLE_DEVICES=2 python main.py -O --text "a DSLR photo of a squirrel-octopus hybrid" --workspace trial_if2_squrrel_octopus --dmtet --iters 5000 --init_with trial_if_squrrel_octopus/checkpoints/df.pth