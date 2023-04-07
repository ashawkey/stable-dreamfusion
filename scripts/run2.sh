#! /bin/bash


CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a DSLR photo of a squirrel-octopus hybrid" --workspace trial_squrrel_octopus --iters 10000
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a DSLR photo of a squirrel-octopus hybrid" --workspace trial2_squrrel_octopus --dmtet --iters 5000 --init_ckpt trial_squrrel_octopus/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a kangaroo sitting on a bench playing the accordion" --workspace trial_kangroo --iters 10000
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a kangaroo sitting on a bench playing the accordion" --workspace trial2_kangroo --dmtet --iters 5000 --init_ckpt trial_kangroo/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "ironman, full body" --workspace trial_ironman --iters 10000
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "ironman, full body" --workspace trial2_ironman --dmtet --iters 5000 --init_ckpt trial_ironman/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "an astronaut, full body" --workspace trial_astronaut --iters 10000
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "an astronaut, full body" --workspace trial2_astronaut --dmtet --iters 5000 --init_ckpt trial_astronaut/checkpoints/df.pth