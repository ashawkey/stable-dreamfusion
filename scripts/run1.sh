#! /bin/bash

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a cat lying on its side batting at a ball of yarn" --workspace trial_cat_lying --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a cat lying on its side batting at a ball of yarn" --workspace trial2_cat_lying --dmtet --iters 5000 --init_ckpt trial_cat_lying/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a peacock on a surfboard" --workspace trial_peacock --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a DSLR photo of a peacock on a surfboard" --workspace trial2_peacock --dmtet --iters 5000 --init_ckpt trial_peacock/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a kangaroo sitting on a bench playing the accordion" --workspace trial_kangroo --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a kangaroo sitting on a bench playing the accordion" --workspace trial2_kangroo --dmtet --iters 5000 --init_ckpt trial_kangroo/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a jumping cat, highly detailed" --workspace trial_cat --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "a jumping cat, highly detailed" --workspace trial2_cat --dmtet --iters 5000 --init_ckpt trial_cat/checkpoints/df.pth