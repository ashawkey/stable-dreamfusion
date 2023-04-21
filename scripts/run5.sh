#! /bin/bash

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "Perched blue jay bird" --workspace trial_jay --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "Perched blue jay bird" --workspace trial2_jay --dmtet --iters 5000 --init_ckpt trial_jay/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "angel statue wings out" --workspace trial_angle --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "angel statue wings out" --workspace trial2_angle --dmtet --iters 5000 --init_ckpt trial_angle/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "devil statue" --workspace trial_devil --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "devil statue" --workspace trial2_devil --dmtet --iters 5000 --init_ckpt trial_devil/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=7 python main.py -O --text "Einstein statue" --workspace trial_einstein --iters 10000
CUDA_VISIBLE_DEVICES=7 python main.py -O --text "Einstein statue" --workspace trial2_einstein --dmtet --iters 5000 --init_ckpt trial_einstein/checkpoints/df.pth
