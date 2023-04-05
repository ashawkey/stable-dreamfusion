#! /bin/bash

# CUDA_VISIBLE_DEVICES=3 python main.py -O --text "superman, full body" --workspace trial_superman --iters 10000
# CUDA_VISIBLE_DEVICES=3 python main.py -O --text "superman, full body" --workspace trial2_superman --dmtet --iters 15000 --ckpt trial_superman/checkpoints/df.pth

# CUDA_VISIBLE_DEVICES=3 python main.py -O --text "an astronaut, full body" --workspace trial_astronaut --iters 10000
# CUDA_VISIBLE_DEVICES=3 python main.py -O --text "an astronaut, full body" --workspace trial2_astronaut --dmtet --iters 15000 --ckpt trial_astronaut/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a ripe strawberry" --workspace trial_strawberry --iters 10000
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a ripe strawberry" --workspace trial2_strawberry --dmtet --iters 15000 --ckpt trial_strawberry/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "an imperial state crown of england" --workspace trial_crown --iters 10000
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "an imperial state crown of england" --workspace trial2_crown --dmtet --iters 15000 --ckpt trial_crown/checkpoints/df.pth
