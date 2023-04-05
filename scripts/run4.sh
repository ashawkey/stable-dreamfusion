#! /bin/bash

# CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a mug of hot chocolate with whipped cream and marshmallows" --workspace trial_mug --iters 10000
# CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a mug of hot chocolate with whipped cream and marshmallows" --workspace trial2_mug --dmtet --iters 15000 --ckpt trial_mug/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a rabbit, animated movie character, high detail 3d model" --workspace trial_rabbit2 --iters 10000
CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a rabbit, animated movie character, high detail 3d model" --workspace trial2_rabbit2 --dmtet --iters 15000 --ckpt trial_rabbit2/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a teddy bear pushing a shopping cart full of fruits and vegetables" --workspace trial_teddy --iters 10000
CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a teddy bear pushing a shopping cart full of fruits and vegetables" --workspace trial2_teddy --dmtet --iters 15000 --ckpt trial_teddy/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a zoomed out DSLR photo of a rabbit cutting grass with a lawnmower" --workspace trial_rabbit --iters 10000
CUDA_VISIBLE_DEVICES=4 python main.py -O --text "a zoomed out DSLR photo of a rabbit cutting grass with a lawnmower" --workspace trial2_rabbit --dmtet --iters 15000 --ckpt trial_rabbit/checkpoints/df.pth