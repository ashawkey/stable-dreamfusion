#! /bin/bash
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a corgi taking a selfie" --workspace trial_if_corgi --iters 5000 --IF
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a corgi taking a selfie" --workspace trial_if2_corgi --dmtet --iters 5000 --init_with trial_if_corgi/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a DSLR photo of a ghost eating a hamburger" --workspace trial_if_ghost --iters 5000 --IF
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a DSLR photo of a ghost eating a hamburger" --workspace trial_if2_ghost --dmtet --iters 5000 --init_with trial_if_ghost/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a DSLR photo of an origami motorcycle" --workspace trial_if_motor --iters 5000 --IF
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a DSLR photo of an origami motorcycle" --workspace trial_if2_motor --dmtet --iters 5000 --init_with trial_if_motor/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a DSLR photo of a Space Shuttle" --workspace trial_if_spaceshuttle --iters 5000 --IF
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a DSLR photo of a Space Shuttle" --workspace trial_if2_spaceshuttle --dmtet --iters 5000 --init_with trial_if_spaceshuttle/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a palm tree, low poly 3d model" --workspace trial_if_palm --iters 5000 --IF
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a palm tree, low poly 3d model" --workspace trial_if2_palm --dmtet --iters 5000 --init_with trial_if_palm/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a zoomed out DSLR photo of a marble bust of a cat, a real mouse is sitting on its head" --workspace trial_if_cat_mouse --iters 5000 --IF
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a zoomed out DSLR photo of a marble bust of a cat, a real mouse is sitting on its head" --workspace trial_if2_cat_mouse --dmtet --iters 5000 --init_with trial_if_cat_mouse/checkpoints/df.pth