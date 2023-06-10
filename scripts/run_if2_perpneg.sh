#! /bin/bash
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a lion bust" --workspace trial_perpneg_if_lion --iters 5000 --IF --batch_size 1 --perpneg
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a DSLR photo of a lion" --workspace trial_perpneg_if2_lion_p --dmtet --iters 5000 --perpneg --init_with trial_perpneg_if_lion/checkpoints/df.pth
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a DSLR photo of a lion" --workspace trial_perpneg_if2_lion_nop --dmtet --iters 5000 --init_with trial_perpneg_if_lion/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a tiger cub" --workspace trial_perpneg_if_tiger --iters 5000 --IF --batch_size 1 --perpneg
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "tiger" --workspace trial_perpneg_if2_tiger_p --dmtet --iters 5000 --perpneg --init_with trial_perpneg_if_tiger/checkpoints/df.pth
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "tiger" --workspace trial_perpneg_if2_tiger_nop --dmtet --iters 5000 --init_with trial_perpneg_if_tiger/checkpoints/df.pth

# larger negative weight is used for the following command because the defult negative weight of -2 is not enough to make the diffusion model to produce the views as desired
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a shiba dog wearing sunglasses" --workspace trial_perpneg_if_shiba --iters 5000 --IF --batch_size 1 --perpneg --negative_w -3.0
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "shiba wearing sunglasses"  --workspace trial_perpneg_if2_shiba_p --dmtet --iters 5000 --perpneg --negative_w -3.0 --init_with trial_perpneg_if_shiba/checkpoints/df.pth
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "shiba wearing sunglasses" --workspace trial_perpneg_if2_shiba_nop --dmtet --iters 5000 --init_with trial_perpneg_if_shiba/checkpoints/df.pth

