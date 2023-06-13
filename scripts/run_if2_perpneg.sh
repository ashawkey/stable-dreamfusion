#! /bin/bash
# To avoid the Janus problem caused by the diffusion model's front view bias, utilize the Perp-Neg algorithm. To maximize its benefits,
# increase the absolute value of "negative_w" for improved Janus problem mitigation. If you encounter flat faces or divergence, consider 
# reducing the absolute value of "negative_w". The value of "negative_w" should vary for each prompt due to the diffusion model's varying 
# bias towards generating front views for different objects. Vary the weights within the range of 0 to -4.
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a lion bust" --workspace trial_perpneg_if_lion --iters 5000 --IF --batch_size 1 --perpneg
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a marble lion head" --workspace trial_perpneg_if2_lion_p --dmtet --iters 5000 --perpneg --init_with trial_perpneg_if_lion/checkpoints/df.pth
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a marble lion head" --workspace trial_perpneg_if2_lion_nop --dmtet --iters 5000 --init_with trial_perpneg_if_lion/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a tiger cub" --workspace trial_perpneg_if_tiger --iters 5000 --IF --batch_size 1 --perpneg
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "tiger" --workspace trial_perpneg_if2_tiger_p --dmtet --iters 5000 --perpneg --init_with trial_perpneg_if_tiger/checkpoints/df.pth
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "tiger" --workspace trial_perpneg_if2_tiger_nop --dmtet --iters 5000 --init_with trial_perpneg_if_tiger/checkpoints/df.pth

# larger absolute value of negative_w is used for the following command because the defult negative weight of -2 is not enough to make the diffusion model to produce the views as desired
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "a shiba dog wearing sunglasses" --workspace trial_perpneg_if_shiba --iters 5000 --IF --batch_size 1 --perpneg --negative_w -3.0
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "shiba wearing sunglasses"  --workspace trial_perpneg_if2_shiba_p --dmtet --iters 5000 --perpneg --negative_w -3.0 --init_with trial_perpneg_if_shiba/checkpoints/df.pth
CUDA_VISIBLE_DEVICES=3 python main.py -O --text "shiba wearing sunglasses" --workspace trial_perpneg_if2_shiba_nop --dmtet --iters 5000 --init_with trial_perpneg_if_shiba/checkpoints/df.pth

