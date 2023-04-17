CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/hamburger_rgba.png --workspace trial_image_hamburger --iters 5000
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/hamburger_rgba.png --workspace trial2_image_hamburger --iters 10000 --dmtet --init_ckpt trial_image_hamburger/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/catstatue_rgba.png --workspace trial_image_catstatue --iters 5000
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/catstatue_rgba.png --workspace trial2_image_catstatue --iters 10000 --dmtet --init_ckpt trial_image_catstatue/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/cake_rgba.png --workspace trial_image_cake --iters 5000
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/cake_rgba.png --workspace trial2_image_cake --iters 10000 --dmtet --init_ckpt trial_image_cake/checkpoints/df.pth