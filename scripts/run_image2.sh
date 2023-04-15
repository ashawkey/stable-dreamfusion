CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/hamburger_rgba.png --workspace trial_image_hamburger --iters 5000
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/hamburger_rgba.png --workspace trial2_image_hamburger --iters 10000 --dmtet --init_ckpt trial_image_hamburger/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/cake_rgba.png --workspace trial_image_cake --iters 5000
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/cake_rgba.png --workspace trial2_image_cake --iters 10000 --dmtet --init_ckpt trial_image_cake/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/backpack_rgba.png --workspace trial_image_backpack --iters 5000
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/backpack_rgba.png --workspace trial2_image_backpack --iters 10000 --dmtet --init_ckpt trial_image_backpack/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/dragon_rgba.png --workspace trial_image_dragon --iters 5000
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/dragon_rgba.png --workspace trial2_image_dragon --iters 10000 --dmtet --init_ckpt trial_image_dragon/checkpoints/df.pth