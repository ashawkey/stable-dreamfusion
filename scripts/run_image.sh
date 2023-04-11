CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/hamburger_rgba.png --workspace trial_image_hamburger --iters 10000
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/bear_rgba.png --workspace trial_image_bear --iters 10000
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/squirrel_rgba.png --workspace trial_image_squirrel --iters 10000

CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/hamburger_rgba.png --workspace trial2_image_hamburger --iters 5000 --dmtet --init_ckpt trial_image_hamburger/checkpoints/df.pth
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/bear_rgba.png --workspace trial2_image_bear --iters 5000 --dmtet --init_ckpt trial_image_bear/checkpoints/df.pth
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/squirrel_rgba.png --workspace trial2_image_squirrel --iters 5000 --dmtet --init_ckpt trial_image_squirrel/checkpoints/df.pth