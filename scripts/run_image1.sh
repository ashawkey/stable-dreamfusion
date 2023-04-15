CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/cat_rgba.png --workspace trial_image_cat --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/cat_rgba.png --workspace trial2_image_cat --iters 10000 --dmtet --init_ckpt trial_image_cat/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/firekeeper_rgba.png --workspace trial_image_firekeeper --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/firekeeper_rgba.png --workspace trial2_image_firekeeper --iters 10000 --dmtet --init_ckpt trial_image_firekeeper/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/cactus_rgba.png --workspace trial_image_cactus --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/cactus_rgba.png --workspace trial2_image_cactus --iters 10000 --dmtet --init_ckpt trial_image_cactus/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/panda_rgba.png --workspace trial_image_panda --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/panda_rgba.png --workspace trial2_image_panda --iters 10000 --dmtet --init_ckpt trial_image_panda/checkpoints/df.pth
