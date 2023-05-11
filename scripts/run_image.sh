# zero123 backend (single object, images like 3d model rendering)

CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/teddy_rgba.png --workspace trial_image_teddy --iters 5000
CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/teddy_rgba.png --workspace trial2_image_teddy --iters 5000 --dmtet --init_with trial_image_teddy/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/catstatue_rgba.png --workspace trial_image_catstatue --iters 5000
CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/catstatue_rgba.png --workspace trial2_image_catstatue --iters 5000 --dmtet --init_with trial_image_catstatue/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/firekeeper_rgba.png --workspace trial_image_firekeeper --iters 5000
CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/firekeeper_rgba.png --workspace trial2_image_firekeeper --iters 5000 --dmtet --init_with trial_image_firekeeper/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/hamburger_rgba.png --workspace trial_image_hamburger --iters 5000
CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/hamburger_rgba.png --workspace trial2_image_hamburger --iters 5000 --dmtet --init_with trial_image_hamburger/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/corgi_rgba.png --workspace trial_image_corgi --iters 5000
CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/corgi_rgba.png --workspace trial2_image_corgi --iters 5000 --dmtet --init_with trial_image_corgi/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/cactus_rgba.png --workspace trial_image_cactus --iters 5000
CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/cactus_rgba.png --workspace trial2_image_cactus --iters 5000 --dmtet --init_with trial_image_cactus/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/cake_rgba.png --workspace trial_image_cake --iters 5000
CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/cake_rgba.png --workspace trial2_image_cake --iters 5000 --dmtet --init_with trial_image_cake/checkpoints/df.pth

# CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/warrior_rgba.png --workspace trial_image_warrior --iters 5000
# CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/warrior_rgba.png --workspace trial2_image_warrior --iters 5000 --dmtet --init_with trial_image_warrior/checkpoints/df.pth