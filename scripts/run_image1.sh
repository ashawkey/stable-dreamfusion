# zero123 backend (single object, images like 3d model rendering)
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/firekeeper_rgba.png --workspace trial_image_firekeeper --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/firekeeper_rgba.png --workspace trial2_image_firekeeper --iters 10000 --dmtet --init_ckpt trial_image_firekeeper/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/cactus_rgba.png --workspace trial_image_cactus --iters 5000
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/cactus_rgba.png --workspace trial2_image_cactus --iters 10000 --dmtet --init_ckpt trial_image_cactus/checkpoints/df.pth

# sd backend (realistic images)
# CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/fox_rgba.png --text "a fox sitting on the ground" --workspace trial_image_fox --iters 5000
# CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/fox_rgba.png --text "a fox sitting on the ground" --workspace trial2_image_fox --iters 10000 --dmtet --init_ckpt trial_image_fox/checkpoints/df.pth