# sd backend (realistic images)

CUDA_VISIBLE_DEVICES=3 python main.py -O --image data/fox_rgba.png --text "a fox sitting on the ground" --workspace trial_imagetext_fox --iters 5000
CUDA_VISIBLE_DEVICES=3 python main.py -O --image data/fox_rgba.png --text "a fox sitting on the ground" --workspace trial2_imagetext_fox --iters 10000 --dmtet --init_ckpt trial_imagetext_fox/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --image data/cactus_rgba.png --text "a cactus plant in a pot" --workspace trial_imagetext_cactus --iters 5000
CUDA_VISIBLE_DEVICES=3 python main.py -O --image data/cactus_rgba.png --text "a cactus plant in a pot" --workspace trial2_imagetext_cactus --iters 10000 --dmtet --init_ckpt trial_imagetext_cactus/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=3 python main.py -O --image data/hamburger_rgba.png --text "a DSLR photo of a delicious hamburger" --workspace trial_imagetext_hamburger --iters 5000
CUDA_VISIBLE_DEVICES=3 python main.py -O --image data/hamburger_rgba.png --text "a DSLR photo of a delicious hamburger" --workspace trial2_imagetext_hamburger --iters 10000 --dmtet --init_ckpt trial_imagetext_hamburger/checkpoints/df.pth