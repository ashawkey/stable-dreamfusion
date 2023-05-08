# sd backend (realistic images)

CUDA_VISIBLE_DEVICES=4 python main.py -O --image data/teddy_rgba.png --text "a brown teddy bear sitting on a ground" --workspace trial_imagetext_teddy --iters 5000
CUDA_VISIBLE_DEVICES=4 python main.py -O --image data/teddy_rgba.png --text "a brown teddy bear sitting on a ground" --workspace trial2_imagetext_teddy --iters 10000 --dmtet --init_with trial_imagetext_teddy/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=4 python main.py -O --image data/corgi_rgba.png --text "a corgi running" --workspace trial_imagetext_corgi --iters 5000
CUDA_VISIBLE_DEVICES=4 python main.py -O --image data/corgi_rgba.png --text "a corgi running" --workspace trial2_imagetext_corgi --iters 10000 --dmtet --init_with trial_imagetext_corgi/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=4 python main.py -O --image data/hamburger_rgba.png --text "a DSLR photo of a delicious hamburger" --workspace trial_imagetext_hamburger --iters 5000
CUDA_VISIBLE_DEVICES=4 python main.py -O --image data/hamburger_rgba.png --text "a DSLR photo of a delicious hamburger" --workspace trial2_imagetext_hamburger --iters 10000 --dmtet --init_with trial_imagetext_hamburger/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=4 python main.py -O --image data/cactus_rgba.png --text "a potted cactus plant" --workspace trial_imagetext_cactus --iters 5000
CUDA_VISIBLE_DEVICES=4 python main.py -O --image data/cactus_rgba.png --text "a potted cactus plant" --workspace trial2_imagetext_cactus --iters 10000 --dmtet --init_with trial_imagetext_cactus/checkpoints/df.pth
