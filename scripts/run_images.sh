# zero123 backend (single object, images like 3d model rendering)

CUDA_VISIBLE_DEVICES=6 python main.py -O --image_config config/corgi.csv --workspace trial_images_corgi --iters 5000
CUDA_VISIBLE_DEVICES=6 python main.py -O --image_config config/corgi.csv --workspace trial2_images_corgi --iters 10000 --dmtet --init_with trial_images_corgi/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=6 python main.py -O --image_config config/car.csv --workspace trial_images_car --iters 5000
CUDA_VISIBLE_DEVICES=6 python main.py -O --image_config config/car.csv --workspace trial2_images_car --iters 10000 --dmtet --init_with trial_images_car/checkpoints/df.pth

CUDA_VISIBLE_DEVICES=6 python main.py -O --image_config config/anya.csv --workspace trial_images_anya --iters 5000
CUDA_VISIBLE_DEVICES=6 python main.py -O --image_config config/anya.csv --workspace trial2_images_anya --iters 10000 --dmtet --init_with trial_images_anya/checkpoints/df.pth