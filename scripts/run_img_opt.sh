# SD fp32
CUDA_VISIBLE_DEVICES=0 python main.py --img_opt --text "A high quality 3D render of a strawberry" --workspace trial_imgopt --batch_size 10 --iters 100 --lr 0.1  --seed 0
# DeepFloydIF fp32
CUDA_VISIBLE_DEVICES=0 python main.py --IF --img_opt --text "A high quality 3D render of a strawberry" --workspace trial_imgopt --batch_size 10 --iters 100 --lr 0.1  --seed 0
# # SD fp16
# CUDA_VISIBLE_DEVICES=0 python main.py -O --img_opt --text "A high quality 3D render of a strawberry" --workspace trial_imgopt --batch_size 10 --iters 100 --lr 0.1 --seed 0
# # DeepFloydIF fp16
# CUDA_VISIBLE_DEVICES=0 python main.py --IF -O --img_opt --text "A high quality 3D render of a strawberry" --workspace trial_imgopt --batch_size 10 --iters 100 --lr 0.1  --seed 0