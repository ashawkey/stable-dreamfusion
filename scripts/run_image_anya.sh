# Phase 1 - barely fits in A100 40GB.
# Conclusion: results in concave-ish face, no neck, excess hair in the back
CUDA_VISIBLE_DEVICES=0 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage \
  --iters 10000 --save_guidance --save_guidance_interval 10 --ckpt scratch --batch_size 2 --test_interval 2 \
  --h 128 --w 128 --zero123_grad_scale None

# Phase 2 - barely fits in A100 40GB.
# Conclusion: Excess hair in the back turns into an extra arm. Takes way too long.
CUDA_VISIBLE_DEVICES=0 python main.py -O --workspace trial_anya_1_refimage_B \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 20000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 10 \
  --h 128 --w 128 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 10 --radius_range 3.5 3.7 --test_interval 2 \
  --vram_O --guidance_scale 100 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01

# Phase 3 - eventually runs OOM on A100 40GB
# Conclusion: Arms merge incorrectly. Takes way too long.
CUDA_VISIBLE_DEVICES=0 python main.py -O --workspace trial_anya_1_refimage_C \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 40000 --ckpt trial_anya_1_refimage_B/checkpoints/df_ep0200.pth --save_guidance --save_guidance_interval 10 \
  --h 512 --w 512 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 4 --radius_range 3.5 3.7 --test_interval 2 \
  --vram_O --guidance_scale 100 --jitter_pose  --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01

# Phase 4 - untested, need to adjust
# CUDA_VISIBLE_DEVICES=0 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage --iters 5000 --dmtet --init_with trial_anya_1_refimage/checkpoints/df.pth

