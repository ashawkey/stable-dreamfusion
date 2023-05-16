# Phase 1 - barely fits in A100 40GB.
# Conclusion: results in concave-ish face, no neck, excess hair in the back
# CUDA_VISIBLE_DEVICES=0 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage \
#   --iters 10000 --save_guidance --save_guidance_interval 10 --ckpt scratch --batch_size 2 --test_interval 2 \
#   --h 128 --w 128 --zero123_grad_scale None

# TODO: try with lower noise... like it's supposed to.
# TODO: try decreasing frequency of ref view rgbd iterations

# Phase 2 - barely fits in A100 40GB.

# BEST SO FAR:
# GPU2: increase resolution. Repeated. Notes:
#  1. the guidance scale behavior changed in my hacked code... this one might be excessive (should be 10)
#  2. `--t_range 0.05 0.4` wasn't taken in consideration in previous code (was overridden)... might need to restore the previous default
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU2_again \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 15000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 256 --w 256 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 5 --radius_range 2.2 2.6 --test_interval 2 \
  --vram_O --guidance_scale 100 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.01 --progressive_view

# OTHER EXPERIMENTS IN PROGRESS or that failed

  # 10X the learning rate (--lr 0.01), and use --progressive_view and --iters 200000 (so the progressive view expansion actually works)
  # and --radius_range 2.2 2.6 (to bring camera closer)
CUDA_VISIBLE_DEVICES=0 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 20000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 128 --w 128 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 6 --radius_range 2.2 2.6 --test_interval 2 \
  --vram_O --guidance_scale 100 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.01 --progressive_view

# GPU1: Like GPU2, with updated code, with guidance scale set to 10, and lower learning rate.
CUDA_VISIBLE_DEVICES=1 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU1 \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 15000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 256 --w 256 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 3 --radius_range 2.2 2.6 --test_interval 2 \
  --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --progressive_view

# new GPU0: like new GPU1, with more iters
CUDA_VISIBLE_DEVICES=0 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU2-0 \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 30000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 256 --w 256 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 5 --radius_range 2.2 2.6 --test_interval 2 \
  --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --progressive_view

# GPU2 AGAIN2: increase resolution. Repeated. Note that the guidance scale behavior changed... this one might be excessive.
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU2_again2 \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 15000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 256 --w 256 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 5 --radius_range 2.2 2.6 --test_interval 2 \
  --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --progressive_view

# GPU3: increase resolution and use progressive level with lr=5X - crashed after 1/3 of epochs due to NaNs... !?
# tried again with fewer epochs, reduced resolution (was 400x400, now 320x320)
CUDA_VISIBLE_DEVICES=3 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU3 \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 15000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 320 --w 320 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 2 --radius_range 2.2 2.6 --test_interval 2 \
  --vram_O --guidance_scale 100 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --progressive_view --progressive_level

# GPU4: increase resolution and use progressive level with lr=5X, without progressive_view
CUDA_VISIBLE_DEVICES=4 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU4 \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 20000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 320 --w 320 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 2 --radius_range 3.5 3.7 --test_interval 2 \
  --vram_O --guidance_scale 100 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.005 --progressive_level


# From this point on, made changes to albedo,textureless, etc...

#5: new "baseline" with no progressive view, old res, increased LR=10X
CUDA_VISIBLE_DEVICES=5 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU5 \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 20000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 128 --w 128 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 6 --radius_range 3.2 3.7 --test_interval 2 \
  --vram_O --guidance_scale 100 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.01

#6: higher-res
CUDA_VISIBLE_DEVICES=6 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU6 \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 20000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 320 --w 320 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 2 --radius_range 3.2 3.7 --test_interval 2 \
  --vram_O --guidance_scale 100 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.01


# GPU 7 ready to try something new
#7: use my best guess for min_ambient_ratio, very low albedo_iter_ratio, no textureless
CUDA_VISIBLE_DEVICES=7 debugpy-run main.py -- -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU7 \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 15000 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 256 --w 256 --albedo_iter_ratio 0.2 --t_range 0.05 0.4 --batch_size 8 --radius_range 1.6 2.3 --test_interval 2 \
  --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --textureless_ratio 0.0 --min_ambient_ratio 0.3 --dont_override_stuff --progressive_view 
            


# Phase 3 - eventually runs OOM on A100 40GB
# Conclusion: Arms merge incorrectly. Takes way too long.
CUDA_VISIBLE_DEVICES=0 python main.py -O --workspace trial_anya_1_refimage_C \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 40000 --ckpt trial_anya_1_refimage_B/checkpoints/df_ep0200.pth --save_guidance --save_guidance_interval 10 \
  --h 512 --w 512 --albedo_iter_ratio 0.7 --t_range 0.05 0.4 --batch_size 4 --radius_range 3.5 3.7 --test_interval 2 \
  --vram_O --guidance_scale 100 --jitter_pose  --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.01

# Phase 4 - untested, need to adjust
# CUDA_VISIBLE_DEVICES=0 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage --iters 5000 --dmtet --init_with trial_anya_1_refimage/checkpoints/df.pth

