# Phase 1 - barely fits in A100 40GB.
# Conclusion: results in concave-ish face, no neck, excess hair in the back
CUDA_VISIBLE_DEVICES=0 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage \
  --iters 10000 --save_guidance --save_guidance_interval 10 --ckpt scratch --batch_size 2 --test_interval 2 \
  --h 128 --w 128 --zero123_grad_scale None

# Phase 2 - barely fits in A100 40GB.
# 20X smaller lambda_3d_normal_smooth, --known_view_interval 2, 3X LR
# Much higher jitter to increase disparity (and eliminate some of the flatness)... not too high either (to avoid cropping the face)
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU2_reproduction1_GPU2 \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 12500 --ckpt trial_anya_1_refimage/checkpoints/df_ep0100.pth --save_guidance --save_guidance_interval 1 \
  --h 256 --w 256 --albedo_iter_ratio 0.0 --t_range 0.2 0.6 --batch_size 4 --radius_range 2.2 2.6 --test_interval 2 \
  --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.1 --jitter_target 0.1 --jitter_up 0.05 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --progressive_view --known_view_interval 2 --dont_override_stuff --lambda_3d_normal_smooth 1

# Phase 3 - increase resolution to 512
# Disable textureless since they can cause catastrophic divergence
# Since radius range is inconsistent, increase it, and reduce the jitter to avoid excessively cropped renders.
# Learning rate may be set too high, since `--batch_size 1`.
CUDA_VISIBLE_DEVICES=2 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage_B_GPU2_reproduction1_GPU2_refinedGPU2 \
  --text "A DSLR 3D photo of a cute anime schoolgirl stands proudly with her arms in the air, pink hair ( unreal engine 5 trending on Artstation Ghibli 4k )" \
  --iters 25000 --ckpt trial_anya_1_refimage_B_GPU2_reproduction1_GPU2/checkpoints/df_ep0125.pth  --save_guidance --save_guidance_interval 1 \
  --h 512 --w 512 --albedo_iter_ratio 0.0 --t_range 0.0 0.5 --batch_size 1 --radius_range 3.2 3.6 --test_interval 2 \
  --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.05 \
  --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --known_view_interval 2 --dont_override_stuff --lambda_3d_normal_smooth 0.5 --textureless_ratio 0.0 --min_ambient_ratio 0.3 

# Phase 4 - untested, need to adjust
# CUDA_VISIBLE_DEVICES=0 python main.py -O --image data/anya_front_rgba.png --workspace trial_anya_1_refimage --iters 5000 --dmtet --init_with trial_anya_1_refimage/checkpoints/df.pth

