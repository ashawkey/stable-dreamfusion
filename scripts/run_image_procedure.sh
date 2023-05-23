# Perform a 2D-to-3D reconstruction, similar to the Anya case study: https://github.com/ashawkey/stable-dreamfusion/issues/263
# Args:
#    bash scripts/run_image_procedure.sh GPU_ID guidance_interval image_name "prompt"
# e.g.:
#    bash scripts/run_image_procedure 1 30 baby_phoenix_on_ice "An adorable baby phoenix made in Swarowski crystal highly detailed intricated concept art 8K"
GPU_ID=$1
GUIDANCE_INTERVAL=$2
DEFAULT_POLAR=$3
PREFIX=$4
PROMPT=$5
EPOCHS1=100
EPOCHS2=200
EPOCHS3=300
IMAGE=data/$PREFIX.png
IMAGE_RGBA=data/${PREFIX}_rgba.png
WS_PH1=trial_$PREFIX-ph1
WS_PH2=trial_$PREFIX-ph2
WS_PH3=trial_$PREFIX-ph3
CKPT1=$WS_PH1/checkpoints/df_ep0${EPOCHS1}.pth
CKPT2=$WS_PH2/checkpoints/df_ep0${EPOCHS2}.pth
CKPT3=$WS_PH3/checkpoints/df_ep0${EPOCHS3}.pth

# Can uncomment to clear up trial folders. Be careful - mistakes could erase important work!
# rm -r $WS_PH1 $WS_PH2 $WS_PH3

# Preprocess
if [ ! -f $IMAGE_RGBA ]
then
    python preprocess_image.py $IMAGE
fi

if [ ! -f $CKPT1 ]
then
    # Phase 1 - zero123-guidance
    # WARNING: claforte: constantly runs out of VRAM with resolution of 128x128 and batch_size 2... no longer able to reproduce Anya result because of this...
    #   I added these to try to reduce mem usage, but this might degrade the quality... `--lambda_depth 0 --lambda_3d_normal_smooth 0`
    # Remove: --ckpt scratch
    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -O --image $IMAGE_RGBA --workspace $WS_PH1 --default_polar $DEFAULT_POLAR \
      --iters ${EPOCHS1}00 --save_guidance --save_guidance_interval $GUIDANCE_INTERVAL --batch_size 1 --test_interval 2 \
      --h 96 --w 96 --zero123_grad_scale None --lambda_3d_normal_smooth 0 --dont_override_stuff \
      --fovy_range 20 20 --guidance_scale 5 
fi

GUIDANCE_INTERVAL=7
if [ ! -f $CKPT2 ]
then
  # Phase 2 - SD-guidance at 256x256
  CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -O --image $IMAGE_RGBA --workspace $WS_PH2 \
    --text "${PROMPT}" --default_polar $DEFAULT_POLAR \
    --iters ${EPOCHS2}00 --ckpt $CKPT1 --save_guidance --save_guidance_interval 7 \
    --h 128 --w 128 --albedo_iter_ratio 0.0 --t_range 0.2 0.6 --batch_size 4 --radius_range 2.2 2.6 --test_interval 2 \
    --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.1 --jitter_target 0.1 --jitter_up 0.05 \
    --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --progressive_view --progressive_view_init_ratio 0.05 --known_view_interval 2 --dont_override_stuff --lambda_3d_normal_smooth 1 --textureless_ratio 0.0 --min_ambient_ratio 0.3 \
    --exp_start_iter ${EPOCHS1}00 --exp_end_iter ${EPOCHS2}00
fi

if [ ! -f $CKPT3 ]
then
  # # Phase 3 - increase resolution to 512
  CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -O --image $IMAGE_RGBA --workspace $WS_PH3 \
    --text "${PROMPT}" --default_polar $DEFAULT_POLAR \
    --iters ${EPOCHS3}00 --ckpt $CKPT2  --save_guidance --save_guidance_interval 7 \
    --h 512 --w 512 --albedo_iter_ratio 0.0 --t_range 0.0 0.5 --batch_size 1 --radius_range 3.2 3.6 --test_interval 2 \
    --vram_O --guidance_scale 10 --jitter_pose --jitter_center 0.015 --jitter_target 0.015 --jitter_up 0.05 \
    --known_view_noise_scale 0 --lambda_depth 0 --lr 0.003 --known_view_interval 2 --dont_override_stuff --lambda_3d_normal_smooth 0.5 --textureless_ratio 0.0 --min_ambient_ratio 0.3 \
    --exp_start_iter ${EPOCHS2}00 --exp_end_iter ${EPOCHS3}00
fi

# Generate 6 views
CUDA_VISIBLE_DEVICES=$GPU_ID python main.py -O --image $IMAGE_RGBA --ckpt $CKPT3 --six_views

