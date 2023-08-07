#! /bin/bash
#SBATCH -N 1
#SBATCH --array=0
#SBATCH -J magic123
#SBATCH -o slurm_logs/%x.%3a.%A.out
#SBATCH -e slurm_logs/%x.%3a.%A.err
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=30G
##SBATCH --gpus=1

module load gcc/7.5.0


#source ~/.bashrc
#source activate magic123
source venv_magic123/bin/activate
which python 

nvidia-smi
nvcc --version

hostname
NUM_GPU_AVAILABLE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
echo "number of gpus:" $NUM_GPU_AVAILABLE

RUN_ID=$2 # jobname for the first stage
RUN_ID2=$3 # jobname for the second stage
DATA_DIR=$4 # path to the directory containing the images, e.g. data/nerf4/chair
IMAGE_NAME=$5 # name of the image file, e.g. rgba.png
step1=$6 # whether to use the first stage
step2=$7 # whether to use the second stage

FILENAME=$(basename $DATA_DIR)
dataset=$(basename $(dirname $DATA_DIR))
echo reconstruct $FILENAME under dataset $dataset from folder $DATA_DIR ...

if (( ${step1} )); then
    CUDA_VISIBLE_DEVICES=$1 python main.py -O \
        --text "A high-resolution DSLR image of <token>" \
        --sd_version 1.5 \
        --image ${DATA_DIR}/${IMAGE_NAME} \
        --learned_embeds_path ${DATA_DIR}/learned_embeds.bin \
        --workspace out/magic123-${RUN_ID}-coarse/$dataset/magic123_${FILENAME}_${RUN_ID}_coarse \
        --optim adam \
        --iters 5000 \
        --guidance SD zero123 \
        --lambda_guidance 1.0 40 \
        --guidance_scale 100 5 \
        --latent_iter_ratio 0 \
        --normal_iter_ratio 0.2 \
        --t_range 0.2 0.6 \
        --bg_radius -1 \
        --save_mesh \
        ${@:8}
fi

if (( ${step2} )); then
    CUDA_VISIBLE_DEVICES=$1 python main.py -O \
        --text "A high-resolution DSLR image of <token>" \
        --sd_version 1.5 \
        --image ${DATA_DIR}/${IMAGE_NAME} \
        --learned_embeds_path ${DATA_DIR}/learned_embeds.bin \
        --workspace out/magic123-${RUN_ID}-${RUN_ID2}/$dataset/magic123_${FILENAME}_${RUN_ID}_${RUN_ID2} \
        --dmtet --init_ckpt out/magic123-${RUN_ID}-coarse/$dataset/magic123_${FILENAME}_${RUN_ID}_coarse/checkpoints/magic123_${FILENAME}_${RUN_ID}_coarse.pth \
        --iters 5000 \
        --optim adam \
        --latent_iter_ratio 0 \
        --guidance SD zero123 \
        --lambda_guidance 1e-3 0.01 \
        --guidance_scale 100 5 \
        --rm_edge \
        --bg_radius -1 \
        --save_mesh 
fi