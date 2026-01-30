#!/bin/bash    
#SBATCH --job-name=image_sample_clevr_rel
#SBATCH --output=logs/image_sample_rel_%j.out
#SBATCH --error=logs/image_sample_rel_%j.err
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8    
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00    
#SBATCH --mem=32G  
  
module load Mambaforge/23.11.0-fasrc01  
mamba activate compose_diff  

# Conjunction (AND) 
MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma True --use_scale_shift_norm False --num_classes 4,3,9,3,3,7 --raw_unet True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
python scripts/image_relation_test.py $MODEL_FLAGS $DIFFUSION_FLAGS --ckpt_path ./logs_clevr_rel_128/ema_0.9999_740000.pt