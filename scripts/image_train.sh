#!/bin/bash    
#SBATCH --job-name=train_clevr_rel    
#SBATCH --output=logs/train_rel_%j.out    
#SBATCH --error=logs/train_rel_%j.err    
#SBATCH --nodes=1    
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8    
#SBATCH --partition=kempner
#SBATCH --account=kempner_ydu_lab
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00    
#SBATCH --mem=64G  
  
module load Mambaforge/23.11.0-fasrc01  
mamba activate compose_diff  
pip install wandb
  
# Set environment variables for single GPU  
export RANK=0  
export WORLD_SIZE=1  
export LOCAL_RANK=0  
export MASTER_ADDR=localhost  
export MASTER_PORT=12355  

export WANDB_API_KEY="4b9dc043dac87d50c74128f48c230b1755f954ea"

MODEL_FLAGS="--image_size 128 --num_channels 192 --num_res_blocks 2 --learn_sigma True --use_scale_shift_norm False --num_classes 4,3,9,3,3,7 --raw_unet True"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule squaredcos_cap_v2 --rescale_learned_sigmas False --rescale_timesteps False"
TRAIN_FLAGS="--lr 1e-5 --batch_size 16 --use_kl False --schedule_sampler loss-second-moment --microbatch -1"
srun python scripts/image_train.py --data_dir ./dataset/ --dataset clevr_rel --resume_checkpoint ./logs_clevr_rel_128/model130000.pt $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS