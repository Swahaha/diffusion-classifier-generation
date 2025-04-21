#!/bin/bash
#SBATCH --job-name=convnettraining
#SBATCH --output=convnet_training%j.log
#SBATCH --error=convnettraining%j.err
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00

module load python/3.6.15
source /home/users/ren11/project/diffusion-data-aug/.venv/bin/activate

cd /home/users/ren11/project/diffusion-data-aug


srun python diffusion_generate_new_dataset.py --dataset map --data_root Synthetic --model deeplabv3plus_resnet50 --output_stride 16 --batch_size 16 --val_batch_size 4 --total_itrs 10000 --lr 0.01 --gpu_id 0