#!/usr/bin/bash
#SBATCH --nodes 1
#SBATCH -t 1-00:00:00
#SBATCH -p debug 
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --cpus-per-task 1
#SBATCH --gres=gpu:1
#SBATCH --exclude=mind-1-23
#SBATCH --output=/user_data/isaacl/cosyne-submission-runs/slurm_out/script_model_5_0to50.out

module load cuda-11.1.1

cd /user_data/isaacl/cosyne-submission-runs
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tangvit
python train_model_5.py 50 100 m1s1

module unload cuda-11.1.1