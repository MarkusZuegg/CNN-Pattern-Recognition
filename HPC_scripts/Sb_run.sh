#!/bin/bash
time=0-00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --job-name="CIFAR10"
#SBATCH --mail-user=s4744924@student.uq.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH —output=output_dir/%j.out

module load cuda/11.8
conda activate pytorch2

# python train_model.py
echo hello world 