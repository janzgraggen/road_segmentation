#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-task=1 
#SBATCH --time 05:00:00
#SBATCH --qos=gpu
#SBATCH -A cs433
#SBATCH -q cs433

cd /home/coberger/ml-project-2-team-kangaroos/

git pull

pip3 install -r requirements.txt

# python3 train.py -cn exp_transform_erase
python3 train.py -cn exp_transform_rotation
python3 train.py -cn exp_transform_noise
python3 train.py -cn exp_transform_noisy