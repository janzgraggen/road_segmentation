#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-task=1 
#SBATCH --time 01:00:00
#SBATCH --qos=gpu
#SBATCH -A cs433
#SBATCH -q cs433

cd /home/coberger/ml-project-2-team-kangaroos/

git pull

pip3 install -r requirements.txt

python3 train.py -cn exp_patch_16 
python3 train.py -cn exp_patch_32 
python3 train.py -cn exp_patch_128 
python3 train.py -cn exp_patch_304 
python3 train.py -cn exp_patch_400 
