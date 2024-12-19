#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --gpus-per-task=1 
#SBATCH --time 15:00:00
#SBATCH --qos=gpu
#SBATCH -A cs433
#SBATCH -q cs433

cd /home/coberger/ml-project-2-team-kangaroos/

git pull

pip3 install -r requirements.txt

# python3 train.py -cn exp_arch_mobilenetv3
python3 train.py -cn exp_arch_unet_small
python3 train.py -cn exp_arch_unet_medium
python3 train.py -cn exp_arch_unet_large
# python3 train.py -cn exp_arch_resnet18
# python3 train.py -cn exp_arch_resnet36