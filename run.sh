#!/bin/bash
#
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=10GB
#SBATCH --job-name=zi2zi
#SBATCH --mail-type=END
#SBATCH --mail-user=lj1035@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

module load pytorch/python3.6/0.3.0_4
module load torchvision/python3.5/0.1.9
module load scipy/intel/0.19.1

module load cuda/8.0.44
module load cudnn/8.0v5.1

time python3 zi2zi.py --n_epochs 250  --sample_interval 50  --generator_type "resnet"  --train_size 100 --checkpoint_interval 40 --augmentation 'flipleftright'
