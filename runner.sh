#!/usr/bin/env bash
#SBATCH -J gan_test
#SBATCH -e gan-%j.txt
#SBATCH -o gan-%j.txt
#SBATCH -p gpgpu-1  --gres=gpu:1 --mem=40G
#SBATCH -t 10080
#SBATCH -n 32
#SBATCH --mail-user josephsd@smu.edu
#SBATCH --mail-type=ALL
#SBATCH --x11=batch

python simplest_gan.py
