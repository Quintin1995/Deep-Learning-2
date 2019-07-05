#!/usr/bin/env bash
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=welu11

python3 main.py 
