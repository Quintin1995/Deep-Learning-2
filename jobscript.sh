#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --mem=64000
#SBATCH --job-name=a3c_python
#SBATCH --cpus-per-task=24

ml TensorFlow/1.12.0-foss-2018a-Python-3.6.4
python main.py -t a3c
