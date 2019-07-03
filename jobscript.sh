#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=16000
#SBATCH --job-name=a3c_python
#SBATCH --cpus-per-task=4

ml TensorFlow/1.12.0-foss-2018a-Python-3.6.4
python main.py -t a3c
