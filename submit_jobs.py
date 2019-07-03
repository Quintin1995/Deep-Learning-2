import os, subprocess, time


#calls a subprocess on peregrine to start an experiment with specific parameters declared below.
def run (act, bat, weight, drop, opt):
        template = """#!/usr/bin/env bash
#SBATCH --time=05:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name={act}{drop}{bat}{weight}

module load OpenCV/3.4.1-foss-2018a-Python-3.6.4
module load TensorFlow/1.10.1-fosscuda-2018a-Python-3.6.4
module load CUDA/9.1.85-GCC-6.4.0-2.28
module load Boost/1.66.0-foss-2018a-Python-3.6.4
module load Python/3.6.4-fosscuda-2018a
pip install keras --user

python3 ./ownnet.py {act} {drop} {bat} {weight} {opt}"""

        filename = "batch_cpu.txt"
        script   = open(filename, "w+")
        jobstring = template.format(act=act, drop=drop, bat=bat, weight=weight, opt=opt)
        script.write(jobstring)
        script.close()
        try:
                subprocess.call(["sbatch", filename])
                pass
        except OSError:
                script.close()
                print("sbatch not found or filename wrong")
        os.remove(filename)
        print ("Submitted job: ", filename)
        print (jobstring)
        time.sleep(1)

# Baseline parameters for the basic cnn architecture defined in old/cnn.py
b = dict()
b["act"] = "relu"
b["bat"] = 0
b["weight"] = 0.0
b["drop"] = 0
b["opt"]  = "adam"

# Run baseline
run(b["act"], b["bat"], b["weight"], b["drop"], b["opt"])

# Run experiments (8)
for act in ["elu", "sigmoid"]:
        run(act, b["bat"], b["weight"], b["drop"], b["opt"])
for bat in [1]:
        run(b["act"], bat, b["weight"], b["drop"], b["opt"])
for weight in [0.0001]:
        run(b["act"], b["bat"], weight, b["drop"], b["opt"])
for drop in [1]:
        run(b["act"], b["bat"], b["weight"], drop, b["opt"])
for opt in ["sgd", "rmsprop"]:
        run(b["act"], b["bat"], b["weight"], b["drop"], opt)