import os, subprocess, time


#calls a subprocess on peregrine to start an experiment with specific parameters declared below.
def run (algo):
        template = """#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=algo_{algo}

module load OpenCV/3.4.1-foss-2018a-Python-3.6.4
module load TensorFlow/1.10.1-fosscuda-2018a-Python-3.6.4
module load CUDA/9.1.85-GCC-6.4.0-2.28
module load Boost/1.66.0-foss-2018a-Python-3.6.4
module load Python/3.6.4-fosscuda-2018a
pip install keras --user

python3 main.py -t dqn {algo}"""

        filename = "batch_gpu.txt"
        script   = open(filename, "w+")
        jobstring = template.format(algo=algo)
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



# Run experiments:
for algo in ["", "--double", "--duel"]:
        run(algo)
