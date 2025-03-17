#!/bin/bash
#SBATCH --job-name=python_multi
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=jinheek@andrew.cmu.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=4:00:00
#SBATCH --output=multi_cpu.log
#SBATCH --error=multi_cpu.err
#SBATCH -p cpu

# Load Python module
module load python3.12

# Run Python script
python /lab_data/barblab/Jinhee/auditory-attention-RSA/code/preprocess/preprocessing_main.py 
~
