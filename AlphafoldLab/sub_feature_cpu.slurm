#!/bin/bash
#SBATCH --job-name=feature_cpu_Af
#SBATCH --partition=cpu
#SBATCH -N 1
#SBATCH --output=task_file/%j_%x.out
#SBATCH --error=task_file/%j_%x.err
#SBATCH --exclusive


module load cuda/10.1.243-gcc-8.3.0
module load miniconda3
source activate alphafold_cuda10
cd /lustre/home/acct-stu/stu/alphafold
./run_alphafold_feature.sh -d data -o output -m model_1 -f input/test_3.fasta -t 2021-07-27
