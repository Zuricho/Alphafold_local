#!/bin/bash
#SBATCH --job-name=test_Af2
#SBATCH --partition=dgx2
##SBATCH -w vol08
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=task_file/%j_%x.out
#SBATCH --error=task_file/%j_%x.err

module load cuda/10.1.243-gcc-8.3.0
module load miniconda3
source activate alphafold_cuda10
cd /lustre/home/acct-stu/stu/alphafold
./run_alphafold.sh -d data -o output -m model_1,model_2,model_3,model_4,model_5 -f input/test.fasta -t 2021-07-27

