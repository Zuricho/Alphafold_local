# How to set up for Alphafold2

Author: Bozitao Zhong



This is a collection of scripts for installing Alphafold2 on HPC, with GPUs only support cuda10.2 (cuda driver)

Overall Alphafold non docker git reference: https://github.com/kalininalab/alphafold_non_docker





## Conda Environment

Set up miniconda

```bash
module load miniconda3
source activate base
```



Create a miniconda environment for Alphafold

```bash
conda create -n alphafold_cuda10 python=3.8
conda activate alphafold_cuda10
```



## Tensorflow

Reference: https://docs.hpc.sjtu.edu.cn/app/tensorflow.html



Install cudatoolkit 10.2 and cudnn

```
conda install cudatoolkit=10.1 cudnn
```



Install tensorflow 2.3.0 by pip

```
pip install tensorflow==2.3.0
```



Then we can submit a job with python code like:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```



slurm script reference (test without test.py)

```bash
#!/bin/bash
#SBATCH --job-name=SCAgo_Af2
#SBATCH --partition=dgx2
#SBATCH -w vol08
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=task_file/%j_%x.out
#SBATCH --error=task_file/%j_%x.err

module load miniconda3
source activate alphafold

python -c 'import tensorflow as tf; \
           print(tf.__version__);   \
           print(tf.test.is_gpu_available());'
```



## Other Packages

Then install other packages with pip and conda

```bash
conda install -c conda-forge openmm pdbfixer
conda install -c bioconda hmmer hhsuite kalign2
```

package version: 

- openmm 7.5.1, pdbfixer 1.7
- hmmer 3.3.2, hhsuite 3.3.0, kalign2 2.0.4



Here you can use `pip list` or `conda list` to check some installed packages



```bash
pip install biopython==1.79 chex==0.0.7 dm-haiku==0.0.4 dm-tree==0.1.6 immutabledict==2.0.0 jax==0.2.14 ml-collections==0.1.0
pip install --upgrade "jax[cuda101]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

package version: 

- biopython 1.79, chex 0.0.7, dm-haiku 0.0.4 dm-tree 0.1.6, immutabledict 2.0.0, jax 0.2.14, ml-collections 0.1.0



jax installation reference: https://github.com/google/jax

- For CUDA 11.1, 11.2, or 11.3, use `cuda111`.
- For CUDA 11.0, use `cuda110`.
- For CUDA 10.2, use `cuda102`.
- For CUDA 10.1, use `cuda101`.

Here I used cuda 10.1



## Apply OpenMM patch

```bash
alphafold_path="/lustre/home/acct-stu/stu/alphafold"
cd ~/.conda/envs/alphafold_cuda10/lib/python3.8/site-packages/
patch -p0 < $alphafold_path/docker/openmm.patch
```





## If you need cuda

Available modules: `cuda/10.1.243-gcc-8.3.0`, `cuda/10.2.89-gcc-8.3.0`



## Change `sh` file

If your `run_alphafold.sh` file cannot find gpu, you can try this:

Change this in `run_alphafold.sh`

```python
# Export ENVIRONMENT variables and set CUDA devices for use
if [[ "$use_gpu" == true ]] ; then
    export CUDA_VISIBLE_DEVICES=0

    if [[ "$gpu_devices" ]] ; then
        export CUDA_VISIBLE_DEVICES=$gpu_devices
    fi
fi
```

to this:

```python

# Export ENVIRONMENT variables and set CUDA devices for use
if [[ "$use_gpu" == true ]] ; then
    export CUDA_VISIBLE_DEVICES=0

    if [[ "$gpu_devices" ]] ; then
        export CUDA_VISIBLE_DEVICES=0
    fi
fi
```

This might help...



## Final job submission

```bash
#!/bin/bash
#SBATCH --job-name=test_Af2
#SBATCH --partition=dgx2
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --output=task_file/%j_%x.out
#SBATCH --error=task_file/%j_%x.err

export 

module load cuda/10.1.243-gcc-8.3.0
module load miniconda3
source activate alphafold_cuda10
cd /lustre/home/acct-stu/stu/alphafold
./run_alphafold.sh -d data -o output -m model_1,model_2,model_3,model_4,model_5 -f input/test.fasta -t 2021-07-27


```

