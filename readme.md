# How to set up for Alphafold2 locally

Author: Bozitao Zhong



This is a collection of scripts for installing Alphafold2 on SJTU HPC, with GPUs only support cuda10.2 (cuda driver). Hope this note can help you to install Alphafold on your local clusters.

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



Install cudatoolkit 10.1 and cudnn

```
conda install cudatoolkit=10.1 cudnn
```

> Why use cudatoolkit 10.1:
>
> - cudatoolkit supports tensorflow 2.3.0, while sometimes tensorflow can't find GPU when using cudatoolkit 10.2 



Install tensorflow 2.3.0 by pip

```
pip install tensorflow==2.3.0
```



Then we can submit a job with python code like this to detect **whether tensorflow can find GPU**:

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```



Another way to detect GPU: slurm script reference (test without test.py)

```bash
#!/bin/bash
#SBATCH --job-name=TF_test
#SBATCH --partition=dgx2
##SBATCH -w vol08
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



If tensorflow can find the GPU, this step is complete



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

Here you should used cuda 10.1 when you use cuda 10.1



## Download Alphafold from DeepMind repo

[git repo](https://github.com/deepmind/alphafold.git)

```
git clone https://github.com/deepmind/alphafold.git
alphafold_path="/path/to/alphafold/git/repo"
```



## Download chemical properties to the common folder

```
wget -q -P alphafold/alphafold/common/ https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
```

This step in necessary! Some guide missed this file.



## Apply OpenMM patch

```bash
# This is you path to your alphafold folder
alphafold_path="/lustre/home/acct-stu/stu/alphafold"
cd ~/.conda/envs/alphafold_cuda10/lib/python3.8/site-packages/
patch -p0 < $alphafold_path/docker/openmm.patch
```





## If you need cuda

Available modules: `cuda/10.1.243-gcc-8.3.0`, `cuda/10.2.89-gcc-8.3.0`

Using this kind of cuda you can skip `conda install cudatoolkit`



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

This step is to fix that sometimes jax may not find GPU. (I have this problem and I use this method to fix this, but this might disable multi-GPU jobs, you can try `export CUDA_VISIBLE_DEVICES=0,1`)





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

module load cuda/10.1.243-gcc-8.3.0
module load miniconda3
source activate alphafold_cuda10
cd /lustre/home/acct-stu/stu/alphafold
./run_alphafold.sh -d data -o output -m model_1,model_2,model_3,model_4,model_5 -f input/test.fasta -t 2021-07-27


```



# How to divide Alphafold2 into CPU part and GPU part

## Remove MSA and template searching from pipeline

I refer to this guide: https://gist.github.com/biochem-fan/1f80e09b6852640629bc4aad69a19cff

Put this file `run_alphafold_noMSA.py` into your working directory, and modify scripts in `run_alphafold.sh` from:

```python
alphafold_script="$current_working_dir/run_alphafold.py"
```

to:

```python
alphafold_script="$current_working_dir/run_alphafold_noMSA.py"
```

The modified `run_alphafold_noMSA.py` is in the `AlphafoldLab `folder 



Then we can detect if there is file `feature.pkl` in output folder.

- If exists, skip all MSA and template finding steps and continue model predictions
- If not exists, follow the normal pipeline (start from MSA steps)



Then we just need a new scripts to output `feature.pkl` using more CPUs and exit when finish output `feature.pkl`

## Use more CPU to run MSA faster

**THE CPU PART IS UNDER DEVELOPMENT**

