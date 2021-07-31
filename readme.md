# 本地使用Alphafold2指南

Author: Bozitao Zhong

本repo主要介绍如何将Alphafold安装在本地集群中（以SJTU HPC为例）。其实已经有一些成熟的用于本地实现Alphafold的方案，本方案的特点主要是安装在了cuda10.2版本的显卡中(V100)

**Under development**: 拆分MSA与模型预测的部分

本方案主要参考了：https://github.com/kalininalab/alphafold_non_docker



## 安装步骤

### Step1: Python environment

进入conda环境，新建一个环境用于Alphafold，python建议选用3.8版本

```bash
module load miniconda3
source activate base
conda create -n alphafold_cuda10 python=3.8
conda activate alphafold_cuda10
```



安装 cudatoolkit 10.1 和 cudnn

```
conda install cudatoolkit=10.1 cudnn
```

> 为什么选择10.1：选择10.2的cuda会报错，超过10.2一般不会兼容cuda driver为10.2的显卡（这个在nvidia-smi)看



使用pip安装tensorflow 2.3.0

```
pip install tensorflow==2.3.0
```

此时我们可以提交一个这样的脚本检测TensorFlow是否能找到GPU：

```python
import tensorflow as tf
print(tf.config.list_physical_devices("GPU"))
```



使用conda和pip安装其他依赖包

```bash
conda install -c conda-forge openmm pdbfixer
conda install -c bioconda hmmer hhsuite kalign2
```

package version: 

- openmm 7.5.1, pdbfixer 1.7
- hmmer 3.3.2, hhsuite 3.3.0, kalign2 2.0.4

这里可以使用 `pip list` 或 `conda list` 来检查那些包已经安装上

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



配置openmm环境

```bash
alphafold_path="/lustre/home/acct-stu/stu/alphafold"
cd ~/.conda/envs/alphafold_cuda10/lib/python3.8/site-packages/
patch -p0 < $alphafold_path/docker/openmm.patch
```



### Step2: 使用run_alphafold.sh文件

下载本repo中的`run_alphafold.sh`文件，放在Alphafold所处文件夹中





### Step3: Alphafold的任务提交脚本

这里以sbatch脚本为例

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

