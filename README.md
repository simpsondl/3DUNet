# 3DUNet
Final project for ELE571

# Creating a working environment on Adroit

Using a GPU for Tensorflow calculations is a huge advantage, but takes effort to correctly configure an environment to use on Princeton's Adroit cluster. Use the following steps to create a working environment (as of 5/24/2019).

### Create conda environment with tensorflow available

```
conda create -n py36 python=3.6
conda activate py36
pip install tensorflow-gpu keras
conda deactivate py36
```

### Add these lines to R scripts

```
library(keras)
use_condaenv("py36")
```

### Add these lines to SLURM scripts

```
module load anaconda3
module load cudatoolkit/10.0
module load cudnn/cuda-10.0/7.5.0

conda activate py36
```

### Ensure a GPU node is requested by adding this directive to SLURM scripts
```
#SBATCH --gres=gpu:1
```
