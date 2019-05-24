#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=100GB
#SBATCH -t 1:00:00
#SBATCH --mail-user=ds65@princeton.edu
#SBATCH --output=slurm-%j-stress-low-1-test001.out

module load fftw/gcc/openmpi-1.10.2/3.3.4
module load anaconda3
module load cudatoolkit/10.0
module load cudnn/cuda-10.0/7.5.0

conda activate py36

echo Using script $script
echo ...

Rscript $script
