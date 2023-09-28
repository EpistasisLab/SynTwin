#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=breast_survival_1000samples_mpom
#SBATCH -p defq
#SBATCH -t 0
#SBATCH --mem=10GB
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=<user_email>
#SBATCH --array=0-999%40

source path_to_conda/anaconda3/profile.d/conda.sh
conda activate env

python mpom_breast_survival_2010-2015_1000samples.py $SLURM_ARRAY_TASK_ID \