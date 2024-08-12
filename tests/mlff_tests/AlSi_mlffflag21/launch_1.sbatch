#!/bin/bash
#SBATCH -JMLFF_train_spacegroup
#SBATCH --account=hive-phanish6
#SBATCH -p hive
#SBATCH -N 2 --ntasks-per-node=24
#SBATCH --mem-per-cpu=7G
#SBATCH -t01:00:00
cd $SLURM_SUBMIT_DIR


module load gcc
module load mkl
echo $PWD
srun  /storage/hive/project/ce-suryanarayana/skumar416/MLFF_MERGE_MAIN/dev_merge/dev_SPARC/tests/sparc -name AlSi_mlffflag21 -log_summary > AlSi_mlffflag21.log

