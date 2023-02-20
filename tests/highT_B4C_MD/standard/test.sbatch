#!/bin/bash
#SBATCH -J SQ                            # Job name
#SBATCH -A gts-phanish6-coda20                    # Account name
#SBATCH -p inferno                             # Partition
#SBATCH -N 2                               # Nodes
#SBATCH --ntasks-per-node=24                # Processors
#SBATCH --mem-per-cpu=7G                    # Memory per processor
#SBATCH -t 0-1                             # walltime (days-hours)
#SBATCH -o ./output  # log file name


export MV2_HOMOGENEOUS_CLUSTER=1

cd $SLURM_SUBMIT_DIR                            # Change to working directory

echo $PWD
echo "======================================================================="


srun ../../../../lib/sparc -name highT_B4C > log