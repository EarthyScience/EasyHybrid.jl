#!/bin/bash
#SBATCH --job-name=q10_fluxnet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=01:10:00
#SBATCH --array=0-331  # 332 jobs(sites)!
#SBATCH -o /Net/Groups/BGI/tscratch/lalonso/slurm_output_EasyHybrid/q10_fluxnet_la-%A_%a.out

module load proxy
module load julia/1.10.10

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

id=$SLURM_ARRAY_TASK_ID
echo "Running with: id=$id"
# Run the program with calculated parameters
julia --project --heap-size-hint=16G q10_slurm.jl $id