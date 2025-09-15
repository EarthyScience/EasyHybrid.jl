#!/bin/bash
#SBATCH --job-name=fluxnet_hybrid
#SBATCH --output=fluxnet_hybrid.out
#SBATCH --error=fluxnet_hybrid.err
#SBATCH -p work
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12                 # <- how many *local* worker procs + 1 master you want to allow
#SBATCH --mem=24G                          # total memory; 6G was tight for CairoMakie + workers
#SBATCH --time=04:00:00

# Avoid BLAS over-subscription when you also spawn worker processes
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

ml purge
ml proxy
ml git-annex
ml gnu12/12.2.0
ml openmpi4/4.1.4
unset LD_LIBRARY_PATH
ml julia/1.11.4
ml --ignore-cache gdal/3.5.3

echo "CPU threads: ${JULIA_NUM_THREADS}"
echo "SLURM_NODELIST: ${SLURM_NODELIST}"
echo "Starting precompilation/instantiation…"

# 1) Make sure the project is instantiated (first run on the cluster, or when deps change)
julia --heap-size-hint=16G -e '
using Pkg; 
project_path = "projects/Respiration_Fluxnet"; 
Pkg.activate(project_path); 
Pkg.instantiate(); 
Pkg.precompile();
'

echo "Running training script…"
# 2) Run your script (it activates its own project)
julia --heap-size-hint=16G projects/Respiration_Fluxnet/distributed.jl