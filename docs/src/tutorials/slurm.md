## Slurm jobs

Slurm is an open source, fault-tolerant, and highly scalable cluster management and job scheduling system for large and small Linux clusters. See https://slurm.schedmd.com/overview.html for more information.

Here, we provide only a minimal script `q10_slurm.sh` to get you started!

::: code-group

```bash [q10_slurm.sh]
#!/bin/bash
#SBATCH --job-name=q10_fluxnet
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=00:10:00 # 10 minutes!
#SBATCH --array=0-4  # 5 jobs(experiments)!
#SBATCH -o /u/slurm_output/q10_fluxnet-%A_%a.out # create this folder `/u/slurm_output` in advance in your file system

module load proxy
module load julia/1.11.4

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

id=$SLURM_ARRAY_TASK_ID
echo "Running with: id=$id"
# Run the program with calculated parameters
julia --project --heap-size-hint=16G q10_slurm.jl $id
```

```julia [q10_slurm.jl]
# julia script
using EasyHybrid
slurm_array_id = Base.parse(Int, ARGS[1]) # get from command line argument
println("SLURM_ARRAY_ID = $slurm_array_id")
```

```sh [sbatch]
# submit your job to the cluster!
u@hpc-node:~/project $ sbatch q10_slurm.sh
```

:::