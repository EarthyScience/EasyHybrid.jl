# [![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
#
# # EasyHybrid Example: Synthetic Data Analysis
#
# This example demonstrates how to use EasyHybrid to train a hybrid model
# on synthetic data for respiration modeling with Q10 temperature sensitivity.
#

## for development, use local setup
# include("../../setup_local_docsrun.jl")

using EasyHybrid
using MLDataDevices

# Load whichever GPU backend is available on this machine (CUDA / AMDGPU /
# Metal / oneAPI), or fall back to CPU. Defines `GPU_BACKEND_PKG` and
# `GPU_DEVICE_TYPE` in the caller's scope.
include("../../setup_gpu_backend.jl")

cpu_device() isa CPUDevice
gpu_device() isa GPU_DEVICE_TYPE

# ## Data Loading and Preprocessing
#
# Load synthetic dataset from GitHub into DataFrame

df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc");

# Select a subset of data for faster execution
#df = df[1:20000, :];
#nothing #hide
#first(df, 5)

# ## Define the Physical Model
#
# **RbQ10 model**: Respiration model with Q10 temperature sensitivity
#
# Parameters:
#   - ta: air temperature [°C]
#   - Q10: temperature sensitivity factor [-]
#   - rb: basal respiration rate [μmol/m²/s]
#   - tref: reference temperature [°C] (default: 15.0)

function RbQ10(; ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end

# ### Define Model Parameters
#
# Parameter specification: (default, lower_bound, upper_bound)

parameters = (
    ## Parameter name | Default | Lower | Upper      | Description
    rb = (3.0f0, 0.0f0, 13.0f0),  # Basal respiration [μmol/m²/s]
    Q10 = (2.0f0, 1.0f0, 4.0f0),   # Temperature sensitivity factor [-]
)

# ## Configure Hybrid Model Components
#
# Define input variables
forcing = [:ta]                    # Forcing variables (temperature)

# Target variable
target = [:reco]                   # Target variable (respiration)

# Parameter classification
global_param_names = [:Q10]        # Global parameters (same for all samples)
neural_param_names = [:rb]         # Neural network predicted parameters

# ## Single NN Hybrid Model Training
predictors_single_nn = [:sw_pot, :dsw_pot]   # Predictor variables (solar radiation, and its derivative)

small_nn_hybrid_model = constructHybridModel(
    predictors_single_nn,              # Input features
    forcing,                 # Forcing variables
    target,                  # Target variables
    RbQ10,                  # Process-based model function
    parameters,              # Parameter definitions
    neural_param_names,      # NN-predicted parameters
    global_param_names,      # Global parameters
    hidden_layers = [16, 16], # Neural network architecture
    activation = sigmoid,      # Activation function
    scale_nn_outputs = true, # Scale neural network outputs
    input_batchnorm = true   # Apply batch normalization to inputs
)

large_nn_hybrid_model = constructHybridModel(
    predictors_single_nn,              # Input features
    forcing,                 # Forcing variables
    target,                  # Target variables
    RbQ10,                  # Process-based model function
    parameters,              # Parameter definitions
    neural_param_names,      # NN-predicted parameters
    global_param_names,      # Global parameters
    hidden_layers = [512, 512, 512], # Neural network architecture
    activation = sigmoid,      # Activation function
    scale_nn_outputs = true, # Scale neural network outputs
    input_batchnorm = true   # Apply batch normalization to inputs
)

# ### train on DataFrame
# Train the hybrid model

cfg = EasyHybrid.TrainConfig(
    nepochs = 10,
    batchsize = 512,
    opt = Adam(0.001),
    loss_types = [:mse, :nse],
    show_progress = false,
    keep_history = false, # set to true to keep per-epoch history, losses, predictions, etc.
    save_training = false, # Set to true to enable saving training history and checkpoints
    plotting = false,
)

using Suppressor
using Printf

function _pure(s)
    return s.time - s.gctime - s.compile_time - s.recompile_time
end

function _warn_overhead(tag, label, s, warn_frac)
    gc_frac        = s.gctime         / s.time
    compile_frac   = s.compile_time   / s.time
    recompile_frac = s.recompile_time / s.time
    overhead_frac  = gc_frac + compile_frac + recompile_frac
    if overhead_frac > warn_frac
        pct(x) = round(100 * x; digits = 1)
        @warn """[$label/$tag] $(pct(overhead_frac))% of elapsed time was overhead:
                 gc=$(pct(gc_frac))% ($(round(s.gctime; digits=3))s), \
                 compile=$(pct(compile_frac))% ($(round(s.compile_time; digits=3))s), \
                 recompile=$(pct(recompile_frac))% ($(round(s.recompile_time; digits=3))s).
                 Elapsed ratio will be misleading; prefer the pure ratio."""
    end
end

# Benchmark `tune` on CPU vs GPU. Uses `@timed` so we can separate out
# compile/recompile/GC overhead from steady-state work, and reports BOTH:
#   - elapsed ratio (wall clock CPU / wall clock GPU)
#   - pure ratio    (elapsed minus GC/compile/recompile, CPU/GPU)
#
# `warn_frac` is the fraction of total time spent in GC / (re)compile above
# which a warning is emitted for that run; defaults to 5%.
function bench_cpu_vs_gpu(model, df, cfg; label::AbstractString, warmup::Bool = true, warn_frac::Real = 0.05, tune_kwargs...)
    # A short 1-epoch run on each device triggers Julia/Zygote specialization
    # and GPU kernel compilation, so the timed runs below measure steady-state
    # training rather than first-call compile latency.
    if warmup
        warm_cfg = EasyHybrid.TrainConfig(;
            nepochs = 1,
            batchsize = cfg.batchsize,
            opt = cfg.opt,
            show_progress = false,
            save_training = false,
            keep_history = false,
            plotting = false,
        )
        @suppress tune(model, df, warm_cfg; gdev = cpu_device(), tune_kwargs...);
        @suppress tune(model, df, warm_cfg; gdev = gpu_device(), tune_kwargs...);
    end

    cpu_stats = @suppress @timed tune(model, df, cfg; gdev = cpu_device(), tune_kwargs...);
    gpu_stats = @suppress @timed tune(model, df, cfg; gdev = gpu_device(), tune_kwargs...);

    _warn_overhead("CPU", label, cpu_stats, warn_frac)
    _warn_overhead("GPU", label, gpu_stats, warn_frac)

    cpu_pure = _pure(cpu_stats)
    gpu_pure = _pure(gpu_stats)
    elapsed_ratio = cpu_stats.time / gpu_stats.time
    pure_ratio    = cpu_pure      / gpu_pure

    @printf("[%s] CPU: total=%.3fs compile=%.3fs recompile=%.3fs gc=%.3fs pure=%.3fs\n",
            label, cpu_stats.time, cpu_stats.compile_time, cpu_stats.recompile_time, cpu_stats.gctime, cpu_pure)
    @printf("[%s] GPU: total=%.3fs compile=%.3fs recompile=%.3fs gc=%.3fs pure=%.3fs\n",
            label, gpu_stats.time, gpu_stats.compile_time, gpu_stats.recompile_time, gpu_stats.gctime, gpu_pure)
    @printf("[%s] with GPU we get: elapsed=%.2fx  pure=%.2fx\n",
            label, elapsed_ratio, pure_ratio)

    # Return a slim summary. We deliberately drop `.value` (the full TrainResults)
    # and `.gcstats` so that REPL/Literate printing of the return value stays compact.
    slim(s) = (; s.time, s.bytes, s.gctime, s.compile_time, s.recompile_time)
    return (; cpu = slim(cpu_stats), gpu = slim(gpu_stats), elapsed_ratio, pure_ratio)
end

bench_cpu_vs_gpu(small_nn_hybrid_model, df, cfg; label = "small NN"); # on our gpu1-hpc22 0.57x
bench_cpu_vs_gpu(large_nn_hybrid_model, df, cfg; label = "large NN"); # on our gpu1-hpc22 2.73x

# ## Sweep: GPU speedup vs hidden-layer width
#
# Build a hybrid model for a range of hidden-layer widths (fixed depth of 3),
# benchmark each on CPU and GPU, and plot the elapsed and pure ratios.
# A ratio > 1 means GPU is faster than CPU; a ratio < 1 means GPU is slower.

widths = [16, 64, 256, 512, 1024]

results = map(widths) do w
    r = bench_cpu_vs_gpu(small_nn_hybrid_model, df, cfg;
        label = "w=$w", hidden_layers = [w, w, w])
    (; width = w, elapsed_ratio = r.elapsed_ratio, pure_ratio = r.pure_ratio,
       cpu_time = r.cpu.time, gpu_time = r.gpu.time)
end

using CairoMakie

fig = Figure(size = (720, 480))
ax = Axis(fig[1, 1];
    xlabel = "Hidden-layer width (depth = 3)",
    ylabel = "GPU speedup (CPU / GPU)",
    title  = "GPU speedup vs hidden-layer width",
    xscale = log2,
    xticks = (widths, string.(widths)),
)

ws        = [r.width for r in results]
elapsed_x = [r.elapsed_ratio for r in results]
pure_x    = [r.pure_ratio    for r in results]

hlines!(ax, [1.0]; color = :gray, linestyle = :dash, label = "CPU = GPU")
scatterlines!(ax, ws, elapsed_x; label = "elapsed", marker = :circle)
scatterlines!(ax, ws, pure_x;    label = "pure",    marker = :diamond)

axislegend(ax; position = :lt)
fig

