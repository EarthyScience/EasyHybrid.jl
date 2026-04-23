# [![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
#
# # EasyHybrid Example: Synthetic Data Analysis
#
# This example demonstrates how to use EasyHybrid to train a hybrid model
# on synthetic data for respiration modeling with Q10 temperature sensitivity.
#

include("../../setup_local_docsrun.jl")

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
)

using BenchmarkTools
using Suppressor
using Printf

# Benchmark `tune` on CPU vs GPU and print the GPU speedup (or slowdown)
# relative to CPU. Uses a single sample because each run already does a full
# training loop (`cfg.nepochs` epochs), which is expensive.
function bench_cpu_vs_gpu(model, df, cfg; label::AbstractString, tune_kwargs...)
    t_cpu = @suppress @belapsed tune($model, $df, $cfg; gdev = cpu_device(), $tune_kwargs...) samples=1 evals=1
    t_gpu = @suppress @belapsed tune($model, $df, $cfg; gdev = gpu_device(), $tune_kwargs...) samples=1 evals=1

    ratio = t_cpu / t_gpu
    @printf("%-10s | CPU: %8.3f s | GPU: %8.3f s | with GPU we get %.2fx\n",
            label, t_cpu, t_gpu, ratio)
    return (; t_cpu, t_gpu, ratio)
end

bench_cpu_vs_gpu(small_nn_hybrid_model, df, cfg; label = "small NN") # on our gpu1-hpc22 0.57x
bench_cpu_vs_gpu(large_nn_hybrid_model, df, cfg; label = "large NN") # on our gpu1-hpc22 1.35x
