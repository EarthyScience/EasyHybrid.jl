# [![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
#
# # EasyHybrid Example: Synthetic Data Analysis
#
# This example demonstrates how to use EasyHybrid to train a hybrid model
# on synthetic data for respiration modeling with Q10 temperature sensitivity.
#

using EasyHybrid
using Metal
## using CUDA
using Lux

# ## Data Loading and Preprocessing
#
# Load synthetic dataset from GitHub into DataFrame

df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc");

# Select a subset of data for faster execution
df = df[1:5000, :];
nothing #hide
first(df, 5)

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

single_nn_hybrid_model = constructHybridModel(
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

# ### train on DataFrame
# Train the hybrid model

cfg = EasyHybrid.TrainConfig(
    nepochs = 20,
    batchsize = 64,
    opt = RMSProp(0.01),
    loss_types = [:mse, :nse],
    show_progress = false,
    keep_history = false, # set to true to keep per-epoch history, losses, predictions, etc.
    save_training = false, # Set to true to enable saving training history and checkpoints
)

using BenchmarkTools

gpu_small_nn() = tune(
    single_nn_hybrid_model, df, cfg;
    gdev = gpu_device(), model_name = "small_nn_gpu"
)

cpu_small_nn() = tune(
    single_nn_hybrid_model, df, cfg;
    gdev = cpu_device(), model_name = "small_nn_cpu"
)

gpu_large_nn() = tune(
    single_nn_hybrid_model, df, cfg;
    gdev = gpu_device(), model_name = "large_nn_gpu", hidden_layers = [512, 256, 128, 64, 32]
)

cpu_large_nn() = tune(
    single_nn_hybrid_model, df, cfg;
    gdev = cpu_device(), model_name = "large_nn_cpu", hidden_layers = [512, 256, 128, 64, 32]
)

# warm-up to pay compilation once
gpu_small_nn()
cpu_small_nn();
gpu_large_nn();
cpu_large_nn();
nothing # hide

# # With Small NN CPU and GPU are on par
using BenchmarkTools
# Small NN on GPU
@benchmark gpu_small_nn() samples = 4 evals = 1
# Small NN on CPU
@benchmark cpu_small_nn() samples = 4 evals = 1

# # With Large NN CPU is slower than GPU
# Large NN on GPU
@benchmark gpu_large_nn() samples = 4 evals = 1
# Large NN on CPU
@benchmark cpu_large_nn() samples = 4 evals = 1
