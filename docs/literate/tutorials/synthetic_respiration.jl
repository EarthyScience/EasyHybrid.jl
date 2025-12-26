# [![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
#
# # EasyHybrid Example: Synthetic Data Analysis
#
# This example demonstrates how to use EasyHybrid to train a hybrid model
# on synthetic data for respiration modeling with Q10 temperature sensitivity.
#

using EasyHybrid

# ## Data Loading and Preprocessing
#
# Load synthetic dataset from GitHub into DataFrame

df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc");

# Select a subset of data for faster execution
df = df[1:20000, :];
first(df, 5)

# ### KeyedArray
# KeyedArray from AxisKeys.jl works, but cannot handle DateTime type
dfnot = Float32.(df[!, Not(:time)]);

ka = to_keyedArray(dfnot);

# ### DimensionalData
using DimensionalData
mat = Array(Matrix(dfnot)')
da = DimArray(mat, (Dim{:col}(Symbol.(names(dfnot))), Dim{:row}(1:size(dfnot, 1))));

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
#
# using WGLMakie
# Create single NN hybrid model using the unified constructor

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

extra_loss = function (ŷ)
    return (; a = sum((5.0 .- ŷ.rb) .^ 2))
end

# Train the hybrid model
single_nn_out = train(
    single_nn_hybrid_model,
    df,
    ();
    nepochs = 10,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = true,
    loss_types = [:mse, :nse],
    show_progress = false,
    extra_loss = extra_loss
)

# ### train on KeyedArray

single_nn_out = train(
    single_nn_hybrid_model,
    ka,
    ();
    nepochs = 10,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    show_progress = false,
    shuffleobs = true
)

# ### train on DimensionalData

single_nn_out = train(
    single_nn_hybrid_model,
    da,
    ();
    nepochs = 10,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    show_progress = false,
    shuffleobs = true
)

LuxCore.testmode(single_nn_out.st)
mean(df.dsw_pot)
mean(df.sw_pot)

# ## Multi NN Hybrid Model Training

predictors_multi_nn = (rb = [:sw_pot, :dsw_pot],)

multi_nn_hybrid_model = constructHybridModel(
    predictors_multi_nn,              # Input features
    forcing,                 # Forcing variables
    target,                  # Target variables
    RbQ10,                  # Process-based model function
    parameters,              # Parameter definitions
    global_param_names,      # Global parameters
    hidden_layers = [16, 16], # Neural network architecture
    activation = sigmoid,      # Activation function
    scale_nn_outputs = true, # Scale neural network outputs
    input_batchnorm = true   # Apply batch normalization to inputs
)

multi_nn_out = train(
    multi_nn_hybrid_model,
    ka,
    ();
    nepochs = 10,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    show_progress = false,
    shuffleobs = true
)

LuxCore.testmode(multi_nn_out.st)
mean(df.dsw_pot)
mean(df.sw_pot)

# ## Pure ML Single NN Model Training

# TODO does not train well build on top of SingleNNHybridModel
predictors_single_nn_ml = [:sw_pot, :dsw_pot, :ta]

single_nn_model = constructNNModel(predictors_single_nn_ml, target; input_batchnorm = true, activation = tanh)
single_nn_out = train(single_nn_model, da, (); nepochs = 10, batchsize = 512, opt = AdamW(0.01), yscale = identity, shuffleobs = true, show_progress = false)
LuxCore.testmode(single_nn_out.st)
mean(df.dsw_pot)
mean(df.sw_pot)

single_nn_model.targets

# ## Pure ML Multi NN Model Training

# TODO does not train well build on top of MultiNNHybridModel
