# CC BY-SA 4.0
# =============================================================================
# EasyHybrid Example: Synthetic Data Analysis
# =============================================================================
# This example demonstrates how to use EasyHybrid to train a hybrid model
# on synthetic data for respiration modeling with Q10 temperature sensitivity.
# =============================================================================

# =============================================================================
# Project Setup and Environment
# =============================================================================
using Pkg

# Set project path and activate environment
project_path = "docs"
Pkg.activate(project_path)
EasyHybrid_path = joinpath(pwd())
Pkg.develop(path = EasyHybrid_path)
#Pkg.resolve()
#Pkg.instantiate()

using EasyHybrid
using AxisKeys
using DimensionalData
using Lux

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
# Load synthetic dataset from GitHub into DataFrame
df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc")

# Select a subset of data for faster execution
df = df[1:20000, :]

# =============================================================================
# neural network model
# =============================================================================
# Define neural network
NN = Chain(Dense(15, 15, Lux.sigmoid), Dense(15, 15, Lux.sigmoid), Dense(15, 1))

broadcast_layer4 = @compact(; layer = Dense(6=>6)) do x::Union{NTuple{<:AbstractArray}, AbstractVector{<:AbstractArray}}
    y = map(layer, x)
    @return permutedims(stack(y; dims=3), (1, 3, 2))
end

NN_Memory = Chain(
    Recurrence(LSTMCell(6 => 6), return_sequence=true),
    broadcast_layer4
)

# =============================================================================
# Define the Physical Model
# =============================================================================
# RbQ10 model: Respiration model with Q10 temperature sensitivity
# Parameters:
#   - ta: air temperature [°C]
#   - Q10: temperature sensitivity factor [-]
#   - rb: basal respiration rate [μmol/m²/s]
#   - tref: reference temperature [°C] (default: 15.0)
function RbQ10(; ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end

# =============================================================================
# Define Model Parameters
# =============================================================================
# Parameter specification: (default, lower_bound, upper_bound)
parameters = (
    # Parameter name | Default | Lower | Upper      | Description
    rb = (3.0f0, 0.0f0, 13.0f0),  # Basal respiration [μmol/m²/s]
    Q10 = (2.0f0, 1.0f0, 4.0f0),   # Temperature sensitivity factor [-]
)

# =============================================================================
# Configure Hybrid Model Components
# =============================================================================
# Define input variables
forcing = [:ta]                    # Forcing variables (temperature)

# Target variable
target = [:reco]                   # Target variable (respiration)

# Parameter classification
global_param_names = [:Q10]        # Global parameters (same for all samples)
neural_param_names = [:rb]         # Neural network predicted parameters

# =============================================================================
# Single NN Hybrid Model Training
# =============================================================================
using GLMakie
# Create single NN hybrid model using the unified constructor
predictors = [:sw_pot, :dsw_pot]   # Predictor variables (solar radiation, and its derivative)

hlstm = constructHybridModel(
    predictors,              # Input features
    forcing,                 # Forcing variables
    target,                  # Target variables
    RbQ10,                  # Process-based model function
    parameters,              # Parameter definitions
    neural_param_names,      # NN-predicted parameters
    global_param_names,      # Global parameters
    hidden_layers = NN_Memory, # Neural network architecture
    scale_nn_outputs = true, # Scale neural network outputs
    input_batchnorm = false   # Apply batch normalization to inputs
)

# =================================================================================
# show steps for data preparation, happens under the hood in the end.

# two KeyedArrays
x, y = prepare_data(hlstm, df)

# new split_into_sequences with input_window, output_window, shift and lead_time
# for many-to-one, many-to-many, and different prediction lead times and overlap
xs, ys = split_into_sequences(x, y; input_window = 20, output_window = 2, shift=1, lead_time=0)
ys_nan = .!isnan.(ys)

# split data as in train
sdf = split_data(df, hlstm, sequence_kwargs = (;input_window = 10, output_window = 3, shift = 1, lead_time = 1));

typeof(sdf)
(x_train, y_train), (x_val, y_val) = sdf;
x_train
y_train
y_train_nan = .!isnan.(y_train)

# put into train loader to compose minibatches
train_dl = EasyHybrid.DataLoader((x_train, y_train); batchsize=32)

# run hybrid model forwards
x_first = first(train_dl)[1]
y_first = first(train_dl)[2]

ps, st = Lux.setup(Random.default_rng(), hlstm)
frun = hlstm(x_first, ps, st)

# extract predicted yhat
reco_mod = frun[1].reco

# bring observations in same shape
reco_obs = dropdims(y_first, dims = 1)
reco_nan = .!isnan.(reco_obs)

# simulate loss -> pick the right window
reco_mod(window = axiskeys(reco_obs, :window)) .- reco_obs

# compute loss
EasyHybrid.compute_loss(hlstm, ps, st, (x_train, (y_train, y_train_nan)), logging = LoggingLoss(train_mode = true))

# Zygote gradient of loss

# =============================================================================
# train on DataFrame
# =============================================================================

out_lstm = train(
    hlstm,
    df,
    ();
    nepochs = 100,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = false,
    loss_types = [:mse, :nse],
    sequence_kwargs = (;input_window = 10, output_window = 4),
    plotting = false
)



#####################################################################################
# is neural network still running?

hm = constructHybridModel(
    predictors,              # Input features
    forcing,                 # Forcing variables
    target,                  # Target variables
    RbQ10,                  # Process-based model function
    parameters,              # Parameter definitions
    neural_param_names,      # NN-predicted parameters
    global_param_names,      # Global parameters
    hidden_layers = NN, # Neural network architecture
    scale_nn_outputs = true, # Scale neural network outputs
    input_batchnorm = false   # Apply batch normalization to inputs
)


# Train the hybrid model
single_nn_out = train(
    hm,
    df,
    ();
    nepochs = 3,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = false,
    loss_types = [:mse, :nse],
)