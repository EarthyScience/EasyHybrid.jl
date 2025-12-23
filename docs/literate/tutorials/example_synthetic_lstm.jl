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

# KeyedArray from AxisKeys.jl works, but cannot handle DateTime type
dfnot = Float32.(df[!, Not(:time)])

ka = to_keyedArray(dfnot)

# DimensionalData
mat = Array(Matrix(dfnot)')
da = DimArray(mat, (Dim{:col}(Symbol.(names(dfnot))), Dim{:row}(1:size(dfnot, 1))))

# =============================================================================
# Split into sequences
# =============================================================================

# =============================================================================
# neural network model
# =============================================================================
# Define neural network
NN = Chain(Dense(15, 15, Lux.sigmoid), Dense(15, 15, Lux.sigmoid), Dense(15, 1))

broadcast_layer4 = @compact(; layer = Dense(6=>6)) do x::Union{NTuple{<:AbstractArray}, AbstractVector{<:AbstractArray}}
    y = map(layer, x)
    @return permutedims(stack(y; dims=3), (1, 3, 2))
    #@return y
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
predictors_single_nn = [:sw_pot, :dsw_pot]   # Predictor variables (solar radiation, and its derivative)

hlstm = constructHybridModel(
    predictors_single_nn,              # Input features
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


x, y = prepare_data(hlstm, df)

xs, ys = split_into_sequences(x, y; input_window = 20, output_window = 2, shift=1, lead_time=0)

ys_nan = .!isnan.(ys)

#? this sets up initial ps for the hybrid model version
rng = Random.default_rng(1234)
ps, st = Lux.setup(rng, hlstm)

train_dl = EasyHybrid.DataLoader((xs, ys); batchsize=5)

reco_obs_3D = first(train_dl)[2]
reco_nan_3D = .!isnan.(reco_obs_3D)
reco_obs = dropdims(reco_obs_3D, dims = 1)
reco_nan = .!isnan.(reco_obs)
x_obs = first(train_dl)[1]

sdf = hlstm(x_obs, ps, st)

reco_mod = sdf[1].reco
reco_mod(window = axiskeys(reco_obs, :window)) .- reco_obs

EasyHybrid.compute_loss(hlstm, ps, st, (x_obs, (reco_obs, reco_nan)), logging = LoggingLoss(train_mode = true))

# =============================================================================
# train on DataFrame
# =============================================================================

out_lstm = train(
    hlstm,
    df,
    ();
    nepochs = 10,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = false,
    loss_types = [:mse, :nse],
    sequence_kwargs = (;input_window = 10, output_window = 4),
)

# Train the hybrid model
single_nn_out = train(
    hm,
    df,
    ();
    nepochs = 10,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = false,
    loss_types = [:mse, :nse],
)

split_data(df, hm)

split_data(df, hm, sequence_kwargs = (;input_window = 10, output_window = 4))

#? this sets up initial ps for the hybrid model version
rng = Random.default_rng(1234)
ps, st = Lux.setup(rng, hm)

xdl = EasyHybrid.DataLoader(xs; batchsize=512)
ydl = EasyHybrid.DataLoader(ys; batchsize=512)



hm = constructHybridModel(
    predictors_single_nn,              # Input features
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

sdf = hm(first(xdl), ps, st)

mid1 =first(xdl)(RbQ10_Memory.forcing)
mid1[:,end,:]


o1, st1 = LuxCore.apply(RbQ10_Memory.NN, first(xdl)(RbQ10_Memory.predictors), ps.ps, st.st)


# Train the hybrid model
out_lstm = tune(
    hm,
    df;
    nepochs = 10,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = false,
    loss_types = [:mse, :nse],
    sequence_kwargs = (;input_window = 10, output_window = 4),
    hidden_layers = NN_Memory,
)