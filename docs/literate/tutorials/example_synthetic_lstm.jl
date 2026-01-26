# # LSTM Hybrid Model with EasyHybrid.jl
#
# This tutorial demonstrates how to use EasyHybrid to train a hybrid model with LSTM
# neural networks on synthetic data for respiration modeling with Q10 temperature sensitivity.
# The code for this tutorial can be found in [docs/src/literate/tutorials](https://github.com/EarthyScience/EasyHybrid.jl/tree/main/docs/src/literate/tutorials/) => example_synthetic_lstm.jl.
#
# ## 1. Load Packages

# Set project path and activate environment
#using Pkg
#project_path = "docs"
#Pkg.activate(project_path)
#EasyHybrid_path = joinpath(project_path, "..")
#Pkg.develop(path = EasyHybrid_path)
#Pkg.resolve()
#Pkg.instantiate()

using EasyHybrid
using AxisKeys
using DimensionalData
using Lux

# ## 2. Data Loading and Preprocessing

# Load synthetic dataset from GitHub
df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc");

# Select a subset of data for faster execution
df = df[1:20000, :];
first(df, 5);

# ## 3. Define Neural Network Architectures

# Define a standard feedforward neural network
NN = Chain(Dense(15, 15, Lux.sigmoid), Dense(15, 15, Lux.sigmoid), Dense(15, 1))

# Define LSTM-based neural network with memory
# Note: When the Chain ends with a Recurrence layer, EasyHybrid automatically adds
# a RecurrenceOutputDense layer to handle the sequence output format.
# The user only needs to define the Recurrence layer itself!
NN_Memory = Chain(
    Recurrence(LSTMCell(15 => 15), return_sequence = true),
)

# ## 4. Define the Physical Model

"""
    RbQ10(; ta, Q10, rb, tref=15.0f0)

Respiration model with Q10 temperature sensitivity.

- `ta`: air temperature [°C]
- `Q10`: temperature sensitivity factor [-]
- `rb`: basal respiration rate [μmol/m²/s]
- `tref`: reference temperature [°C] (default: 15.0)
"""
function RbQ10(; ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end

# ## 5. Define Model Parameters

# Parameter specification: (default, lower_bound, upper_bound)
parameters = (
    rb = (3.0f0, 0.0f0, 13.0f0),  # Basal respiration [μmol/m²/s]
    Q10 = (2.0f0, 1.0f0, 4.0f0),   # Temperature sensitivity factor [-]
)

# ## 6. Configure Hybrid Model Components

# Define input variables
# Forcing variables (temperature)
forcing = [:ta]
# Predictor variables (solar radiation, and its derivative)
predictors = [:sw_pot, :dsw_pot]
# Target variable (respiration)
target = [:reco]

# Parameter classification
# Global parameters (same for all samples)
global_param_names = [:Q10]
# Neural network predicted parameters
neural_param_names = [:rb]

# ## 7. Construct LSTM Hybrid Model

# Create LSTM hybrid model using the unified constructor
hlstm = constructHybridModel(
    predictors,
    forcing,
    target,
    RbQ10,
    parameters,
    neural_param_names,
    global_param_names,
    hidden_layers = NN_Memory, # Neural network architecture
    scale_nn_outputs = true, # Scale neural network outputs
    input_batchnorm = false   # Apply batch normalization to inputs
)

# ## 8. Data Preparation Steps (Demonstration)

# The following steps demonstrate what happens under the hood during training.
# In practice, you can skip to Section 9 and use the `train` function directly.

# :KeyedArray and :DimArray are supported
pref_array_type = :DimArray
x, y = prepare_data(hlstm, df, array_type = pref_array_type);

# New split_into_sequences with input_window, output_window, shift and lead_time
# for many-to-one, many-to-many, and different prediction lead times and overlap
xs, ys = split_into_sequences(x, y; input_window = 20, output_window = 2, shift = 1, lead_time = 0);
ys_nan = .!isnan.(ys);
typeof(xs)

# Split data as in train
sdf = split_data(df, hlstm, sequence_kwargs = (; input_window = 10, output_window = 3, shift = 1, lead_time = 1), array_type = pref_array_type);

(x_train, y_train), (x_val, y_val) = sdf;
x_train
y_train
y_train_nan = .!isnan.(y_train)

# Put into train loader to compose minibatches
train_dl = EasyHybrid.DataLoader((x_train, y_train); batchsize = 32);

# Run hybrid model forwards
x_first = first(train_dl)[1]
y_first = first(train_dl)[2]

ps, st = Lux.setup(Random.default_rng(), hlstm);
frun = hlstm(x_first, ps, st);

# Extract predicted yhat
reco_mod = frun[1].reco

# Bring observations in same shape
reco_obs = dropdims(y_first, dims = 1)
reco_nan = .!isnan.(reco_obs);

# Compute loss
EasyHybrid.compute_loss(hlstm, ps, st, (x_train, (y_train, y_train_nan)), logging = LoggingLoss(train_mode = true))

# ## 9. Train LSTM Hybrid Model

out_lstm = train(
    hlstm,
    df,
    ();
    nepochs = 2,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = false,
    loss_types = [:mse, :nse],
    sequence_kwargs = (; input_window = 10, output_window = 4),
    plotting = false,
    array_type = pref_array_type
);

out_lstm.val_obs_pred

# ## 10. Train Single NN Hybrid Model (Optional)

# For comparison, we can also train a hybrid model with a standard feedforward neural network
hm = constructHybridModel(
    predictors,
    forcing,
    target,
    RbQ10,
    parameters,
    neural_param_names,
    global_param_names,
    hidden_layers = NN, # Neural network architecture
    scale_nn_outputs = true, # Scale neural network outputs
    input_batchnorm = false,   # Apply batch normalization to inputs
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
    array_type = :DimArray
);
