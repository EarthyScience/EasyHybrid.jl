# # LSTM Hybrid Model with EasyHybrid.jl
#
# This tutorial demonstrates how to use EasyHybrid to train a hybrid model with LSTM
# neural networks on synthetic data for respiration modeling with Q10 temperature sensitivity.
# The code for this tutorial can be found in [docs/src/literate/tutorials](https://github.com/EarthyScience/EasyHybrid.jl/tree/main/docs/src/literate/tutorials/) => example_synthetic_lstm.jl.
#
# ## 1. Load Packages

# Set project path and activate environment

using EasyHybrid
using AxisKeys
using DimensionalData
using Lux

# ## 2. Data Loading and Preprocessing

# Load synthetic dataset from GitHub - it's tabular data
df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc");

# Select a subset of data for faster execution
df = df[1:20000, :];
first(df, 5);

# ## 3. Define Neural Network Architectures

# Define a standard feedforward neural network
NN = Chain(Dense(15, 15, Lux.sigmoid), Dense(15, 15, Lux.sigmoid), Dense(15, 1))

# Define LSTM-based neural network with memory
#
# ::: tip
#
# When the `Chain` ends with a Recurrence layer, EasyHybrid automatically adds
# a `RecurrenceOutputDense` layer to handle the sequence output format.
# The user only needs to define the Recurrence layer itself!
#
# :::
#

NN_Memory = Chain(
    Recurrence(LSTMCell(15 => 15), return_sequence = true),
)

# ## 4. We define the process-based model, a classical Q10 model for respiration

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
nothing # hide

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

# `:KeyedArray` and `:DimArray` are supported
pref_array_type = :DimArray
x, y = prepare_data(hlstm, df, array_type = pref_array_type);

# Convert a (single) time series into *many* training samples by `windowing`.
#
# Each sample consists of:
#   - `input_window`: number of past steps given to the model (sequence length)
#   - `output_window`: number of steps to predict
#   - `output_shift`: stride between consecutive windows (controls overlap)
#   - `lead_time`: prediction lead (e.g. lead_time=1 predicts starting 1 step ahead)
#
# This supports `many-to-one` / `many-to-many` forecasting depending on output_window.
# Creates an array of shape (`variable`, `time`, `batch_size`) with variable being feature, time the input window, and `batch_size` 1:n samples (`= full batch`)

output_shift = 1
output_window = 1
input_window = 10
xs, ys = split_into_sequences(x, y; input_window = input_window, output_window = output_window, output_shift = output_shift, lead_time = 0);
ys_nan = .!isnan.(ys);

# First input_window/sample
xs[:, :, 1]
# Second input_window/sample
xs[:, :, 2]

# test of shift
xs[:, output_shift + 1, 1] == xs[:, 1, 2]


# First `output_window/sample` with time label like `:x30_to_x5_y4` which indicates an `accumulation` of memory from `x30 to x5` for the prediction of `y4`
ys[:, :, 1]
# Second output_window/sample
ys[:, :, 2]

# Any of the first output_window the same as the second output_window?
# ideally not big overlap
overlap = output_window - output_shift
overlap_length = sum(in(ys[:, :, 1]), ys[:, :, 2])

# Split the (windowed) dataset into train/validation in the same way as `train` does.
sdf = split_data(df, hlstm, sequence_kwargs = (; input_window = input_window, output_window = output_window, output_shift = output_shift, lead_time = 0), array_type = pref_array_type);

(x_train, y_train), (x_val, y_val) = sdf;
x_train
y_train
y_train_nan = .!isnan.(y_train)

# Wrap the training windows/samples in a `DataLoader` to form batches.
#
# ::: warning
#
# `batchsize` is the number of windows/samples used per gradient step to update the parameters.
# Processing 32 windows in one array is usually much faster than doing 32 separate forward/backward passes with batch_size=1.
#
# :::
#

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
    nepochs = 100,           # Number of training epochs
    batchsize = 128,         # Batch size of training windows/samples
    opt = RMSProp(0.01),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = true,
    training_loss = :nseLoss,
    loss_types = [:nse],
    sequence_kwargs = (; input_window = input_window, output_window = output_window, output_shift = output_shift, lead_time = 0),
    plotting = false,
    show_progress = false,
    input_batchnorm = false,
    array_type = pref_array_type
);

out_lstm.val_obs_pred

# ## 10. Train Single NN Hybrid Model (Optional)

# For comparison, we can also train a hybrid model with a standard feed-forward neural network
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
);

# Train the hybrid model
single_nn_out = train(
    hm,
    df,
    ();
    nepochs = 100,           # Number of training epochs
    batchsize = 128,         # Batch size for training
    opt = RMSProp(0.01),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = true,
    training_loss = :nseLoss,
    loss_types = [:nse],
    array_type = :DimArray,
    plotting = false,
    show_progress = false,
);

# Close enough
out_lstm.best_loss
single_nn_out.best_loss
