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
da = DimArray(mat, (Dim{:variable}(Symbol.(names(dfnot))), Dim{:batch_size}(1:size(dfnot, 1))));

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

function rb_prior_extra_loss(ŷ, ps)
    return (; a = sum((5.0 .- ŷ.rb) .^ 2))
end

# Train the hybrid model
single_nn_out = train(
    single_nn_hybrid_model,
    df,
    ();
    nepochs = 100,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = true,
    loss_types = [:mse, :nse],
    show_progress = false,
    extra_loss = rb_prior_extra_loss,
    model_name = "RbQ10_synthetic1"
)

# ## Learning the observation noise σ (Gaussian NLL)
#
# To fit a Gaussian negative log-likelihood
# ``\tfrac12 \sum_i (r_i/σ_i)^2 + \log σ_i`` we need a *learned* noise scale ``σ``.
# The mechanistic model needs **no changes at all**: `σ` is just declared as a
# parameter. The framework only forwards to the mechanistic model the kwargs it
# declares, so `σ` (which `RbQ10` does not accept) is skipped there, yet it is
# still optimized and exposed to the loss.
#
# Full-context training losses use the signature
# `loss(ŷ, y, y_nan, ps, targets, parameters)` (auto-detected — just pass such a
# function to `training_loss`). The last argument, `parameters`, is `ŷ.parameters`:
# all model parameters (NN-predicted, global and fixed). We read `σ` from there.
# Parameters the mechanistic model doesn't consume (like `σ`) are also surfaced as
# top-level fields of `ŷ`, so they can be listed in `monitor_names` for plots.

parameters_σ = (
    rb = (3.0f0, 0.0f0, 13.0f0),   # Basal respiration [μmol/m²/s]
    Q10 = (2.0f0, 1.0f0, 4.0f0),    # Temperature sensitivity factor [-]
    sigma = (1.0f0, 0.01f0, 5.0f0), # Learned noise scale (bounds keep σ > 0)
)

# One loss for both cases: `σ` comes from `parameters.sigma` — a single value
# (global parameter) or one value per observation (NN output). Min–max scaling of
# global/NN outputs keeps it positive.
#
# It is deliberately written type-stably, which is what makes it usable with Enzyme
# as well (see the *Autodiff backends* section at the end).
function gaussian_nll(ŷ, y, y_nan, ps, targets, parameters)
    mask = y_nan.reco
    r = ŷ.reco[mask] .- y.reco[mask]
    σ = length(parameters.sigma) == 1 ?
        fill(parameters.sigma[1], length(r)) : parameters.sigma[mask]
    return sum(@. 0.5f0 * (r / σ)^2 + log(σ))
end

# ### Per-target σ: σ is a global parameter (one value, read from `parameters`)
# We reuse the unchanged `RbQ10` model defined above.

model_σ_global = constructHybridModel(
    predictors_single_nn,
    forcing,
    target,
    RbQ10,
    parameters_σ,
    [:rb],           # NN-predicted parameters
    [:Q10, :sigma],  # σ is a global (per-target) learned parameter
    hidden_layers = [16, 16],
    activation = sigmoid,
    scale_nn_outputs = true,
    input_batchnorm = true
)

nll_global_out = train(
    model_σ_global,
    df;
    nepochs = 100,
    batchsize = 512,
    opt = AdamW(0.1),
    monitor_names = [:rb, :Q10, :sigma], # σ is surfaced top-level in `ŷ` (and in `ŷ.parameters`)
    yscale = identity,
    shuffleobs = true,
    loss_types = [:mse, :nse],   # metrics for logging (still masked f(ŷ, y))
    training_loss = gaussian_nll, # full-context loss, auto-detected
    show_progress = false,
    model_name = "RbQ10_nll_global"
)

# ### Per-observation σ: σ is predicted by the neural network (one value per obs)
#
# Only the construction changes — `sigma` moves from the global parameters to the
# NN-predicted ones, so `parameters.sigma` becomes a heteroscedastic
# per-observation vector aligned with the predictions. The model (`RbQ10`) and the
# loss are exactly the same as above.

model_σ_nn = constructHybridModel(
    predictors_single_nn,
    forcing,
    target,
    RbQ10,
    parameters_σ,
    [:rb, :sigma],   # NN predicts both rb and a per-observation σ
    [:Q10],          # global parameters
    hidden_layers = [16, 16],
    activation = sigmoid,
    scale_nn_outputs = true,
    input_batchnorm = true
)

nll_nn_out = train(
    model_σ_nn,
    df,
    ();
    nepochs = 100,
    batchsize = 512,
    opt = AdamW(0.1),
    monitor_names = [:rb, :Q10, :sigma], # σ is surfaced top-level in `ŷ` (and in `ŷ.parameters`)
    yscale = identity,
    shuffleobs = true,
    loss_types = [:mse, :nse],
    training_loss = gaussian_nll,
    show_progress = false,
    model_name = "RbQ10_nll_nn"
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
    shuffleobs = true,
    model_name = "RbQ10_synthetic2"
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
    shuffleobs = true,
    model_name = "RbQ10_synthetic3"
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
    shuffleobs = true,
    model_name = "RbQ10_synthetic4"
)

LuxCore.testmode(multi_nn_out.st)
mean(df.dsw_pot)
mean(df.sw_pot)

# ## Autodiff backends

using Enzyme
using EasyHybrid.Lux: AutoEnzyme   # ADTypes constructors are not re-exported

nll_enzyme_out = train(
    model_σ_global,
    df;
    nepochs = 100,
    batchsize = 512,
    opt = AdamW(0.1),
    monitor_names = [:rb, :Q10, :sigma],
    yscale = identity,
    shuffleobs = true,
    loss_types = [:mse, :nse],
    training_loss = gaussian_nll,
    autodiff_backend = AutoEnzyme(mode = Enzyme.set_runtime_activity(Enzyme.Reverse)),
    show_progress = false,
    model_name = "RbQ10_nll_enzyme"
)

# This reaches a loss of the same magnitude as `nll_global_out`, not an identical one:
# the two backends agree to Float32 roundoff on any single gradient, and Adam amplifies
# that into visibly different trajectories. Compare single-step gradients, not
# multi-epoch losses, when checking a backend.
#
# Timings for exactly the run above (CPU, Julia 1.12.5, Enzyme 0.13.199), taken with a
# 1-epoch warmup to absorb compilation followed by 50 timed epochs:
#
# | backend | warmup (1 epoch, incl. compilation) | per epoch afterwards |
# |:---|---:|---:|
# | `AutoZygote()` | ~90 s | ~0.12 s |
# | `AutoEnzyme(…)` | ~205 s | ~0.08 s |
#
# Enzyme is roughly 1.5× faster per epoch here, but spends ~115 s more on compilation,
# so it only breaks even after a few thousand epochs — and it recompiles whenever the
# model, loss or data types change. Zygote therefore remains the default; Enzyme is the
# better choice for long runs, and for mechanistic models with scalar-heavy or loop-heavy
# code where Zygote's array-level rules are weakest.
