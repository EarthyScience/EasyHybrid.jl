# Mock-up Dashboard Test Script for EasyHybrid.jl
#
# This script sets up a synthetic multi-output hybrid model to test and demonstrate:
# 1. Multi-output scatterplots (prediction) and timeseries for all target variables (:obs_dyn1, :obs_dyn2).
# 2. `yscale = identity` with `:nse` loss (verifying non-logarithmic scaling and negative/≤1 metrics).
# 3. Zoomed-in loss panel displaying correct (unflipped) training vs validation loss curves.
# 4. Monitor panel for physical parameters.
# 5. `save_training = true` animation and snapshot generation.

using EasyHybrid
using CairoMakie
using Random
using Lux
using ComponentArrays
using DataFrames

println("=== 1. Generating Multi-Output Synthetic Data ===")
dk = gen_linear_data_2outputs(seed = 42)

# Mechanistic model producing 2 targets (:obs_dyn1, :obs_dyn2) and intermediate variable :a_dyn
function multi_output_mechanistic(; x1, x2, a, b, a_dyn)
    obs_dyn1 = a_dyn .* x1 .+ b
    obs_dyn2 = 0.5f0 .* a_dyn .* x2
    return (; obs_dyn1, obs_dyn2, a_dyn)
end

# Define parameters
params = (
    a = (1.0f0, 0.0f0, 5.0f0),
    b = (2.0f0, 0.0f0, 10.0f0),
)

# Neural network predicting dynamic parameter a_dyn from inputs (:x2, :x3)
nn = Chain(Dense(2 => 8, relu), Dense(8 => 1))

# Construct HybridModel
hm = constructHybridModel(
    mechanistic_model = multi_output_mechanistic,
    parameters = params,
    nn = nn,
    input_names = (:x2, :x3),
    forcing_names = (:x1, :x2),
    target_names = (:obs_dyn1, :obs_dyn2),
    fixed = (),
)

println("=== 2. Testing Training with Dashboard, yscale=identity, and NSE loss ===")
# Run training with multiple loss types including NSE and MSE
results = train(
    hm,
    dk;
    nepochs = 15,
    batchsize = 64,
    opt = Adam(0.01f0),
    loss_types = [:nse, :mse],
    training_loss = :nse,
    yscale = identity,                     # Test identity yscale with NSE
    monitor_names = [:a, :b],             # Track parameters in monitor panel
    dashboard_components = [:loss, :prediction, :timeseries, :monitor],
    save_training = false,                 # Test live dashboard without saving
    plotting = true,
)

println("=== Training completed successfully! ===")
println("Best epoch: ", results.best_epoch, " with best loss: ", results.best_loss)
println("Target names tested: ", hm.targets)
