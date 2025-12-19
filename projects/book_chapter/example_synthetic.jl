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
project_path = "projects/book_chapter"
Pkg.activate(project_path)
EasyHybrid_path = joinpath(pwd())
Pkg.develop(path = EasyHybrid_path)
#Pkg.resolve()
#Pkg.instantiate()

using EasyHybrid
using AxisKeys
using DimensionalData

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
using WGLMakie
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

# =============================================================================
# train on DataFrame
# =============================================================================

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
    extra_loss = extra_loss
)

# =============================================================================
# train on KeyedArray
# =============================================================================
single_nn_out = train(
    single_nn_hybrid_model,
    ka,
    ();
    nepochs = 10,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = true
)

# =============================================================================
# train on DimensionalData
# =============================================================================
single_nn_out = train(
    single_nn_hybrid_model,
    da,
    ();
    nepochs = 10,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    shuffleobs = true
)

LuxCore.testmode(single_nn_out.st)
mean(df.dsw_pot)
mean(df.sw_pot)

# =============================================================================
# Multi NN Hybrid Model Training
# =============================================================================
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
    shuffleobs = true
)

LuxCore.testmode(multi_nn_out.st)
mean(df.dsw_pot)
mean(df.sw_pot)

# =============================================================================
# Pure ML Single NN Model Training
# =============================================================================

# TODO does not train well build on top of SingleNNHybridModel
predictors_single_nn_ml = [:sw_pot, :dsw_pot, :ta]

single_nn_model = constructNNModel(predictors_single_nn_ml, target; input_batchnorm = true, activation = tanh)
single_nn_out = train(single_nn_model, da, (); nepochs = 10, batchsize = 512, opt = AdamW(0.01), yscale = identity, shuffleobs = true)
LuxCore.testmode(single_nn_out.st)
mean(df.dsw_pot)
mean(df.sw_pot)

single_nn_model.targets

# =============================================================================
# Pure ML Multi NN Model Training
# =============================================================================

# TODO does not train well build on top of MultiNNHybridModel


# =============================================================================
# Results Analysis
# =============================================================================
# Check the training differences for Q10 parameter
# This shows how close the model learned the true Q10 value
out.train_diffs.Q10

using Hyperopt
using Distributed
using WGLMakie

mspempty = ModelSpec()

nhyper = 4
ho = @thyperopt for i in nhyper,
        opt in [AdamW(0.01), AdamW(0.1), RMSProp(0.001), RMSProp(0.01)],
        input_batchnorm in [true, false]

    hyper_parameters = (; opt, input_batchnorm)
    println("Hyperparameter run: \n", i, " of ", nhyper, "\t with hyperparameters \t", hyper_parameters, "\t")
    out = EasyHybrid.tune(hybrid_model, df, mspempty; hyper_parameters..., nepochs = 10, plotting = false, show_progress = false, file_name = "test$i.jld2")
    #out.best_loss
    # out.best_loss, has to be first element of the tuple, return a rich record for this trial (stored in ho.results[i])
    (out.best_loss, hyperps = hyper_parameters, ps_st = (out.ps, out.st), i = i)
end

losses = getfield.(ho.results, :best_loss)
hyperps = getfield.(ho.results, :hyperps)

# Helper function to make optimizer names short and readable
function short_opt_name(opt)
    if opt isa AdamW
        return "AdamW(η=$(opt.eta))"
    elseif opt isa RMSProp
        return "RMSProp(η=$(opt.eta))"
    else
        return string(typeof(opt))
    end
end

# Sort losses and associated data by increasing loss
idx = sortperm(losses)
sorted_losses = losses[idx]
sorted_hyperps = hyperps[idx]

fig = Figure(figure_padding = 50)
# Prepare tick labels with hyperparameter info for each trial (sorted)
sorted_ticklabels = [
    join(
            [
                k == :opt ? "opt=$(short_opt_name(v))" : "$k=$(repr(v))"
                for (k, v) in pairs(hp)
            ], "\n"
        )
        for hp in sorted_hyperps
]
ax = Makie.Axis(
    fig[1, 1];
    xlabel = "Trial",
    ylabel = "Loss",
    title = "Hyperparameter Tuning Results",
    xgridvisible = false,
    ygridvisible = false,
    xticks = (1:length(sorted_losses), sorted_ticklabels),
    xticklabelrotation = 45
)
scatter!(ax, 1:length(sorted_losses), sorted_losses; markersize = 15, color = :dodgerblue)

# Get the best trial
best_idx = argmin(losses)
best_trial = ho.results[best_idx]

best_params = best_trial.ps_st        # (ps, st)

# Print the best hyperparameters
printmin(ho)

# Plot the results
import Plots
using Plots.PlotMeasures
# rebuild the ho object as intended by plot function for hyperopt object
ho2 = deepcopy(ho)
ho2.results = getfield.(ho.results, :best_loss)

Plots.plot(ho2, xrotation = 25, left_margin = [100mm 0mm], bottom_margin = 60mm, ylab = "loss", size = (900, 900))

# Train the model with the best hyperparameters
best_hyperp = best_hyperparams(ho)
out = EasyHybrid.tune(hybrid_model, df, mspempty; best_hyperp..., nepochs = 100, train_from = best_params)
