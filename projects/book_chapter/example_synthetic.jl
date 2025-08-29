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

# Check if manifest exists, create project if needed
manifest_path = joinpath(project_path, "Manifest.toml")
if !isfile(manifest_path)
    package_path = pwd() 
    if !endswith(package_path, "EasyHybrid")
        @error "You opened in the wrong directory. Please open the EasyHybrid folder, create a new project in the projects folder and provide the relative path to the project folder as project_path."
    end
    Pkg.develop(path=package_path)
    Pkg.instantiate()
end

using EasyHybrid

# =============================================================================
# Data Loading and Preprocessing
# =============================================================================
# Load synthetic dataset from GitHub
ds = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc")

# Select a subset of data for faster execution
ds = ds[1:20000, :]

# =============================================================================
# Define the Physical Model
# =============================================================================
# RbQ10 model: Respiration model with Q10 temperature sensitivity
# Parameters:
#   - ta: air temperature [°C]
#   - Q10: temperature sensitivity factor [-]
#   - rb: basal respiration rate [μmol/m²/s]
#   - tref: reference temperature [°C] (default: 15.0)
function RbQ10(;ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end

# =============================================================================
# Define Model Parameters
# =============================================================================
# Parameter specification: (default, lower_bound, upper_bound)
parameters = (
# Parameter name | Default | Lower | Upper      | Description
    rb       = ( 3.0f0,      0.0f0,  13.0f0 ),  # Basal respiration [μmol/m²/s]
    Q10      = ( 2.0f0,      1.0f0,  4.0f0 ),   # Temperature sensitivity factor [-]
)

# =============================================================================
# Configure Hybrid Model Components
# =============================================================================
# Define input variables
forcing = [:ta]                    # Forcing variables (temperature)
predictors = [:sw_pot, :dsw_pot]   # Predictor variables (solar radiation, and its derivative)

# Target variable
target = [:reco]                   # Target variable (respiration)

# Parameter classification
global_param_names = [:Q10]        # Global parameters (same for all samples)
neural_param_names = [:rb]         # Neural network predicted parameters

# =============================================================================
# Construct the Hybrid Model
# =============================================================================
# Create hybrid model using the unified constructor
hybrid_model = constructHybridModel(
    predictors,              # Input features
    forcing,                 # Forcing variables
    target,                  # Target variables
    RbQ10,                  # Process-based model function
    parameters,              # Parameter definitions
    neural_param_names,      # NN-predicted parameters
    global_param_names,      # Global parameters
    hidden_layers = [16, 16], # Neural network architecture
    activation = sigmoid,      # Activation function
    scale_nn_outputs = true, # Scale neural network outputs
    input_batchnorm = false   # Apply batch normalization to inputs
)

# =============================================================================
# Model Training
# =============================================================================
using WGLMakie

# Train the hybrid model
out = train(
    hybrid_model, 
    ds, 
    (); 
    nepochs = 100,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = AdamW(0.1),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity       # Scaling for outputs
)

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
ho = @thyperopt for i=nhyper,
    opt = [AdamW(0.01), AdamW(0.1), RMSProp(0.001), RMSProp(0.01)],
    input_batchnorm = [true, false]
    hyper_parameters = (;opt, input_batchnorm)
    println("Hyperparameter run: \n", i, " of ", nhyper, "\t with hyperparameters \t", hyper_parameters, "\t")
    out = EasyHybrid.tune(hybrid_model, ds, mspempty; hyper_parameters..., nepochs = 10, plotting = false, show_progress = false, file_name = "test$i.jld2")
    #out.best_loss
    # return a rich record for this trial (stored in ho.results[i])
    (out.best_loss,
     hyperps = hyper_parameters,
     ps_st = (ps = out.ps, st = out.st),
     file = "test$i.jld2",
     i = i)
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
    join([
        k == :opt ? "opt=$(short_opt_name(v))" : "$k=$(repr(v))"
        for (k, v) in pairs(hp)
    ], "\n")
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
scatter!(ax, 1:length(sorted_losses), sorted_losses; markersize=15, color=:dodgerblue)



best_idx = argmin(losses)
best_trial = ho.results[best_idx]

best_params = best_trial.params        # (ps, st)

# Print the best hyperparameters
printmin(ho)

# Plot the results
import Plots
using Unitful
Plots.plot(ho, xrotation=25, left_margin=[100mm 0mm], bottom_margin=60mm, ylab = "loss", size = (900, 900)) 

# Train the model with the best hyperparameters
best_hyperp = best_hyperparams(ho)
out = EasyHybrid.tune(hybrid_model, ds, mspempty; best_hyperp..., nepochs = 100)
