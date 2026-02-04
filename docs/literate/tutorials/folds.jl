# # Cross-Validation in EasyHybrid.jl
#
# This tutorial demonstrates one option for cross-validation in EasyHybrid.
# The code for this tutorial can be found in [docs/src/literate/tutorials](https://github.com/EarthyScience/EasyHybrid.jl/tree/main/docs/src/literate/tutorials/) => folds.jl.
#
# ## 1. Load Packages

using EasyHybrid
using OhMyThreads
using CairoMakie

# ## 2. Data Loading and Preprocessing

# Load synthetic dataset from GitHub
df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc");

# Select a subset of data for faster execution
df = df[1:20000, :];
first(df, 5)

# ## 3. Define the Physical Model

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

# ## 4. Define Model Parameters

# Parameter specification: (default, lower_bound, upper_bound)
parameters = (
    rb = (3.0f0, 0.0f0, 13.0f0),
    Q10 = (2.0f0, 1.0f0, 4.0f0),
)

# ## 5. Configure Hybrid Model Components

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

# ## 6. Construct the Hybrid Model

hybrid_model = constructHybridModel(
    predictors,
    forcing,
    target,
    RbQ10,
    parameters,
    neural_param_names,
    global_param_names,
    hidden_layers = [16, 16],
    activation = sigmoid,
    scale_nn_outputs = true,
    input_batchnorm = true
)

# ## 7. Model Training: k-Fold Cross-Validation

k = 3
folds = make_folds(df, k = k, shuffle = true)

results = Vector{Any}(undef, k)

@time @tasks for val_fold in 1:k
    @info "Split data outside of train function. Training fold $val_fold of $k"
    sdata = split_data(df, hybrid_model; val_fold = val_fold, folds = folds)
    out = train(
        hybrid_model,
        sdata,
        ();
        nepochs = 10,
        patience = 10,
        batchsize = 512,         # Batch size for training
        opt = RMSProp(0.001),    # Optimizer and learning rate
        monitor_names = [:rb, :Q10],
        hybrid_name = "folds_$(val_fold)",
        folder_to_save = "CV_results",
        file_name = "trained_model_folds_$(val_fold).jld2",
        show_progress = false,
        plotting = false
    )
    results[val_fold] = out
end
