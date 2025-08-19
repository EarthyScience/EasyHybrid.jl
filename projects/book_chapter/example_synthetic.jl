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
    activation = swish,      # Activation function
    scale_nn_outputs = true, # Scale neural network outputs
    input_batchnorm = true   # Apply batch normalization to inputs
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
    opt = RMSProp(0.001),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    patience = 30            # Early stopping patience
)

# =============================================================================
# Results Analysis
# =============================================================================
# Check the training differences for Q10 parameter
# This shows how close the model learned the true Q10 value
out.train_diffs.Q10