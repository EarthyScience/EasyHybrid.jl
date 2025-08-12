# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
project_path = "projects/book_chapter"
Pkg.activate(project_path)

manifest_path = joinpath(project_path, "Manifest.toml")
if !isfile(manifest_path)
    package_path = pwd() 
    if !endswith(package_path, "EasyHybrid")
        @error "You opened in the wrong directory. Please open the EasyHybrid folder, create a new project in the projects folder and provide the relative path to the project folder as project_path."
    end
    Pkg.develop(path=package_path)
    Pkg.instantiate()
end

# =============================================================================
# Load packages
# =============================================================================

using EasyHybrid
using NCDatasets

# load data from bookchapter
include("load_nc.jl")

ds = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc")

# define Model
RbQ10 = function(;ta, Q10, rb, tref = 15.0f0)

    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)

end

parameters = (
    #            default                  lower                     upper                description
    rb       = ( 3.0f0,                  0.0f0,                   13.0f0 ),            # Basal respiration [μmol/m²/s]
    Q10      = ( 2.0f0,                  1.0f0,                   4.0f0 ),            # Temperature sensitivity factor [-]
)

# hybrid model creation
forcing = [:ta]
predictors = [:sw_pot, :dsw_pot]
target = [:reco]

global_param_names = [:Q10]
neural_param_names = [:rb]

# Create the hybrid model using the unified constructor
hybrid_model = constructHybridModel(
    predictors,
    forcing,
    target,
    RbQ10,
    parameters,
    neural_param_names,
    global_param_names,
    hidden_layers = [16, 16],
    activation = tanh,
    scale_nn_outputs=true,
    input_batchnorm = true
)

using WGLMakie
out = train(hybrid_model, ds, (); nepochs=100, batchsize=512, opt=RMSProp(0.005), monitor_names=[:rb, :Q10], yscale = identity, patience=30)

# What do you think? Close to the true value?
out.train_diffs.Q10






