# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
Pkg.activate("projects/soil_water_retention_curve")
Pkg.develop(path=pwd())
Pkg.instantiate()

using EasyHybrid
using GLMakie
import EasyHybrid: poplot, poplot!
using Statistics
using ComponentArrays

# =============================================================================
# Data Source Information
# =============================================================================
# Source:
#   Norouzi S, Pesch C, Arthur E et al. (2025)
#   "Physics‐Informed Neural Networks for Estimating a Continuous Form of the Soil Water Retention Curve From Basic Soil Properties."
#   Water Resources Research, 61.
#
# Dataset:
#   Norouzi, S., Pesch, C., Arthur, E., Norgaard, T., Greve, M. H., Iversen, B. V., & de Jonge, L. W. (2024).
#   Input dataset for estimating continuous soil water retention curves using physics‐informed [Dataset]. Neural Networks.
#   https://doi.org/10.5281/ZENODO.14041446
#
# Local Path (MPI-BGC server):
#   /Net/Groups/BGI/scratch/bahrens/data_Norouzi/Norouzi_et_al_2024_WRR_Final.csv
# =============================================================================
# Load and preprocess data
@__DIR__

df_o = CSV.read(joinpath(@__DIR__, "./data/Norouzi_et_al_2024_WRR_Final.csv"), DataFrame, normalizenames=true)

df = copy(df_o)
df.h = 10 .^ df.pF # convert pF to cm

# Rename :WC to :θ in the DataFrame
df.θ = df.WC # keep at % scale - seems like better training, better gradients?

ds_keyed = to_keyedArray(Float32.(df))

# =============================================================================
# Parameter Structure Definition
# =============================================================================

struct FXWParams <: AbstractHybridModel # name your ParameterContainer to your liking
    hybrid::EasyHybrid.ParameterContainer
end

# construct a named tuple of parameters with tuples of (default, lower, upper)
parameters = (
    #            default                  lower                     upper                description
    θ_s      = ( 0.396f0,                 0.302f0,                  0.700f0 ),           # Saturated water content [cm³/cm³]
    h_r      = ( 1500.0f0,                1500.0f0,                 1500.0f0 ),          # Pressure head at residual water content [cm]
    h_0      = ( 6.3f6,                   6.3f6,                    6.3f6 ),             # Pressure head at zero water content [cm]
    log_α    = ( log(0.048f0),            log(0.01f0),              log(7.874f0) ),      # Shape parameter [cm⁻¹] 
    log_nm1  = ( log(3.302f0 - 1),        log(1.100f0 - 1),         log(20.000f0 - 1) ), # Shape parameter [-]
    log_m    = ( log(0.199f0),            log(0.100f0),             log(2.000f0) ),      # Shape parameter [-]
)


parameter_container = build_parameters(parameters, FXWParams)

# =============================================================================
# Model Functions
# =============================================================================

# generate function with parameters as keyword arguments -> needed for hybrid model
function mechanistic_model(h; θ_s, h_r, h_0, log_α, log_nm1, log_m)
    return mFXW_theta(h, θ_s, h_r, h_0, exp.(log_α), exp.(log_nm1) .+ 1, exp.(log_m)) * 100.0 # scale to %
end

function mechanistic_model(h, params::AbstractHybridModel)
    return mechanistic_model(h; values(default(params))...)
end

# =============================================================================
# Default Model Behaviour
# =============================================================================
h_values = sort(Array(ds_keyed(:h)))
pF_values = sort(Array(ds_keyed(:pF)))

θ_pred = mechanistic_model(h_values, parameter_container)

GLMakie.activate!(inline=false)
fig_swrc = Figure()
ax = Makie.Axis(fig_swrc[1, 1], xlabel = "θ", ylabel = "pF")
plot!(ax, ds_keyed(:θ), ds_keyed(:pF), label="data", color=(:grey25, 0.25))
lines!(ax, θ_pred, pF_values, color=:red, label="FXW default")
axislegend(ax; position=:rt)
fig_swrc

fig_po = poplot(Array(ds_keyed(:θ)), θ_pred, "Default")

# =============================================================================
# Global Parameter Training
# =============================================================================
targets = [:θ]
forcing = [:h]

# Build hybrid model with global parameters only
hybrid_model = constructHybridModel(
    [],               # predictors
    forcing,          # forcing
    targets,          # target
    mechanistic_model,          # mechanistic model
    parameter_container,               # parameter defaults and bounds of mechanistic model
    [],               # nn_names
    [:θ_s, :log_α, :log_nm1, :log_m]  # global_names
)

tout = train(hybrid_model, ds_keyed, (); nepochs=100, batchsize=512, opt=AdaGrad(0.01), file_name = "tout.jld2", training_loss=:nse, loss_types=[:mse, :nse])

θ_pred1 = tout.val_obs_pred[!, Symbol("θ_pred")]
θ_obs1 = tout.val_obs_pred[!, :θ]

poplot!(fig_po, θ_pred1, θ_obs1, "Global parameters", 2, 1)


# =============================================================================
# Neural Network Training
# =============================================================================
predictors = [:BD, :OC, :clay, :silt, :sand]

# Build hybrid model with neural network
hybrid_model_nn = constructHybridModel(
    predictors,                                 # predictors
    forcing,                                    # forcing
    targets,                                    # targets
    mechanistic_model,                                    # mechanistic model
    parameter_container,                                         # parameter bounds
    [:θ_s, :log_α, :log_nm1, :log_m],           # neural_param_names
    []                                          # global_names
)

tout2 = train(hybrid_model_nn, ds_keyed, (); nepochs=100, batchsize=512, opt=AdaGrad(0.01), file_name = "tout2.jld2", training_loss=:nse, loss_types=[:mse, :nse])

# =============================================================================
# Results Visualization
# =============================================================================

θ_pred2 = tout2.val_obs_pred[!, Symbol(string(:θ, "_pred"))]
θ_obs2 = tout2.val_obs_pred[!, :θ]

poplot!(fig_po, θ_pred2, θ_obs2, "Neural parameters", 1, 2)







