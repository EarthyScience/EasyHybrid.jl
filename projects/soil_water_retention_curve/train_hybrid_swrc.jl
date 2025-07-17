# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
project_path = "projects/soil_water_retention_curve"
Pkg.activate(project_path)
Pkg.develop(path=pwd())
# Only instantiate if Manifest.toml is missing
manifest_path = joinpath(project_path, "Manifest.toml")
if !isfile(manifest_path)
    Pkg.instantiate()
end

using EasyHybrid
using WGLMakie
import EasyHybrid: poplot, poplot!
using Statistics
using ComponentArrays
using AxisKeys

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
df_o = CSV.read(joinpath(@__DIR__, "./data/Norouzi_et_al_2024_WRR_Final.csv"), DataFrame, normalizenames=true)

df = copy(df_o)
df.h = 10 .^ df.pF # convert pF to cm

# Rename :WC to :θ in the DataFrame
df.θ = df.WC ./ 100.0 # TODO keep at % scale - seems like better training, better gradients?

ds_keyed = to_keyedArray(Float32.(df))

# =============================================================================
# Parameter Structure Definition
# =============================================================================

# name your ParameterContainer to your liking
struct FXWParams <: AbstractHybridModel 
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
# Mechanistic Model and helper functions
# =============================================================================
# Modified FX model by Wang et al. (2018) 
function mFXW_theta(h, θ_s, h_r, h_0, α, n, m)

    # Arguments:
    #   h    :: Pressure head [L] 
    #   θ_s  :: Saturated water content [L3 L-3]
    #   h_r  :: Pressure head corresponding to the residual water content [L]
    #   h_0  :: Pressure head at zero water content [L]
    #   α, n, m :: Shape parameters [L-3, -, -]

    # Correction factor C_f(h)
    C_f = @. 1.f0 - log(1.f0 + h / h_r) / log(1.f0 + h_0 / h_r)
    
    # Gamma function Γ(h)
    Γ = @. (log(exp(1.f0) + abs(α * h)^n))^(-m)
    
    # Effective saturation S_e(h)
    S_e = @. C_f * Γ
    
    # Volumetric water content θ(h), Soil water retention curve
    θ = θ_s .* S_e

    return (; θ, S_e, C_f, Γ)
end

# generate function with parameters as keyword arguments 
# -> needed for hybrid model, 
# log_α, log_nm1, log_m are log transformed parameters for better training across orders of magnitude

function mechanistic_model(h; θ_s, h_r, h_0, log_α, log_nm1, log_m)
    return mFXW_theta(h, θ_s, h_r, h_0, exp.(log_α), exp.(log_nm1) .+ 1, exp.(log_m))
end

function mechanistic_model(;h, θ_s, h_r, h_0, log_α, log_nm1, log_m)
    return mFXW_theta(h, θ_s, h_r, h_0, exp.(log_α), exp.(log_nm1) .+ 1, exp.(log_m))
end

# KeyedArray version needed for hybrid model
function mechanistic_model(forcing_data::KeyedArray; kwargs...)
    h = vec(forcing_data([:h]))  # Extract h from forcing data
    return mechanistic_model(h; kwargs...)
end

function mechanistic_model(h, params::AbstractHybridModel)
    return mechanistic_model(h; values(default(params))...)
end

function mechanistic_model(forcing_data::KeyedArray, params::AbstractHybridModel)
    return mechanistic_model(forcing_data; values(default(params))...)
end



# =============================================================================
# Default Model Behaviour
# =============================================================================
h_values = sort(Array(ds_keyed(:h)))
pF_values = sort(Array(ds_keyed(:pF)))

θ_pred = mechanistic_model(h_values, parameter_container).θ

#GLMakie.activate!(inline=false)
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
    [:θ_s, :log_α, :log_nm1, :log_m],  # global_names
    scale_nn_outputs=false
)

tout = train(hybrid_model, ds_keyed, (); nepochs=100, batchsize=256, opt=AdaGrad(0.01), file_name = "tout.jld2", training_loss=:nse, loss_types=[:mse, :nse])

θ_pred1 = tout.val_obs_pred[!, Symbol("θ_pred")]
θ_obs1 = tout.val_obs_pred[!, :θ]

poplot(θ_pred1, θ_obs1, "Global parameters")


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
    [],                                          # global_names
    scale_nn_outputs=true,
    hidden_layers = [32, 32],
    activation = tanh
)

tout2 = train(hybrid_model_nn, ds_keyed, (); nepochs=100, batchsize=256, opt=AdaGrad(0.01), file_name = "tout2.jld2", training_loss=:nse, loss_types=[:mse, :nse])

# =============================================================================
# Results Visualization
# =============================================================================

θ_pred2 = tout2.val_obs_pred[!, Symbol(string(:θ, "_pred"))]
θ_obs2 = tout2.val_obs_pred[!, :θ]

poplot(θ_pred2, θ_obs2, "Neural parameters")


