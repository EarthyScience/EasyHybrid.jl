# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
Pkg.activate("projects/soil_water_retention_curve")
Pkg.develop(path=pwd())
Pkg.instantiate()

using EasyHybrid
using GLMakie
using Statistics
using ComponentArrays

# Load and preprocess data
@__DIR__
df_o = CSV.read(joinpath(@__DIR__, "./data/Norouzi_et_al_2024_WRR_Final.csv"), DataFrame, normalizenames=true)

df = copy(df_o)
df.h = 10 .^ df.pF

# Rename :WC to :θ in the DataFrame
df.θ = df.WC # make at % scale - seems like better training, better gradients?

df

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

typeof(FXWParams)

function construct_hybrid(parameters::NamedTuple, f::DataType)
    ca = EasyHybrid.ParameterContainer(parameters)
    return f(ca)
end

parameter_container = construct_hybrid(parameters, FXWParams)

function default(p::AbstractHybridModel)
    p.hybrid.table[:, :default]
end

default(parameter_container)


# =============================================================================
# Model Functions
# =============================================================================

# Explicit parameter method
mechanistic_model(h; θ_s, h_r, h_0, log_α, log_nm1, log_m) = mFXW_theta(h, θ_s, h_r, h_0, exp.(log_α), exp.(log_nm1) .+ 1, exp.(log_m)) * 100.0 # scale to %

mechanistic_model(h, params::AbstractHybridModel) = mechanistic_model(h; values(default(params))...)


# =============================================================================
# Default Model Behaviour
# =============================================================================
pFs = vec(collect(range(-1.0, 7, length=100))) .|> Float32

h_values = sort(Array(ds_keyed(:h)))
pF_values = sort(Array(ds_keyed(:pF)))

θ_pred = mechanistic_model(h_values, parameter_container)

GLMakie.activate!(inline=true)
fig = Figure()
ax = Makie.Axis(fig[1, 1], xlabel = "θ", ylabel = "pF")
plot!(ax, ds_keyed(:θ), ds_keyed(:pF), label="data", color=(:grey25, 0.25))
lines!(ax, θ_pred, pF_values, color=:red, label="FXW default")
axislegend(ax; position=:rt)
fig

fig = Figure(size=(800, 600))
plot_pred_vs_obs!(fig, Array(ds_keyed(:θ)), θ_pred, "Default", 1, 1)

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

fieldnames(typeof(hybrid_model))

hybrid_model.mechanistic_model

ps = LuxCore.initialparameters(Random.default_rng(), hybrid_model)
st = LuxCore.initialstates(Random.default_rng(), hybrid_model)

hybrid_model(ds_keyed, ps, st)

tout = train(hybrid_model, ds_keyed, (); nepochs=100, batchsize=512, opt=AdaGrad(0.01), file_name = "tout.jld2", training_loss=:nse, loss_types=[:mse, :nse])

θ_pred1 = tout.val_obs_pred[!, Symbol("θ_pred")]
θ_obs1 = tout.val_obs_pred[!, :θ]

using GLMakie
plot_pred_vs_obs!(fig, θ_pred1, θ_obs1, "Global parameters", 2, 1)


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

ps = LuxCore.initialparameters(Random.default_rng(), hybrid_model_nn)
st = LuxCore.initialstates(Random.default_rng(), hybrid_model_nn)

hybrid_model_nn(ds_keyed, ps, st)

tout2 = train(hybrid_model_nn, ds_keyed, (); nepochs=100, batchsize=512, opt=AdaGrad(0.01), file_name = "tout2.jld2", training_loss=:nse, loss_types=[:mse, :nse])

# =============================================================================
# Results Visualization
# =============================================================================

θ_pred2 = tout2.val_obs_pred[!, Symbol(string(:θ, "_pred"))]
θ_obs2 = tout2.val_obs_pred[!, :θ]

plot_pred_vs_obs!(fig, θ_pred2, θ_obs2, "Neural parameters", 1, 2)







