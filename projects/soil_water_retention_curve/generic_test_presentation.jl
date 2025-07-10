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
df.θ = df.WC # make it larger an rescale in hybrid model as well - better gradients?

df

ds_keyed = to_keyedArray(Float32.(df))

# =============================================================================
# Parameter Structure Definition
# =============================================================================

abstract type AbstractHybridModel end

mutable struct FXWParams10 <: AbstractHybridModel
    table::ComponentMatrix{Float32, Matrix{Float32}} 

    # inner constructor lives inside the struct block
    function FXWParams10()
        values = (
                    # "columns" are: default, lower, upper 
                    θ_s = (0.396f0,     0.302f0,     0.700f0),     # Saturated water content [cm³/cm³]
                    h_r = (1500.0f0,   1500.0f0,    1500.0f0),    # Pressure head at residual water content [cm]
                    h_0 = (6.3f6,    6.3f6,     6.3f6),     # Pressure head at zero water content [cm]
                    log_α   = (log(0.048f0), log(0.01f0), log(7.874f0)),     # Shape parameter [cm⁻¹] 
                    log_nm1   = (log(3.302f0 - 1), log(1.100f0 - 1), log(20.000f0 - 1)),     # Shape parameter [-]
                    log_m   = (log(0.199f0),    log(0.100f0),     log(2.000f0)),     # Shape parameter [-]
        )
        cm = build_cm(values)
        new(cm)             # wrap it up
    end
end

ca = FXWParams10()

# =============================================================================
# Model Functions
# =============================================================================
using PrettyTables
display_parameter_bounds(ca.table)

# Explicit parameter method
mechfun(h, θ_s, h_r, h_0, log_α, log_nm1, log_m) = mFXW_theta(h, θ_s, h_r, h_0, exp.(log_α), exp.(log_nm1) .+ 1, exp.(log_m)) .* 100

# Struct-based method delegates to the above
mechfun(h, params::FXWParams10) = begin
    p = params.table[:, :default]
    mechfun(h, p.θ_s, p.h_r, p.h_0, p.log_α, p.log_nm1, p.log_m)
end

# =============================================================================
# Default Model Bahviour
# =============================================================================
pFs = vec(collect(range(-1.0, 7, length=100))) .|> Float32
θ_pred = mechfun(10.0 .^ pFs, ca)

GLMakie.activate!(inline=true)
fig = Figure()
ax = Makie.Axis(fig[1, 1], xlabel = "θ", ylabel = "pF")
plot!(ax, ds_keyed(:θ), ds_keyed(:pF), label="data", color=(:grey25, 0.25))
lines!(ax, θ_pred, pFs, color=:red, label="FXW default", linewidth=2)
axislegend(ax; position=:rt)
fig

# =============================================================================
# Global Parameter Training
# =============================================================================
targets = [:θ]
forcing = [:h]

# Build hybrid model with global parameters only
hm = constructHybridModel8(
    [],               # predictors
    forcing,          # forcing
    targets,          # target
    mechfun,          # physics function
    ca,               # parameter bounds
    [],               # nn_names
    [:θ_s, :log_α, :log_nm1, :log_m]  # global_names
)


ps = LuxCore.initialparameters(Random.default_rng(), hm)
st = LuxCore.initialstates(Random.default_rng(), hm)

tout = train(hm, ds_keyed, (); nepochs=100, batchsize=512, opt=AdaGrad(0.01), file_name = "tout.jld2", training_loss=:nse, loss_types=[:mse, :nse])

# =============================================================================
# Neural Network Training
# =============================================================================
predictors = [:BD, :OC, :clay, :silt, :sand]

# Build hybrid model with neural network
hm2 = constructHybridModel8(
    predictors,        # predictors
    forcing,          # forcing
    targets,          # target
    mechfun,          # physics function
    ca,               # parameter bounds
    [:θ_s, :log_α, :log_nm1, :log_m],  # nn_names
    []                # global_names
)

ps = LuxCore.initialparameters(Random.default_rng(), hm2)
st = LuxCore.initialstates(Random.default_rng(), hm2)

tout2 = train(hm2, ds_keyed, (); nepochs=100, batchsize=512, opt=AdaGrad(0.01), file_name = "tout2.jld2", training_loss=:nse, loss_types=[:mse, :nse])

# =============================================================================
# Results Visualization
# =============================================================================
θ_pred1 = tout.val_obs_pred[!, Symbol("θ_pred")]
θ_obs1 = tout.val_obs_pred[!, :θ]

using GLMakie
fig = Figure(size=(800, 600))
opplot!(fig, θ_pred1, θ_obs1, "Global parameters", 1, 1)

θ_pred2 = tout2.val_obs_pred[!, Symbol(string(:θ, "_pred"))]
θ_obs2 = tout2.val_obs_pred[!, :θ]

opplot!(fig, θ_pred2, θ_obs2, "Neural parameters", 1, 2)







