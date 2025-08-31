# CC BY-SA 4.0
# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
project_path = "projects/ODE_example"
Pkg.activate(project_path)
Pkg.instantiate()
Pkg.precompile()

Pkg.develop(path=pwd())

# start using the package
using EasyHybrid
using AxisKeys
using WGLMakie

include(joinpath(pwd(), "projects", "Respiration_Fluxnet", "Data", "load_data.jl"))

# =============================================================================
# Load data
# =============================================================================

# copy data to data/data20240123/ from here /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123
# or adjust the path to /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123 + FluxNetSite

fluxnet_data = load_fluxnet_nc(joinpath("projects", "Respiration_Fluxnet", "Data", "data20240123", "US-SRG.nc"), timevar="date")

fluxnet_data.timeseries.dayofyear = dayofyear.(fluxnet_data.timeseries.time)

# select timeseries data
df = fluxnet_data.timeseries

# =============================================================================
# Mechanistic Model Definition
# =============================================================================

using OrdinaryDiffEq
using StaticArrays

# Data grid
# Convert from milliseconds to days
Δt_ms = diff(fluxnet_data.timeseries.time)[1]
Δt = Δt_ms / Day(1)
t0 = 1.0 * Δt
tend = length(fluxnet_data.timeseries.time) * Δt
tgrid = collect(t0:Δt:tend)

# Remove any use of Union{Missing, Float64} or similar types.
# Instead, ensure all data is Float64 (or appropriate concrete type) and handle missings by converting them to NaN.

using DataFrames
df = mapcols(col -> Float64.(replace!(col, missing => NaN)), df; cols = names(df, Union{Missing, Real}))


# piecewise-constant forcing aligned to the grid
struct GridForcing{T}; t::Vector{T}; x::Vector{T}; end
function (gf::GridForcing)(t)
    k = argmin(abs.(gf.t .- t))
    return gf.x[k]
end

fTemp = GridForcing(tgrid, df.TA)
fSW_IN = GridForcing(tgrid, df.SW_IN)

fTemp(1.0)

fNEE(u, p, t) = @SVector [p.fSW_IN(t) .* p.RUE ./ 12.011f0 .- p.Rb .* 0.1 .* p.Q10.^(p.fTemp(t) .- p.Tref) .* u[1]]
fGPP(p, t) = @SVector [p.fSW_IN(t) .* p.RUE ./ 12.011f0]

u0    = @SVector [0.0]
p0 = (;fTemp, fSW_IN, RUE = 0.1, Rb = 1.0, Q10 = 1.5, Tref = 15.0)
prob  = ODEProblem(fNEE, u0, (tgrid[1], tgrid[end]), p0)
sol = solve(prob, RK4(); p=p0, u0=u0, saveat = tgrid[1:3])

import SciMLSensitivity as SMS
function flux_part(;SW_IN, TA, RUE, Rb, Q10, tgrid, Tref)

    fTemp = GridForcing(tgrid, TA)
    fSW_IN = GridForcing(tgrid, SW_IN)

    #@show Tref
    u0 = @SVector [mean(SW_IN).* RUE ./ 12.011f0 ./ mean(Rb .* 0.1 .* Q10.^(TA .- Tref))]

    p = (;fTemp, fSW_IN, RUE, Rb, Q10, Tref)

    sol = solve(prob, RK4(); p=p, u0=u0, saveat = tgrid, sensealg=SMS.SensitivityADPassThrough())
    NEE = fNEE(sol.u, p, sol.t)
    GPP = fGPP(p, sol.t)
    RECO = NEE .+ GPP
    return (;NEE, RECO, GPP, Q10, RUE, Rb, C = sol.u)
end

flux_part(;SW_IN=df.SW_IN, TA=df.TA, RUE=p0.RUE, Rb=p0.Rb, Q10=p0.Q10, tgrid=tgrid, Tref=p0.Tref)


using Zygote
L, back = Zygote.pullback(flux_part, p0)
∂L_∂p = first(back(1.0))


function flux_part_mechanistic_model(;SW_IN, TA, RUE, Rb, Q10)
    # -------------------------------------------------------------------------
    # Arguments:
    #   SW_IN     : Incoming shortwave radiation
    #   TA      : Air temperature
    #   RUE     : Radiation Use Efficiency
    #   Rb      : Basal respiration
    #   Q10     : Temperature sensitivity 
    #
    # Returns:
    #   NEE     : Net Ecosystem Exchange
    #   RECO    : Ecosystem respiration
    #   GPP     : Gross Primary Production
    # -------------------------------------------------------------------------

    # Calculate fluxes
    GPP = SW_IN .* RUE ./ 12.011f0  # µmol/m²/s
    RECO = Rb .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
    NEE = RECO .- GPP
    
    return (;NEE, RECO, GPP, Q10, RUE)
end

# =============================================================================
# Parameter container for the mechanistic model
# =============================================================================

# Define parameter structure with bounds
parameters = (
    #            default                  lower                     upper                description
    RUE      = ( 0.1f0,                  0.0f0,                   1.0f0 ),            # Radiation Use Efficiency [g/MJ]
    Rb       = ( 1.0f0,                  0.0f0,                   6.0f0 ),            # Basal respiration [μmol/m²/s]
    Q10      = ( 1.5f0,                  1.0f0,                   4.0f0 ),            # Temperature sensitivity factor [-]
)

# =============================================================================
# Hybrid Model Creation
# =============================================================================

# Select target and forcing variables and predictors
target_FluxPartModel = [:NEE]
forcing_FluxPartModel = [:SW_IN, :TA]

# Define predictors as NamedTuple - this automatically determines neural parameter names
predictors = (Rb = [:SWC_shallow, :P, :WS, :sine_dayofyear, :cos_dayofyear], 
              RUE = [:TA, :P, :WS, :SWC_shallow, :VPD, :SW_IN_POT, :dSW_IN_POT, :dSW_IN_POT_DAY])

# Define global parameters (none for this model, Q10 is fixed)
global_param_names = [:Q10]

# Create the hybrid model using the unified constructor
hybrid_model = constructHybridModel(
    predictors,
    forcing_FluxPartModel,
    target_FluxPartModel,
    flux_part_mechanistic_model,
    parameters,
    global_param_names,
    scale_nn_outputs=true,
    hidden_layers = [15, 15],
    activation = sigmoid,
    input_batchnorm = true,
    start_from_default = false
)

# =============================================================================
# Model Training
# =============================================================================
# Train FluxPartModel
out_Generic = train(hybrid_model, df, (); nepochs=5, batchsize=512, opt=AdamW(0.01), loss_types=[:nse, :mse], training_loss=:nse, random_seed=123, yscale = identity, monitor_names=[:RUE, :Q10]);

EasyHybrid.poplot(out_Generic)
