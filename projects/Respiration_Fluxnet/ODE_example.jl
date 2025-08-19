# CC BY-SA 4.0
# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
project_path = "projects/ODE_example    "
Pkg.activate(project_path)

Pkg.develop(path=pwd())
Pkg.instantiate()


# start using the package
using EasyHybrid
using AxisKeys
using WGLMakie

include("projects/Respiration_Fluxnet/Data/load_data.jl")

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

using DifferentialEquations, SciMLSensitivity, Zygote

# Data grid
Δt   = diff(fluxnet_data.timeseries.time)[1]
t0   = 0.0
tend = length(fluxnet_data.timeseries.time) * Δt
tgrid = collect(t0:Δt:tend)

# piecewise-constant forcing aligned to the grid
struct GridForcing{T}; t::Vector{T}; x::Vector{T}; end
function (gf::GridForcing)(t)
    k = clamp(fld(Integer, round((t - gf.t[1]) / (gf.t[2]-gf.t[1])), 1), 1, length(gf.t))  # nearest/left index
    return gf.x[k]
end

fTemp = GridForcing(tgrid, df.TA)
fSW_IN = GridForcing(tgrid, df.SW_IN)

f(u, p, t) = @SVector [p.fSW_IN(t) .* p.RUE ./ 12.011f0 .- p.Rb .* 0.1 .* p.Q10.^(p.fTemp(t) .- Tref) .* u[1]]

u0    = @SVector [0.0]
prob  = ODEProblem(f, u0, (tgrid[1], tgrid[end]), p0)

p0 = (;Temp = mean(df.TA), SW_IN = mean(df.SW_IN), RUE = 0.1, Rb = 1.0, Q10 = 1.5)

function flux_part(;SW_IN, TA, RUE, Rb, Q10, tgrid)

    fTemp = GridForcing(tgrid, TA)
    fSW_IN = GridForcing(tgrid, SW_IN)

    u0 = @SVector [mean(SW_IN).* RUE ./ 12.011f0 ./ mean(Rb .* 0.1 .* Q10.^(TA .- Tref))]

    p = (;fTemp, fSW_IN, RUE, Rb, Q10)

    sol = solve(prob, RK4(); p=p, u0=u0, adaptive=false, dt=Δt,
                save_everystep=true,
                sensealg=SensitivityADPassThrough())
    NEE = map((u,t)->f(u,p,t), sol.u, sol.t)
    RECO = map((u,t)->h(u,p,t), sol.u, sol.t)
    GPP = NEE .+ RECO
    return (;NEE, RECO, GPP, Q10, RUE, Rb, C = u[1])
end

L, back = Zygote.pullback(loss, p0)
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
