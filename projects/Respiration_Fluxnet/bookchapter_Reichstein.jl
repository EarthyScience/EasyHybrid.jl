# CC BY-SA 4.0
# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
project_path = "projects/Respiration_Fluxnet"
Pkg.activate(project_path)

Pkg.develop(path=pwd())
Pkg.instantiate()


# start using the package
using EasyHybrid
using AxisKeys
using WGLMakie

include("Data/load_data.jl")

# =============================================================================
# Load data
# =============================================================================

# copy data to data/data20240123/ from here /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123
# or adjust the path to /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123 + FluxNetSite

fluxnet_data = load_fluxnet_nc(joinpath(project_path, "Data", "data20240123", "US-SRG.nc"), timevar="date")

# explore data structure
println(names(fluxnet_data.timeseries))
println(fluxnet_data.scalars)
println(names(fluxnet_data.profiles))

# select timeseries data
df = fluxnet_data.timeseries

# Replace missings with NaN and convert all columns to Float64 where possible
for col in names(df)
    if eltype(df[!, col]) <: Union{Missing, Number}
        df[!, col] = coalesce.(df[!, col], NaN)
        try
            df[!, col] = Float64.(df[!, col])
        catch
            # If conversion fails, leave as is
        end
    end
end


# =============================================================================
# Artificial data
# =============================================================================
function fM(;SWC, K_limit, K_inhib)
    rm_SWC = SWC ./ (SWC .+ K_limit) .* K_inhib ./ (K_inhib .+ SWC)
    return rm_SWC
end


fig = Figure()
ax = Makie.Axis(fig[1, 1], xlabel = "Time", ylabel = "SWC_shallow", title = "Time Series of SWC_shallow")
lines!(ax, df.time, df.SWC_shallow, color = :dodgerblue)

# moisture limitation
df.rm_SWC = fM.(SWC = df.SWC_shallow, K_limit = 10, K_inhib = 30)

# Scatterplot of SWC_shallow against rm_SWC
ax2 = Makie.Axis(fig[2, 1], xlabel = "SWC_shallow", ylabel = "rm_SWC", title = "SWC_shallow vs rm_SWC")
scatter!(ax2, df.SWC_shallow, df.rm_SWC, color = :purple, markersize = 6, alpha = 0.6)

function fT(;TA, Tref, Q10)
    return Q10 .^ ((TA .- Tref) ./ 10.0)
end

df.rm_T = fT.(TA = df.TA, Tref = 15, Q10 = 1.5)

# Scatterplot of TA against rm_T
ax3 = Makie.Axis(fig[3, 1], xlabel = "TA", ylabel = "rm_T", title = "TA vs rm_T")
scatter!(ax3, df.TA, df.rm_T, color = :green, markersize = 6, alpha = 0.6)

df.RECO_syn = df.rm_T .* df.rm_SWC

# Scatterplot of SWC_shallow against RECO_syn
fig3 = Figure()
ax = Makie.Axis(fig3[1, 1], xlabel = "Time", ylabel = "RECO_syn", title = "RECO_syn")
lines!(ax, df.time, df.RECO_syn, color = :red)

# Scatterplot of TA against RECO_syn, colored by SWC_shallow (soil moisture)
ax1, sc = scatter(fig3[2,1], df.TA, df.RECO_syn;
              color = df.SWC_shallow, colormap = :viridis,
              axis = (xlabel = "TA", ylabel = "RECO_syn"),
              markersize = 1)
Colorbar(fig3[2, 2], sc, label = "SWC_shallow")

function RbQ10(;Rb, Q10, TA, Tref)
    return Rb .* Q10 .^ ((TA .- Tref) ./ 10.0)
end

RECO_Rb1 = RbQ10.(Rb = 0.2, Q10 = 1.5, TA = df.TA, Tref = 15)
scatter!(ax1, df.TA, RECO_Rb1, color = :red, markersize = 6, alpha = 0.6, label = "Q10 = 1.5, Rb = 0.2") # add text at the right end of the scatter Q10 = 1.5
RECO_Rb2 = RbQ10.(Rb = 0.4, Q10 = 1.5, TA = df.TA, Tref = 15)
scatter!(ax1, df.TA, RECO_Rb2, color = :red, markersize = 6, alpha = 0.6, label = "Q10 = 1.5, Rb = 0.4") # add text at the right end of the scatter Q10 = 1.5


# =============================================================================
# Mechanistic Model Definition
# =============================================================================

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
out_Generic = train(hybrid_model, df, (); nepochs=1000, batchsize=512, opt=RMSProp(0.01), loss_types=[:nse, :mse], training_loss=:nse, random_seed=123, yscale = identity, monitor_names=[:RUE, :Q10], patience = 50, shuffleobs = true)

EasyHybrid.poplot(out_Generic)
