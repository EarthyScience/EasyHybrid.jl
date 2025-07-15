# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
Pkg.activate("projects/Respiration_Fluxnet")
Pkg.develop(path=pwd())

manifest_path = joinpath(pwd(), "Manifest.toml")
if !isfile(manifest_path)
    Pkg.instantiate()
end

# start using the package
using EasyHybrid
using AxisKeys

include("Data/load_data.jl")

# =============================================================================
# Load data
# =============================================================================

# copy data to data/data20240123/ from here /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123
# or adjust the path to /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123 + FluxNetSite

site = load_fluxnet_nc(joinpath(@__DIR__, "Data", "data20240123", "US-SRG.nc"), timevar="date")

site.timeseries.dayofyear = dayofyear.(site.timeseries.time)
site.timeseries.sine_dayofyear = sin.(site.timeseries.dayofyear)
site.timeseries.cos_dayofyear = cos.(site.timeseries.dayofyear)

# explore data structure
println(names(site.timeseries))
println(site.scalars)
println(names(site.profiles))

df = copy(site.timeseries[!, Not(:time, :date)])

# =============================================================================
# Targets, Forcing and Predictors definition
# =============================================================================

# Select target and forcing variables and predictors
target_FluxPartModel = [:NEE]
forcing_FluxPartModel = [:SW_IN, :TA]

# Define predictors as NamedTuple - this automatically determines neural parameter names
predictors = (Rb = [:SWC_shallow, :P, :WS, :sine_dayofyear, :cos_dayofyear], 
              RUE = [:TA, :P, :WS, :SWC_shallow, :VPD, :SW_IN_POT, :dSW_IN_POT, :dSW_IN_POT_DAY])

# =============================================================================
# More Data Processing and Creation of KeyedArray
# =============================================================================

# Flatten predictors to get all unique column names
all_predictor_cols = unique(vcat(values(predictors)...))
col_to_select = unique([all_predictor_cols..., forcing_FluxPartModel..., target_FluxPartModel...])

# select columns and drop rows with any NaN values
sdf = copy(df[!, col_to_select])
dropmissing!(sdf)

for col in names(sdf)
    T = eltype(sdf[!, col])
    if T <: Union{Missing, Real} || T <: Real
        sdf[!, col] = Float64.(coalesce.(sdf[!, col], NaN))
    end
end

ds_keyed_FluxPartModel = to_keyedArray(Float32.(sdf))

# =============================================================================
# Parameter container for the mechanistic model
# =============================================================================

# Parameter structure for FluxPartModel
struct FluxPartParams <: AbstractHybridModel 
    hybrid::EasyHybrid.ParameterContainer
end

# Define parameter structure with bounds
parameters = (
    #            default                  lower                     upper                description
    RUE      = ( 0.1f0,                  0.0f0,                   1.0f0 ),            # Radiation Use Efficiency [g/MJ]
    Rb       = ( 1.0f0,                  0.0f0,                   6.0f0 ),            # Basal respiration [μmol/m²/s]
    Q10      = ( 1.5f0,                  1.0f0,                   4.0f0 ),            # Temperature sensitivity factor [-]
)

parameter_container = build_parameters(parameters, FluxPartParams)


# =============================================================================
# Mechanistic Model Definition
# =============================================================================

function flux_part_mechanistic_model(sw_in, ta; RUE, Rb, Q10)
    # -------------------------------------------------------------------------
    # Arguments:
    #   sw_in   : Incoming shortwave radiation
    #   ta      : Air temperature
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
    GPP = sw_in .* RUE ./ 12.011f0  # µmol/m²/s
    RECO = Rb .* Q10 .^ (0.1f0 .* (ta .- 15.0f0))
    NEE = RECO .- GPP
    
    return (;NEE, RECO, GPP)
end

# KeyedArray version needed for hybrid model
function flux_part_mechanistic_model(forcing_data::KeyedArray; RUE, Rb, Q10)
    sw_in = vec(forcing_data([:SW_IN]))  # SW_IN
    ta = vec(forcing_data([:TA]))     # TA
    return flux_part_mechanistic_model(sw_in, ta; RUE, Rb, Q10)
end

function flux_part_mechanistic_model(forcing_data::KeyedArray, parameter_container::FluxPartParams)
    return flux_part_mechanistic_model(forcing_data; values(default(parameter_container))...)
end



# =============================================================================
# Plot with defaults
# =============================================================================

o_def = flux_part_mechanistic_model(ds_keyed_FluxPartModel, parameter_container)

using GLMakie
GLMakie.activate!(inline=false)
fig = Figure()
ax = Makie.Axis(fig[1, 1], title="NEE", xlabel="Time", ylabel="NEE")
lines!(ax, ds_keyed_FluxPartModel(:NEE))
lines!(ax, o_def.NEE)
hidexdecorations!(ax)

ax = Makie.Axis(fig[2, 1], title="RECO, GPP", xlabel="Time", ylabel="RECO, GPP")
lines!(ax, o_def.RECO)
lines!(ax, -o_def.GPP)
linkxaxes!(filter(x -> x isa Makie.Axis, fig.content)...)

fig


# =============================================================================
# Hybrid Model Creation
# =============================================================================

# recall forcing and target variables
forcing_FluxPartModel
target_FluxPartModel

# recall predictors
predictors

# Neural parameter names are automatically determined from predictor names
neural_param_names = collect(keys(predictors))

# Define global parameters (none for this model, Q10 is fixed)
global_param_names = [:Q10]

# Create the hybrid model using the unified constructor
hybrid_model = constructHybridModel(
    predictors,
    forcing_FluxPartModel,
    target_FluxPartModel,
    flux_part_mechanistic_model,
    parameter_container,
    neural_param_names,
    global_param_names,
    scale_nn_outputs=false
)

# =============================================================================
# Model Training
# =============================================================================

# Train FluxPartModel
out_FluxPart = train(hybrid_model, ds_keyed_FluxPartModel, (); nepochs=10, batchsize=512, opt=AdamW(0.01), loss_types=[:mse, :nse], training_loss=:mse);

# =============================================================================
# Results Visualization
# =============================================================================

using GLMakie
GLMakie.activate!(inline=false)

# Plot training results for FluxPartModel
fig_FluxPart = Figure(size=(1200, 600))
ax_train = Makie.Axis(fig_FluxPart[1, 1], title="FluxPartModel (New) - Training Results", xlabel = "Time", ylabel = "NEE")
lines!(ax_train, out_FluxPart.val_obs_pred[!, Symbol(string(:NEE, "_pred"))], color=:orangered, label="prediction")
lines!(ax_train, out_FluxPart.val_obs_pred[!, :NEE], color=:dodgerblue, label="observation")
axislegend(ax_train; position=:lt)
fig_FluxPart

# Plot the NEE predictions as scatter plot
fig_NEE = Figure(size=(800, 600))

# Calculate NEE statistics
nee_pred = out_FluxPart.val_obs_pred[!, Symbol(string(:NEE, "_pred"))]
nee_obs = out_FluxPart.val_obs_pred[!, :NEE]
ss_res = sum((nee_obs .- nee_pred).^2)
ss_tot = sum((nee_obs .- mean(nee_obs)).^2)
nee_modelling_efficiency = 1 - ss_res / ss_tot
nee_rmse = sqrt(mean((nee_pred .- nee_obs).^2))

ax_NEE = Makie.Axis(fig_NEE[1, 1], 
    title="FluxPartModel (New) - NEE Predictions vs Observations
    \n Modelling Efficiency: $(round(nee_modelling_efficiency, digits=3)) 
    \n RMSE: $(round(nee_rmse, digits=3)) μmol CO2 m-2 s-1",
    xlabel="Predicted NEE", 
    ylabel="Observed NEE", aspect=1)

scatter!(ax_NEE, nee_pred, nee_obs, color=:purple, alpha=0.1, markersize=8)

# Add 1:1 line
max_val = max(maximum(nee_obs), maximum(nee_pred))
min_val = min(minimum(nee_obs), minimum(nee_pred))
lines!(ax_NEE, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=1, label="1:1 line")

axislegend(ax_NEE; position=:lt)
fig_NEE
