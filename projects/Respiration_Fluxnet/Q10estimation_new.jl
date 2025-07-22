# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
project_path = "projects/Respiration_Fluxnet"
Pkg.activate(project_path)

manifest_path = joinpath(project_path, "Manifest.toml")
if !isfile(manifest_path)
    Pkg.develop(path=pwd())
    Pkg.instantiate()
end

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

df = load_fluxnet_nc(joinpath(project_path, "Data", "data20240123", "US-SRG.nc"), timevar="date")

df.timeseries.dayofyear = dayofyear.(df.timeseries.time)
df.timeseries.sine_dayofyear = sin.(df.timeseries.dayofyear)
df.timeseries.cos_dayofyear = cos.(df.timeseries.dayofyear)

# explore data structure
println(names(df.timeseries))
println(df.scalars)
println(names(df.profiles))

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
    
    return (;NEE, RECO, GPP)
end

mech_model = construct_dispatch_functions(flux_part_mechanistic_model)

out_test = mech_model(df, parameter_container, forcing_FluxPartModel)

# =============================================================================
# Plot with defaults
# =============================================================================
Figure()
fig = Figure()
if nameof(Makie.current_backend()) == :WGLMakie # TODO for our CPU cluster - alternatives?
    sleep(2.0) 
end
ax = Makie.Axis(fig[1, 1], title="NEE", xlabel="Time", ylabel="NEE")
lines!(ax, df[!, :NEE])
lines!(ax, out_test.NEE)
hidexdecorations!(ax)

ax = Makie.Axis(fig[2, 1], title="RECO, GPP", xlabel="Time", ylabel="RECO, GPP")
lines!(ax, out_test.RECO)
lines!(ax, -out_test.GPP)
linkxaxes!(filter(x -> x isa Makie.Axis, fig.content)...)


# =============================================================================
# Hybrid Model Creation
# =============================================================================

# recall forcing and target variables
forcing_FluxPartModel
target_FluxPartModel

# recall predictors
predictors

# Define global parameters (none for this model, Q10 is fixed)
global_param_names = [:Q10]

# Create the hybrid model using the unified constructor
hybrid_model = constructHybridModel(
    predictors,
    forcing_FluxPartModel,
    target_FluxPartModel,
    mech_model,
    parameter_container,
    global_param_names,
    scale_nn_outputs=false,
    hidden_layers = [15, 15],
    activation = sigmoid,
    input_batchnorm = true
)

# =============================================================================
# Model Training
# =============================================================================

ps, st = LuxCore.setup(Random.default_rng(), hybrid_model)
ps_st = (ps, st)
ps_st2 = deepcopy(ps_st)

hybrid_model(ds_keyed_FluxPartModel, ps, st)

# Train FluxPartModel
out_Generic = train(hybrid_model, df, (); nepochs=30, batchsize=512, opt=AdamW(0.01), loss_types=[:nse, :mse], training_loss=:nse, random_seed=123, train_from=ps_st, yscale = identity);

# =============================================================================
# train hybrid FluxPartModel_Q10_Lux model on NEE to get Q10, GPP, and Reco
# =============================================================================

NNRb = Chain(BatchNorm(length(predictors.Rb), affine=false), Dense(length(predictors.Rb), 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1))
NNRUE = Chain(BatchNorm(length(predictors.Rb), affine=false), Dense(length(predictors.Rb), 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1))

FluxPart = FluxPartModelQ10Lux(NNRUE, NNRb, predictors.RUE, predictors.Rb, forcing_FluxPartModel, target_FluxPartModel, Q10start)

ps_st2[1].Q10 .= Q10start

# Train FluxPartModel
out_Individual = train(FluxPart, ds_keyed_FluxPartModel, (:Q10,); nepochs=30, batchsize=512, opt=AdamW(0.01), loss_types=[:nse, :mse], training_loss=:nse, random_seed=123, train_from=ps_st2, yscale = identity);

# =============================================================================
# Results Visualization
# =============================================================================

# Plot training results for FluxPartModel
fig_FluxPart = Figure(size=(1200, 600))
ax_train = Makie.Axis(fig_FluxPart[1, 1], title="FluxPartModel (New) - Training Results", xlabel = "Time", ylabel = "NEE")
lines!(ax_train, out_Generic.val_obs_pred[!, Symbol(string(:NEE, "_pred"))], color=:orangered, label="generic model")
lines!(ax_train, out_Generic.val_obs_pred[!, :NEE], color=:dodgerblue, label="observation")
lines!(ax_train, out_Individual.val_obs_pred[!, Symbol(string(:NEE, "_pred"))], color=:green, label="individual model")
axislegend(ax_train; position=:lt)


# Plot the NEE predictions as scatter plot
fig_NEE = Figure(size=(800, 600))

EasyHybrid.poplot!(fig_NEE, out_Generic.val_obs_pred[!, :NEE], out_Generic.val_obs_pred[!, Symbol(string(:NEE, "_pred"))], "generic model", 1, 1)
EasyHybrid.poplot!(fig_NEE, out_Individual.val_obs_pred[!, :NEE], out_Individual.val_obs_pred[!, Symbol(string(:NEE, "_pred"))], "individual model", 1, 2)


