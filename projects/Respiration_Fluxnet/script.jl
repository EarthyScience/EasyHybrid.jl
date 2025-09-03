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

site = "US-SRG"

fluxnet_data = load_fluxnet_nc(joinpath(project_path, "Data", "data20240123", "$site.nc"), timevar="date")

fluxnet_data.timeseries.dayofyear = dayofyear.(fluxnet_data.timeseries.time)
fluxnet_data.timeseries.sine_dayofyear = sin.(fluxnet_data.timeseries.dayofyear)
fluxnet_data.timeseries.cos_dayofyear = cos.(fluxnet_data.timeseries.dayofyear)

# explore data structure
println(names(fluxnet_data.timeseries))
println(fluxnet_data.scalars)
println(names(fluxnet_data.profiles))

# select timeseries data
df = fluxnet_data.timeseries


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
target_FluxPartModel = [:NEE]
forcing_FluxPartModel = [:SW_IN, :TA]

predictors = (Rb = [:SWC_shallow, :P, :WS], 
              RUE = [:TA, :P, :WS, :SWC_shallow, :VPD, :SW_IN_POT, :dSW_IN_POT, :dSW_IN_POT_DAY])

global_param_names = [:Q10]

hybrid_model = constructHybridModel(
    predictors,
    forcing_FluxPartModel,
    target_FluxPartModel,
    flux_part_mechanistic_model,
    parameters,
    global_param_names,
    scale_nn_outputs=true,
    hidden_layers = [32, 32],
    activation = tanh,
    input_batchnorm = true,
    start_from_default = false
)

# =============================================================================
# Model Training
# =============================================================================

for site in ["US-SRG"]
    fluxnet_data = load_fluxnet_nc(joinpath(project_path, "Data", "data20240123", "$site.nc"), timevar="date")
    df = fluxnet_data.timeseries

    out_Generic = train(
        hybrid_model, df, ();
        nepochs = 100,
        batchsize = 512,
        opt = RMSProp(0.01),
        loss_types = [:nse, :mse],
        training_loss = :nse,
        random_seed = 123,
        yscale = identity,
        monitor_names = [:RUE, :Q10],
        patience = 50,
        shuffleobs = true,
        hybrid_name = site,
        plotting = false,
        folder_to_save = site
    )
end

EasyHybrid.poplot(out_Generic)

# forward model Run

forward_run = hybrid_model(df, out_Generic.ps, out_Generic.st)

forward_run.NEE_pred
forward_run.GPP_pred
forward_run.RECO_pred
forward_run.RUE_pred

# https://nrennie.rbind.io/blog/introduction-julia-r-users/
using TidierPlots
using WGLMakie
beautiful_makie_theme = Attributes(
    fonts=(;regular="CMU Serif"),
)
ggplot(forward_run, aes(x=:GPP_NT, y=:GPP_pred)) + geom_point() + beautiful_makie_theme

idx = .!isnan.(forward_run.GPP_NT) .& .!isnan.(forward_run.GPP_pred)
EasyHybrid.poplot(forward_run.GPP_NT[idx], forward_run.GPP_pred[idx], "GPP", xlabel = "Nighttime GPP", ylabel = "Hybrid GPP")