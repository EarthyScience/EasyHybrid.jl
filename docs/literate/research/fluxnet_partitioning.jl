# [![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
#
# # EasyHybrid Example: NEE Flux Partitioning at ES-LM1
#
# One year (2018) at ES-LM1 (Majadas de Tiétar, Spain). Measured NEE is
# partitioned into GPP and RECO by ``GPP = RUE ⋅ SW_{IN}`` and
# ``NEE = RECO - GPP``. Two neural networks predict ``RUE`` and ``RECO``.
#
# Data from the [FLUXNET Shuttle](https://github.com/fluxnet/shuttle)
# (ONEFlux v1.3, CC-BY-4.0).

# using Pkg
# Pkg.activate("/User/homes/bahrens/BGI/scratch/bahrens/projects/EasyHybrid_Xin/docs")
# Pkg.develop(path = "/Net/Groups/BGI/scratch/bahrens/projects/EasyHybrid_Xin")

using EasyHybrid
using CairoMakie

# ## Data
#
# [`load_fluxnet_shuttle`](@ref) reads half-hourly fluxes and meteorology for 2018.
# `_F` columns are gap-filled. `NEE_VUT_USTAR50_QC == 0` marks measured NEE.
# NT and DT products are the reference.

site = "ES-LM1"
years = 2018:2018

raw = load_fluxnet_shuttle(
    site;
    years,
    variables = [
        :TA_F, :SW_IN_F, :SW_IN_POT, :VPD_F, :P_F, :LE_F_MDS,
        :NEE_VUT_USTAR50, :NEE_VUT_USTAR50_QC,
        :RECO_NT_VUT_USTAR50, :GPP_NT_VUT_USTAR50, :RECO_DT_VUT_USTAR50, :GPP_DT_VUT_USTAR50,
    ],
);
size(raw)

# ## Features
#
# [`flux_features`](@ref) builds the predictors: meteorology, daily night-time
# NEE context, light-response slopes, soil-water buckets ([`soil_bucket`](@ref)),
# and diurnal and seasonal harmonics. The target keeps measured NEE only.
# `min_x` and `min_range` drop light-response slopes from windows that stay below
# 200 W/m² or span less than 50 W/m². Pass `min_x = 0, min_range = 0` to keep them.

df = flux_features(raw);
first(df, 5)

# ## Process-based Model
#
# `GPP` from incoming radiation and NN `RUE`; `NEE = RECO - GPP`.

function flux_part_mechanistic_model(; SW_IN, RECO, RUE)
    ## µmol/m2/s = J/s/m2 * g/MJ / g/mol
    GPP = SW_IN .* RUE ./ 12.011f0
    NEE = RECO .- GPP
    return (; NEE, RECO, GPP, RUE)
end

# Parameter specification: (default, lower_bound, upper_bound)

parameters = (
    RUE = (0.1f0, 0.0f0, 4.0f0),
    RECO = (1.0f0, 0.0f0, 100.0f0),
)

# ## Hybrid Model
#
# One network each for `RECO` and `RUE`. Both see soil water and harmonics.

soilwater = Symbol.("SWC_capa_" .* string.((1, 5, 25, 125, 625)))
diurnal = [:cosHR, :sinHR]
seasonal = Symbol.(vec(string.(["sin_" "cos_"], 1.0:5.0, "_year")))

predictors = (
    RECO = [:NEE_night_movmean, :TA, :TA_night_movmean, soilwater..., seasonal..., diurnal...],
    RUE = [
        :TA, :RH_F, :SW_IN, :SW_IN_POT, :SW_IN_POT_diff,
        :NEE_SWin_lowlight_lm_slope, :NEE_RUE_at_max_SW, :NEE_light_night_diff,
        soilwater..., diurnal..., seasonal...,
    ],
)

forcing = [:SW_IN]
target = [:NEE]
global_param_names = Symbol[]

hybrid_model = constructHybridModel(
    predictors,                 # Input features per NN
    forcing,                    # Forcing variables
    target,                     # Target variables
    flux_part_mechanistic_model, # Process-based model function
    parameters,                 # Parameter definitions
    global_param_names,         # Global parameters
    scale_nn_outputs = true,
    hidden_layers = [15],
    activation = tanh,
    input_batchnorm = true,
    start_from_default = false,
)

# ## Training
#
# Rows without measured NEE are dropped.

out = train(
    hybrid_model,
    df;
    nepochs = 100,
    batchsize = 512,
    opt = Adam(0.03),
    loss_types = [:nse, :mse],
    training_loss = :mse,
    random_seed = 123,
    yscale = identity,
    monitor_names = [:RUE, :RECO, :GPP],
    patience = 30,
    shuffleobs = true,
    show_progress = false,
    plotting = false,
    save_training = false,
    model_name = "FluxPart_$(site)",
);

# ## Partitioned Fluxes
#
# Predict every half-hour of 2018. `DateTime` is left out because the forward
# pass builds one numeric matrix. Row order matches `raw`, and measured NEE is
# read from `raw`.

pred = hybrid_model(select(df, Not(:DateTime)), out.ps, out.st);


# [`plot_flux_partition`](@ref) shows the same series at daily and hourly scale.
# Measured NEE uses QC-passed half-hours only, mostly daytime, so a daily mean
# is biased toward uptake. The time axis is ticked once a month, and the year
# is written only in front of January.

daily = combine_flux_partition(raw, pred; scale = :daily, labels = ["UN"]);
fig = plot_flux_partition(raw, pred; scale = :daily, labels = ["UN"])

# ## Diurnal cycle

qc = isfinite.(df.NEE)
diurnal = DataFrame(
    DateTime = df.DateTime[qc],
    NEE = df.NEE[qc],
    NEE_hybrid = vec(pred.NEE_pred)[qc],
    NEE_DT = (df.RECO_DT_VUT_USTAR50 .- df.GPP_DT_VUT_USTAR50)[qc],
    SW_IN = df.SW_IN[qc],
    SW_IN_POT = df.SW_IN_POT[qc],
)
diurnal.RECO_NT = df.RECO_NT_VUT_USTAR50[qc]
fig_diurnal = diurnalPlot(
    diurnal;
    yvars = [:NEE, :NEE_hybrid, :NEE_DT, :RECO_NT],
    obs = :NEE,
    ylabel = "NEE [μmol m⁻² s⁻¹]",
)
