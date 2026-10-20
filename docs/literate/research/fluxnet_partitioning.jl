# [![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
#
# # EasyHybrid Example: NEE Flux Partitioning at DE-Hai
#
# This example trains a hybrid model that partitions the measured net ecosystem
# exchange (NEE) into gross primary production (GPP) and ecosystem respiration (RECO)
# at the FLUXNET site DE-Hai (Hainich, Germany). The NEE target alone constrains
# both fluxes: a light-use-efficiency model ``GPP = RUE ⋅ SW_{IN}`` and
# ``NEE = RECO - GPP``, where ``RUE`` and ``RECO`` are predicted by two neural networks.
#
# The data come from the [FLUXNET Shuttle](https://github.com/fluxnet/shuttle), which
# serves ONEFlux v1.3 FLUXNET products from the AmeriFlux, ICOS and TERN data hubs.
# The data are shared under CC-BY-4.0; see `DATA_POLICY_LICENSE_AND_INSTRUCTIONS.txt`
# in the downloaded archive for the attribution requirements.

using EasyHybrid
using Dates
using CairoMakie

# ## Data Download via the FLUXNET Shuttle
#
# [`load_fluxnet_shuttle`](@ref) fetches the FLUXNET archive of a site from the ICOS
# Carbon Portal, the ICOS data hub of the shuttle, and reads its half-hourly fluxes and
# meteorology (`FLUXMET_HH`). It only needs Julia: the shuttle's Python client is not
# required, and the `https_proxy` environment variable is respected.
#
# The archive always holds the full record (2000–2025 and ≈260 MB for DE-Hai), so
# `years` only limits what is read and trained on. It is cached in `FLUXNET_DATA_DIR`, by default
# `joinpath(tempdir(), "fluxnet_shuttle")`. A zip that you downloaded with the
# `fluxnet-shuttle` command line tool (also from AmeriFlux or TERN) is picked up as
# well if you point `FLUXNET_DATA_DIR` to its output folder.
#
# We read only the variables we need. `_F` variables are gap-filled;
# `NEE_VUT_USTAR50_QC == 0` flags measured (not gap-filled) NEE. The NT/DT
# partitioning results serve as reference.

site = "DE-Hai"
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

# ## Feature Engineering
#
# [`flux_features`](@ref) turns the raw half-hourly data into the predictors of the two
# neural networks:
#
#   - meteorology: air temperature `TA`, radiation `SW_IN`, potential radiation
#     `SW_IN_POT` and its change `SW_IN_POT_diff`, and relative humidity `RH` from VPD;
#   - daily context from the NEE observations themselves: moving medians over 49
#     half-hours (about one day) of night-time NEE and temperature
#     (`NEE_night_movmean`, `TA_night_movmean`), the night–day NEE difference
#     (`NEE_light_night_diff`) and the local light response of NEE below 300 W/m²
#     (`NEE_SWin_lowlight_slope`);
#   - soil-moisture proxies `SWC_capa_*`: buckets of 1 to 625 mm capacity filled by
#     precipitation and emptied by evapotranspiration ([`soil_bucket`](@ref));
#   - diurnal and seasonal harmonics (`sinHR`, `cosHR`, `sin_doy_k`, `cos_doy_k`).
#
# The target `NEE` keeps only measured values (`NEE_VUT_USTAR50_QC == 0`); all predictors
# are gap-filled by linear interpolation ([`fillgaps`](@ref)).

df = flux_features(raw);
first(df, 5)

# ## Define the Physical Model
#
# Light-use-efficiency model for GPP and the NEE mass balance:
#
#   - SW_IN: incoming shortwave radiation [W/m²]
#   - RUE: radiation use efficiency, predicted by a neural network
#   - RECO: ecosystem respiration [μmol/m²/s], predicted by a neural network

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

# ## Configure Hybrid Model Components
#
# Each NN-predicted parameter gets its own network and predictors: RECO sees night-time
# NEE and temperature, RUE sees radiation, humidity and light-response features. Both
# see the soil-moisture buckets and the diurnal and seasonal harmonics.

soilwater = Symbol.("SWC_capa_" .* string.((1, 5, 25, 125, 625)))
harmonics = [:sinHR, :cosHR, :sin_doy_1, :cos_doy_1, :sin_doy_2, :cos_doy_2]

predictors = (
    RECO = [:NEE_night_movmean, :TA, :TA_night_movmean, soilwater..., harmonics...],
    RUE = [
        :TA, :RH, :SW_IN, :SW_IN_POT, :SW_IN_POT_diff,
        :NEE_SWin_lowlight_slope, :NEE_light_night_diff, soilwater..., harmonics...,
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
    activation = relu,
    input_batchnorm = true,
    start_from_default = false,
)

# ## Train the Hybrid Model
#
# Rows without measured NEE (gap-filled or failed QC) are dropped automatically.

out = train(
    hybrid_model,
    df;
    nepochs = 100,
    batchsize = 480,
    opt = Adam(0.03),
    loss_types = [:nse, :mse],
    training_loss = :mse,
    random_seed = 123,
    yscale = identity,
    monitor_names = [:RUE, :RECO, :GPP],
    patience = 30,
    shuffleobs = true,
    show_progress = false,
    model_name = "FluxPart_$(site)",
)

# ## Partitioned Fluxes
#
# Forward run over the full period, including the gaps, and comparison with the
# night-time (NT) and day-time (DT) partitioning of the FLUXNET product.

df_in = select(df, Not(:DateTime))
df_in.NEE = ifelse.(isfinite.(df_in.NEE), df_in.NEE, 0.0f0)
pred = hybrid_model(df_in, out.ps, out.st)
pred.NEE = df.NEE
pred.DateTime = df.DateTime

finite_cor(a, b) = (ok = isfinite.(a) .& isfinite.(b); cor(a[ok], b[ok]))
DataFrame(
    flux = ["NEE (measured)", "RECO vs NT", "RECO vs DT", "GPP vs NT", "GPP vs DT"],
    correlation = [
        finite_cor(pred.NEE_pred, pred.NEE),
        finite_cor(pred.RECO_pred, pred.RECO_NT_VUT_USTAR50),
        finite_cor(pred.RECO_pred, pred.RECO_DT_VUT_USTAR50),
        finite_cor(pred.GPP_pred, pred.GPP_NT_VUT_USTAR50),
        finite_cor(pred.GPP_pred, pred.GPP_DT_VUT_USTAR50),
    ],
)

# Daily means of the partitioned fluxes. The measured NEE average covers only the
# half-hours that passed QC, which are mostly daytime, so it is biased towards uptake.

finite_mean(x) = (v = filter(isfinite, x); isempty(v) ? NaN : mean(v))
pred.Date = Date.(pred.DateTime)
daily = combine(
    groupby(pred, :Date),
    [
        :NEE, :NEE_pred, :RECO_pred, :RECO_NT_VUT_USTAR50, :RECO_DT_VUT_USTAR50,
        :GPP_pred, :GPP_NT_VUT_USTAR50, :GPP_DT_VUT_USTAR50,
    ] .=> finite_mean,
    renamecols = false,
)

fig = Figure(size = (1000, 800))
panels = [
    ("NEE", [:NEE => "measured", :NEE_pred => "hybrid"]),
    ("RECO", [:RECO_NT_VUT_USTAR50 => "NT", :RECO_DT_VUT_USTAR50 => "DT", :RECO_pred => "hybrid"]),
    ("GPP", [:GPP_NT_VUT_USTAR50 => "NT", :GPP_DT_VUT_USTAR50 => "DT", :GPP_pred => "hybrid"]),
]
for (i, (flux, series)) in enumerate(panels)
    ax = Axis(fig[i, 1], ylabel = "$flux [μmol/m²/s]", xlabelvisible = false, xticklabelsvisible = i == length(panels))
    for (col, label) in series
        lines!(ax, daily.Date, daily[!, col], label = label)
    end
    axislegend(ax, position = :lt, framevisible = false)
end
fig
