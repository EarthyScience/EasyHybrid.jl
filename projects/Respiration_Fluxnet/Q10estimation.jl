using Pkg
Pkg.activate("projects/Respiration_Fluxnet")
Pkg.develop(path=pwd())
Pkg.instantiate()

# start using the package
using EasyHybrid

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

# =============================================================================
# Create a figure and plot RECO_NT and RECO_DT time series
# =============================================================================
using GLMakie
GLMakie.activate!(inline=false)  # use non-inline (external) window for plots

# fig1 = Figure()

# ax1 = fig1[1, 1] = Makie.Axis(fig1; ylabel = "RECO")
# lines!(site.timeseries.time, site.timeseries.RECO_NT, label = "RECO_NT")
# lines!(site.timeseries.time, site.timeseries.RECO_DT, label = "RECO_DT")
# fig1[1, 2] = Legend(fig1, ax1, framevisible = false)
# hidexdecorations!()

# fig1[2,1] = Makie.Axis(fig1; ylabel = "SWC")
# lines!(site.timeseries.time, site.timeseries.SWC_shallow, label = "SWC_shallow")
# hidexdecorations!()

# fig1[3,1] = Makie.Axis(fig1; ylabel = "Precipitation")
# lines!(site.timeseries.time, site.timeseries.P, label = "P")
# hidexdecorations!()

# ax4 = fig1[4,1] = Makie.Axis(fig1; xlabel = "Time", ylabel = "Temperature")
# lines!(site.timeseries.time, site.timeseries.TA, label = "air")
# lines!(site.timeseries.time, site.timeseries.TS_shallow, label = "soil")
# fig1[4, 2] = Legend(fig1, ax4, framevisible = false)

# linkxaxes!(filter(x -> x isa Makie.Axis, fig1.content)...)

# fig1

# =============================================================================
# train hybrid Q10 model on daytime and nightime method RECO
# =============================================================================

# Data preprocessing for RECO models
# Collect all available variables and create keyed array
available_vars = names(site.timeseries);
println("Available variables: ", available_vars)

df = copy(site.timeseries[!, Not(:time, :date)])
rename!(df, :RECO_NT => :R_soil)

# Select target and forcing variables and predictors
target_RbQ10 = :R_soil
forcing_RbQ10 = :TA

df.sine_dayofyear = sin.(df.dayofyear)
df.cos_dayofyear = cos.(df.dayofyear)

predictors_RbQ10 = [:SWC_shallow, :P, :WS, :sine_dayofyear, :cos_dayofyear] # similar to Tramontana et al. 2020 - wind direction is missing

# select columns and drop rows with any NaN values
sdf = copy(df[!, [predictors_RbQ10..., target_RbQ10, forcing_RbQ10]])
dropmissing!(sdf)

for col in names(sdf)
    T = eltype(sdf[!, col])
    if T <: Union{Missing, Real} || T <: Real
        sdf[!, col] = Float64.(coalesce.(sdf[!, col], NaN))
    end
end

ds_keyed_reco = to_keyedArray(Float32.(sdf))

# TODO check batchnorm
NN_RbQ10 = Chain(BatchNorm(length(predictors_RbQ10), affine = false), Dense(length(predictors_RbQ10), 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1, x -> x^2))

# Instantiate RespirationRbQ10 model
RbQ10_model = RespirationRbQ10(NN_RbQ10, predictors_RbQ10, (target_RbQ10,), (forcing_RbQ10,), 1.5f0)

# ? test lossfn
ps, st = LuxCore.setup(Random.default_rng(), RbQ10_model)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_p_f, ds_t = EasyHybrid.prepare_data(RbQ10_model, ds_keyed_reco)
ds_t_nan = .!isnan.(ds_t)
ls = EasyHybrid.lossfn(RbQ10_model, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss())

# Train RbQ10 model
out_RbQ10 = train(RbQ10_model, ds_keyed_reco, (:Q10,); nepochs=100, batchsize=512, opt=AdaGrad(0.01))

# Plot training results for RbQ10
fig_RbQ10 = Figure(size=(1200, 600))
ax_train = Makie.Axis(fig_RbQ10[1, 1], title="RbQ10 Model - Training Results", xlabel = "Time", ylabel = "RECO")
lines!(ax_train, out_RbQ10.train_obs_pred[!, Symbol(string(target_RbQ10, "_pred"))], color=:orangered, label="prediction")
lines!(ax_train, out_RbQ10.train_obs_pred[!, target_RbQ10], color=:dodgerblue, label="observation")
axislegend(ax_train; position=:lt)
fig_RbQ10

# Plot the RECO predictions as scatter plot
fig_RECO = Figure(size=(800, 600))

# Calculate RECO statistics
reco_pred = out_RbQ10.train_obs_pred[!, Symbol(string(target_RbQ10, "_pred"))]
reco_obs = out_RbQ10.train_obs_pred[!, target_RbQ10]
ss_res = sum((reco_obs .- reco_pred).^2)
ss_tot = sum((reco_obs .- mean(reco_obs)).^2)
reco_modelling_efficiency = 1 - ss_res / ss_tot
reco_rmse = sqrt(mean((reco_pred .- reco_obs).^2))

ax_RECO = Makie.Axis(fig_RECO[1, 1], 
    title="RbQ10 Model - RECO Predictions vs Observations
    \n Modelling Efficiency: $(round(reco_modelling_efficiency, digits=3)) 
    \n RMSE: $(round(reco_rmse, digits=3)) μmol CO2 m-2 s-1",
    xlabel="Predicted RECO", 
    ylabel="Observed RECO", aspect=1)
scatter!(ax_RECO, reco_pred, reco_obs, color=:purple, alpha=0.6, markersize=8)

# Add 1:1 line
max_val = max(maximum(reco_obs), maximum(reco_pred))
min_val = min(minimum(reco_obs), minimum(reco_pred))
lines!(ax_RECO, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=1, label="1:1 line")

axislegend(ax_RECO; position=:lt)

fig_RECO

# =============================================================================
# train hybrid FluxPartModel_Q10_Lux model on NEE to get Q10, GPP, and Reco
# =============================================================================

target_FluxPartModel = [:NEE]
forcing_FluxPartModel = [:SW_IN, :TA]

predictors_Rb_FluxPartModel = [:SWC_shallow, :P, :WS, :sine_dayofyear, :cos_dayofyear]
predictors_RUE_FluxPartModel = [:TA, :P, :WS, :SWC_shallow, :VPD, :SW_IN_POT, :dSW_IN_POT, :dSW_IN_POT_DAY]

df[!, predictors_RUE_FluxPartModel]

# select columns and drop rows with any NaN values
sdf = copy(df[!, unique([predictors_Rb_FluxPartModel...,predictors_RUE_FluxPartModel..., forcing_FluxPartModel..., target_FluxPartModel...])])
dropmissing!(sdf)

ds_keyed_FluxPartModel = to_keyedArray(Float32.(sdf))

NNRb = Chain(BatchNorm(length(predictors_Rb_FluxPartModel), affine=false), Dense(length(predictors_Rb_FluxPartModel), 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1, x -> x^2))
NNRUE = Chain(BatchNorm(length(predictors_RUE_FluxPartModel), affine=false), Dense(length(predictors_RUE_FluxPartModel), 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1, x -> x^2))

FluxPart = FluxPartModelQ10Lux(NNRUE, NNRb, predictors_RUE_FluxPartModel, predictors_Rb_FluxPartModel, forcing_FluxPartModel, target_FluxPartModel, 1.5f0)

FluxPart.RUE_predictors

ds_keyed_FluxPartModel(predictors_Rb_FluxPartModel)

# Train FluxPartModel
out_FluxPart = train(FluxPart, ds_keyed_FluxPartModel, (:Q10,); nepochs=100, batchsize=512, opt=AdamW(0.01))

# Plot training results for FluxPartModel
fig_FluxPart = Figure(size=(1200, 600))
ax_train = Makie.Axis(fig_FluxPart[1, 1], title="FluxPartModel - Training Results", xlabel = "Time", ylabel = "NEE")
lines!(ax_train, out_FluxPart.train_obs_pred[!, Symbol(string(:NEE, "_pred"))], color=:orangered, label="prediction")
lines!(ax_train, out_FluxPart.train_obs_pred[!, :NEE], color=:dodgerblue, label="observation")
axislegend(ax_train; position=:lt)
fig_FluxPart

# Plot the NEE predictions as scatter plot
fig_NEE = Figure(size=(800, 600))

# Calculate NEE statistics
nee_pred = out_FluxPart.train_obs_pred[!, Symbol(string(:NEE, "_pred"))]
nee_obs = out_FluxPart.train_obs_pred[!, :NEE]
ss_res = sum((nee_obs .- nee_pred).^2)
ss_tot = sum((nee_obs .- mean(nee_obs)).^2)
nee_modelling_efficiency = 1 - ss_res / ss_tot
nee_rmse = sqrt(mean((nee_pred .- nee_obs).^2))

ax_NEE = Makie.Axis(fig_NEE[1, 1], 
    title="FluxPartModel - NEE Predictions vs Observations
    \n Modelling Efficiency: $(round(nee_modelling_efficiency, digits=3)) 
    \n RMSE: $(round(nee_rmse, digits=3)) μmol CO2 m-2 s-1",
    xlabel="Predicted NEE", 
    ylabel="Observed NEE", aspect=1)

scatter!(ax_NEE, nee_pred, nee_obs, color=:purple, alpha=0.6, markersize=8)

# Add 1:1 line
max_val = max(maximum(nee_obs), maximum(nee_pred))
min_val = min(minimum(nee_obs), minimum(nee_pred))
lines!(ax_NEE, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=1, label="1:1 line")

axislegend(ax_NEE; position=:lt)
fig_NEE

# look at Q10s

out_FluxPart.ps.Q10[1]
out_RbQ10.ps.Q10[1]


