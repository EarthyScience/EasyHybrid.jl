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

NN_RbQ10 = Chain(Dense(length(predictors_RbQ10), 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1, x -> x^2))

# Instantiate RespirationRbQ10 model
RbQ10_model = RespirationRbQ10(NN_RbQ10, predictors_RbQ10, (target_RbQ10,), (forcing_RbQ10,), 2.5f0)

# Train RbQ10 model
out_RbQ10 = train(RbQ10_model, ds_keyed_reco, (:Q10,); nepochs=10, batchsize=512, opt=Adam(0.01))

# Plot training results for RbQ10
fig_RbQ10 = Figure(size=(1200, 600))
ax_train = Makie.Axis(fig_RbQ10[1, 1], title="RbQ10 Model - Training Results", xlabel = "Time", ylabel = "RECO")
lines!(ax_train, out_RbQ10.train_obs_pred[!, Symbol(string(target_RbQ10, "_pred"))], color=:orangered, label="prediction")
lines!(ax_train, out_RbQ10.train_obs_pred[!, target_RbQ10], color=:dodgerblue, label="observation")
axislegend(ax_train; position=:lt)
fig_RbQ10

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

NNRb = Chain(Dense(length(predictors_Rb_FluxPartModel), 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1, x -> x^2))
NNRUE = Chain(Dense(length(predictors_RUE_FluxPartModel), 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1, x -> x^2))

FluxPart = FluxPartModelQ10Lux(NNRUE, NNRb, predictors_RUE_FluxPartModel, predictors_Rb_FluxPartModel, forcing_FluxPartModel, target_FluxPartModel, 2.5f0)

FluxPart.RUE_predictors

ds_keyed_FluxPartModel(predictors_Rb_FluxPartModel)

out =EasyHybrid.prepare_data(FluxPart, ds_keyed_FluxPartModel)

# Train FluxPartModel
out_FluxPart = train(FluxPart, ds_keyed_FluxPartModel, (:Q10,); nepochs=100, batchsize=512, opt=AdaGrad(0.01))

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
nee_correlation = cor(nee_pred, nee_obs)
nee_rmse = sqrt(mean((nee_pred .- nee_obs).^2))
nee_mean_diff = mean(nee_pred .- nee_obs)

ax_NEE = Makie.Axis(fig_NEE[1, 1], 
    title="FluxPartModel - NEE Predictions vs Observations
    \n Correlation: $(round(nee_correlation, digits=3)) 
    \n RMSE: $(round(nee_rmse, digits=3)) μmol/m²/s
    \n Mean Difference: $(round(nee_mean_diff, digits=3)) μmol/m²/s", 
    xlabel="Observed NEE", 
    ylabel="Predicted NEE", aspect=1)

scatter!(ax_NEE, nee_obs, nee_pred, color=:purple, alpha=0.6, markersize=8)

# Add 1:1 line
max_val = max(maximum(nee_obs), maximum(nee_pred))
min_val = min(minimum(nee_obs), minimum(nee_pred))
lines!(ax_NEE, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=1, label="1:1 line")

axislegend(ax_NEE; position=:lt)
fig_NEE

mean(ds_keyed_reco(:R_soil))

# =============================================================================
# Compare respiration from RbQ10 model vs FluxPartModel
# =============================================================================

# Get respiration predictions from both models
# For RbQ10 model, we already have the predictions in out_RbQ10.train_obs_pred
# For FluxPartModel, we need to extract the RECO component from the model output

# Get the trained parameters and states
ps_RbQ10 = out_RbQ10.ps
st_RbQ10 = out_RbQ10.st
ps_FluxPart = out_FluxPart.ps
st_FluxPart = out_FluxPart.st

# Get respiration predictions from FluxPartModel
# We need to run the model to get the RECO component
_, FluxPart_outputs = FluxPart(ds_keyed_FluxPartModel, ps_FluxPart, st_FluxPart)
reco_FluxPart = FluxPart_outputs.RECO

# Get respiration predictions from RbQ10 model
RbQ10_outputs, _ = RbQ10_model(ds_keyed_reco, ps_RbQ10, st_RbQ10)
reco_RbQ10 = RbQ10_outputs.R_soil

# Create comparison plot
fig_comparison = Figure(size=(1400, 800))

# Main comparison plot
ax_main = Makie.Axis(fig_comparison[1, 1], 
    title="Respiration Comparison: RbQ10 vs FluxPartModel", 
    xlabel="Time", 
    ylabel="Respiration (μmol/m²/s)")

# Plot both respiration estimates
lines!(ax_main, vec(reco_RbQ10), color=:dodgerblue, linewidth=2, label="RbQ10 Model")
lines!(ax_main, vec(reco_FluxPart), color=:orangered, linewidth=2, label="FluxPartModel")

axislegend(ax_main; position=:lt)

# Calculate statistics
correlation = cor(vec(reco_RbQ10)[1:min_length], vec(reco_FluxPart)[1:min_length])
rmse = sqrt(mean((vec(reco_RbQ10)[1:min_length] .- vec(reco_FluxPart)[1:min_length]).^2))


# Scatter plot comparing the two estimates
ax_scatter = Makie.Axis(fig_comparison[1, 2], 
    title="Respiration Correlation 
    \n Correlation: $(round(correlation, digits=3)) 
    \n RMSE: $(round(rmse, digits=3)) μmol/m²/s", 
    xlabel="RbQ10 Respiration", 
    ylabel="FluxPartModel Respiration")

# Use the shorter length to match data
min_length = min(length(vec(reco_RbQ10)), length(vec(reco_FluxPart)))
scatter!(ax_scatter, vec(reco_RbQ10)[1:min_length], vec(reco_FluxPart)[1:min_length], 
    color=:purple, alpha=0.6, markersize=8)

# Add 1:1 line
max_val = max(maximum(vec(reco_RbQ10)), maximum(vec(reco_FluxPart)))
min_val = min(minimum(vec(reco_RbQ10)), minimum(vec(reco_FluxPart)))
lines!(ax_scatter, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=1, label="1:1 line")

axislegend(ax_scatter; position=:lt)

fig_comparison

