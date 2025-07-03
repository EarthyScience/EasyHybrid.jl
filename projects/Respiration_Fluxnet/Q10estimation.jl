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
site = load_fluxnet_nc(joinpath(@__DIR__, "Data", "data20240123", "US-SRG.nc"), timevar="date")

# explore data structure
println("Available timeseries variables:")
println(names(site.timeseries))
println("\nScalar variables:")
println(site.scalars)
println("\nProfile variables:")
println(names(site.profiles))

# =============================================================================
# Create a figure and plot RECO_NT and RECO_DT time series
# =============================================================================
using GLMakie
GLMakie.activate!(inline=false)  # use non-inline (external) window for plots

fig1 = Figure()

ax1 = fig1[1, 1] = Makie.Axis(fig1; ylabel = "RECO")
lines!(site.timeseries.time, site.timeseries.RECO_NT, label = "RECO_NT")
lines!(site.timeseries.time, site.timeseries.RECO_DT, label = "RECO_DT")
fig1[1, 2] = Legend(fig1, ax1, framevisible = false)
hidexdecorations!()

fig1[2,1] = Makie.Axis(fig1; ylabel = "SWC")
lines!(site.timeseries.time, site.timeseries.SWC_shallow, label = "SWC_shallow")
hidexdecorations!()

fig1[3,1] = Makie.Axis(fig1; ylabel = "Precipitation")
lines!(site.timeseries.time, site.timeseries.P, label = "P")
hidexdecorations!()

ax4 = fig1[4,1] = Makie.Axis(fig1; xlabel = "Time", ylabel = "Temperature")
lines!(site.timeseries.time, site.timeseries.TA, label = "air")
lines!(site.timeseries.time, site.timeseries.TS_shallow, label = "soil")
fig1[4, 2] = Legend(fig1, ax4, framevisible = false)

linkxaxes!(filter(x -> x isa Makie.Axis, fig1.content)...)

fig1

# =============================================================================
# train hybrid Q10 model on daytime and nightime method RECO
# =============================================================================

# Data preprocessing for RECO models
# Collect all available variables and create keyed array
available_vars = names(site.timeseries)
println("Available variables: ", available_vars)

df = site.timeseries[!, Not(:time, :date)]
rename!(df, :RECO_NT => :Rh)

# Select target and forcing variables and predictors
target_RbQ10 = :Rh
forcing_RbQ10 = :TA

predictors_RbQ10 = [:SWC_shallow, :P]

# select columns and drop rows with any NaN values
sdf = df[!, [predictors_RbQ10..., target_RbQ10, forcing_RbQ10]]
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
ax_train = Makie.Axis(fig_RbQ10[1, 1], title="RbQ10 Model - Training Results")
lines!(ax_train, out_RbQ10.train_obs_pred[!, Symbol(string(target_RbQ10, "_pred"))], color=:orangered, label="prediction")
lines!(ax_train, out_RbQ10.train_obs_pred[!, target_RbQ10], color=:dodgerblue, label="observation")
axislegend(ax_train; position=:lt)
Label(fig_RbQ10[0,1], "RbQ10 Model: Observations vs Predictions", tellwidth=false)
fig_RbQ10

# =============================================================================
# train hybrid FluxPartModel_Q10_Lux model on NEE to get Q10, GPP, and Reco
# =============================================================================
#TODO implement and check
# Data preprocessing for FluxPartModel
# Use the same keyed array, just select different variables
ds_keyed_flux = ds_keyed_reco  # reuse the same keyed array

# Define neural networks for FluxPartModel
# Select available predictors for FluxPartModel
predictors_FluxPart = []
if :SWC_shallow in available_vars
    push!(predictors_FluxPart, :SWC_shallow)
elseif :SWC in available_vars
    push!(predictors_FluxPart, :SWC)
end
if :P in available_vars
    push!(predictors_FluxPart, :P)
end
if :VPD in available_vars
    push!(predictors_FluxPart, :VPD)
end

# Ensure we have at least one predictor
if isempty(predictors_FluxPart)
    push!(predictors_FluxPart, :SWC_shallow)  # fallback
end

println("FluxPartModel predictors: ", predictors_FluxPart)

RUE_predictors = predictors_FluxPart  # predictors for RUE
Rb_predictors = predictors_FluxPart   # predictors for Rb
forcing = [:SW_IN, :TA]               # forcing variables

# Instantiate FluxPartModelQ10Lux model
FluxPart_model = FluxPartModelQ10Lux(RUE_predictors, Rb_predictors; forcing=forcing, Q10=1.5f0, neurons=15)

# Train FluxPartModel
out_FluxPart = train(FluxPart_model, ds_keyed_flux, (:Q10,); nepochs=200, batchsize=512, opt=Adam(0.01))

# Plot training results for FluxPartModel
fig_FluxPart = Figure(size=(1200, 800))

# NEE predictions
ax_nee = Makie.Axis(fig_FluxPart[1, 1], title="FluxPartModel - NEE Predictions")
lines!(ax_nee, out_FluxPart.train_obs_pred[!, :NEE_pred], color=:orangered, label="prediction")
lines!(ax_nee, out_FluxPart.train_obs_pred[!, :NEE], color=:dodgerblue, label="observation")
axislegend(ax_nee; position=:lt)

# GPP and Reco components
ax_components = Makie.Axis(fig_FluxPart[2, 1], title="FluxPartModel - GPP and Reco Components")
lines!(ax_components, out_FluxPart.train_obs_pred[!, :GPP], color=:green, label="GPP")
lines!(ax_components, out_FluxPart.train_obs_pred[!, :RECO], color=:red, label="Reco")
axislegend(ax_components; position=:lt)

Label(fig_FluxPart[0,1], "FluxPartModel: NEE, GPP, and Reco Predictions", tellwidth=false)
fig_FluxPart

# =============================================================================
# Compare Q10 values from both models
# =============================================================================

println("Q10 from RbQ10 model: ", out_RbQ10.ps.Q10[end])
println("Q10 from FluxPartModel: ", out_FluxPart.ps.Q10[end])

# Plot Q10 evolution during training
fig_Q10 = Figure(size=(800, 400))
ax_Q10 = Makie.Axis(fig_Q10[1, 1], title="Q10 Evolution During Training", xlabel="Epoch", ylabel="Q10")
lines!(ax_Q10, out_RbQ10.ps.Q10, color=:blue, label="RbQ10 Model")
lines!(ax_Q10, out_FluxPart.ps.Q10, color=:red, label="FluxPartModel")
axislegend(ax_Q10; position=:rt)
fig_Q10
