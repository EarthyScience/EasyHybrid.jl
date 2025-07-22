# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
project_path = "projects/ExpoHybrid"
Pkg.activate(project_path)

manifest_path = joinpath(project_path, "Manifest.toml")
if !isfile(manifest_path)
    Pkg.develop(path=pwd())
    Pkg.instantiate()
end

# start using the package
using EasyHybrid
using EasyHybrid.AxisKeys
using EasyHybrid.DataFrameMacros
using GLMakie, AlgebraOfGraphics
using Chain: @chain as @c

struct ExpoHybParams <: AbstractHybridModel 
    hybrid::EasyHybrid.ParameterContainer
end

parameters = (
    #            default                  lower                     upper                description
    k      = ( 0.01f0,                  0.0f0,                   0.2f0 ),            # Exponent
    Resp0  = ( 2.0f0,                   0.0f0,                   8.0f0 ),          # Basal respiration [μmol/m²/s]
)

targets = [:Resp_obs]
forcings = [:T]
predictors = (Resp0=[:SM],)

parameter_container = build_parameters(parameters, ExpoHybParams)

### Create synthetic data: Resp = Resp0 * exp(k*T); Resp0 = f(SM)
##
begin
T = rand(500) .* 40 .- 10      # Random temperature
SM = rand(500) .* 0.8 .+ 0.1   # Random soil moisture
SM_fac = exp.(-8.0*(SM .- 0.6) .^ 2)
Resp0 = 1.1 .* SM_fac # Base respiration dependent on soil moisture
Resp = Resp0 .* exp.(0.07 .* T)
Resp_obs = Resp .+ randn(length(Resp)) .* 0.05 .* mean(Resp)  # Add some noise
end;
df = DataFrame(; T, SM, SM_fac, Resp0, Resp, Resp_obs)

@c data(df) * mapping(:T, :Resp, color=:SM) * visual(Scatter)  draw(figure=(;title="Soil respiration vs Temperature with Soil Moisture as color"))
@c data(df) * mapping(:SM, :SM_fac) * visual(Scatter, color=:blue)  draw(figure=(;title="Soil moisture factor vs Soil Moisture"))

#fig, ax, sc = scatter(df.T, df.Resp_obs, label="Observed Respiration", color=df.SM)
#Colorbar(fig[1,2],sc,  label="Soil Moisture")
#scatter!(df.T, df.Resp, label="Synthetic Respiration", color=:red)
# scatter(df.SM, df.SM_fac, label="Observed Respiration", color=:blue, markersize=3)

# Define global parameters (none for this model, Q10 is fixed)
global_param_names = [:k]


# =============================================================================
# Parameter container for the mechanistic model
# =============================================================================

# Parameter structure for FluxPartModel
struct ExpoHybParams <: AbstractHybridModel 
    hybrid::EasyHybrid.ParameterContainer
end
##
# =============================================================================
# Mechanistic Model Definition
# =============================================================================

function Expo_resp_model(;T, Resp0, k)
    # -------------------------------------------------------------------------
    # Arguments:
    #   T     : Air temperature
    #   Resp0     : Basal respiration
    #   k    : Temperature sensitivity 
    #
    # Returns:
    #   Resp     : Respiration
    #   Resp0     : Respiration at T=0
    # -------------------------------------------------------------------------

    # Calculate fluxes
    #k=0.07f0  # Fixed value for k
    Resp_obs = Resp0 .* exp.(k .* T)
    return (;Resp_obs, Resp0)
end    

hybrid_model = constructHybridModel(
    predictors,
    forcings,
    targets,
    Expo_resp_model,
    parameter_container,
    global_param_names,
    scale_nn_outputs=true,
    hidden_layers = [16, 16],
    activation = sigmoid,
    input_batchnorm = true
)

out =  train(hybrid_model, df, (); nepochs=300, batchsize=64, opt=AdamW(0.01, (0.9, 0.999), 0.01), loss_types=[:mse, :nse], training_loss=:nse, random_seed=123, yscale = identity)

EasyHybrid.poplot(out.val_obs_pred[!, :Resp_obs], out.val_obs_pred[!, :Resp_obs_pred], "Respiration Predictions vs Observations")

result=fit_df(df, Expo_resp_model; 
    glob_params=global_param_names, 
    parameters=parameters, 
    targets=targets, 
    forcings=forcings, 
    predictors=predictors,
    nepochs=300, batchsize=64, opt=AdamW(0.1), loss_types=[:mse, :r2], training_loss=:mse, random_seed=123)

result=fit_df(df, Expo_resp_model; 
    from_result=result,
    glob_params=global_param_names, 
    parameters=parameters, 
    targets=targets, 
    forcings=forcings, 
    predictors=predictors,
    nepochs=300, batchsize=64, opt=AdamW(0.1), loss_types=[:mse, :r2], training_loss=:mse, random_seed=123)


run_model(data; based_on=nothing) = based_on.hybrid_model(data, based_on.result.ps, based_on.result.st)[1]

preds = run_model(df .|> Float32 |> to_keyedArray; based_on=result)
preds = NamedTuple{Symbol.(string.(keys(preds)[1:end-2]) .* "_pred")}(Tuple(preds)[1:end-2])
insertcols!(df, pairs(preds)...)

p = data(df) * mapping(:Resp_obs_pred, :Resp_obs, color=:SM) * visual(Scatter) +
  mapping([0], [1]) * visual(ABLines, linestyle = :dash, color = :black) 
draw(p, figure=(;title="Respiration Predictions vs Observations", size=(800, 600)))

p = data(df) * (mapping(:SM, :Resp0) * visual(Scatter, color=:blue) +
 mapping(:SM, :Resp0_pred) * visual(Scatter, color=:red));
draw(p, figure=(;title="Respiration0 Predictions vs Soil moisture", size=(800, 600)))

# =============================================================================
# train hybrid FluxPartModel_Q10_Lux model on NEE to get Q10, GPP, and Reco
# =============================================================================

NNRb = Chain(BatchNorm(length(predictors.Rb), affine=false), Dense(length(predictors.Rb), 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1))
NNRUE = Chain(BatchNorm(length(predictors.Rb), affine=false), Dense(length(predictors.Rb), 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1))

Q10start = collect(scale_single_param("Q10", ps_st2[1].Q10, parameter_container))[1]
FluxPart = FluxPartModelQ10Lux(NNRUE, NNRb, predictors.RUE, predictors.Rb, forcing_FluxPartModel, target_FluxPartModel, Q10start)

ps_st2[1].Q10 .= Q10start

# Train FluxPartModel
out_FluxPart = train(FluxPart, ds_keyed_FluxPartModel, (:Q10,); nepochs=30, batchsize=512, opt=AdamW(0.01), loss_types=[:mse, :r2], training_loss=:mse, random_seed=123, ps_st=ps_st2);

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
