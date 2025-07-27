# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
project_path = "projects/ExpoHybrid"
Pkg.activate(project_path)

manifest_path = joinpath(project_path, "Manifest.toml")
if !isfile(manifest_path)
    package_path = pwd() 
    if !endswith(package_path, "EasyHybrid")
        @error "You opened in the wrong directory. Please open the EasyHybrid folder, create a new project in the projects folder and provide the relative path to the project folder as project_path."
    end
    Pkg.develop(path=package_path)
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
    return (;Resp_obs, Resp0, k)
end    

hybrid_model = constructHybridModel(
    predictors,
    forcings,
    targets,
    Expo_resp_model,
    parameter_container,
    global_param_names,
    scale_nn_outputs=false, # TODO true also works with good lower and upper bounds
    hidden_layers = [16, 16],
    activation = sigmoid,
    input_batchnorm = true
)

out =  train(hybrid_model, df, (:k,); nepochs=300, batchsize=64, opt=AdamW(0.01, (0.9, 0.999), 0.01), loss_types=[:mse, :nse], training_loss=:nse, random_seed=123, yscale = identity)

EasyHybrid.poplot(out)
EasyHybrid.plot_loss(out)
EasyHybrid.plot_parameters(out)
EasyHybrid.plot_training_summary(out, yscale=identity) # TODO needs work  
# TODO plot parameters on scale of model

preds = hybrid_model(df .|> Float32 |> to_keyedArray, out.ps, out.st)[1]
preds = NamedTuple{Symbol.(string.(keys(preds)[1:end-2]) .* "_pred")}(Tuple(preds)[1:end-2])
insertcols!(df, pairs(preds)...)

p = data(df) * mapping(:Resp_obs_pred, :Resp_obs, color=:SM) * visual(Scatter) +
  mapping([0], [1]) * visual(ABLines, linestyle = :dash, color = :black) 
draw(p, figure=(;title="Respiration Predictions vs Observations", size=(800, 600)))

p = data(df) * (mapping(:SM, :Resp0) * visual(Scatter, color=:blue) +
 mapping(:SM, :Resp0_pred) * visual(Scatter, color=:red));
draw(p, figure=(;title="Respiration0 Predictions vs Soil moisture", size=(800, 600)))
