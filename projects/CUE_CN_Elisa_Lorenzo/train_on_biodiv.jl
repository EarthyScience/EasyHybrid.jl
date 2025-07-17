using Pkg

# -----------------------------------------------------------------------------
# Project Setup
# -----------------------------------------------------------------------------
project_path = "projects/CUE_CN_Elisa_Lorenzo"
Pkg.activate(project_path)

# Only instantiate if Manifest.toml is missing
manifest_path = joinpath(project_path, "Manifest.toml")
if !isfile(manifest_path)
    Pkg.develop(path = pwd())
    Pkg.instantiate()
end

using DataFrames
using CSV

# ? move the `csv` file into the `BulkDSOC/data` folder (create folder)
df = CSV.read(joinpath(@__DIR__, "data", "data_PCA_reduced_microbial.csv"), DataFrame, normalizenames=true)

df_unstacked = unstack(df, :dim_2, :dim_1, :value)


# Print column names and types
for col in names(df_unstacked)
    println(col, ":      ", typeof(df_unstacked[!, col]))
end

# -----------------------------------------------------------------------------
# Targets, Forcing, and Predictors Definition
# -----------------------------------------------------------------------------

rename!(df_unstacked, :Avg_Rh_m => :Resp, :Mineralizacion_mg_m => :MinN)
rename!(df_unstacked, :C_Org_kg_m => :CS, :Microbial_Biomass_m => :CB, :C_N => :CN_CS)

targets   = [:Resp, :MinN]
predictors = filter(col -> startswith(string(col), "PC"), Symbol.(names(df_unstacked)))
forcing   = [:CS, :CB, :CN_CS]

# -----------------------------------------------------------------------------
# Data Processing and Creation of KeyedArray
# -----------------------------------------------------------------------------
col_to_select = unique([predictors... , forcing... , targets...])

# Select columns and drop rows with any NaN values
sdf = copy(df_unstacked[:, col_to_select])
dropmissing!(sdf)

for col in names(sdf)
    T = eltype(sdf[!, col])
    if T <: Union{Missing, Real} || T <: Real
        sdf[!, col] = Float64.(coalesce.(sdf[!, col], NaN))
    end
end


mean_Resp = Float32(mean(sdf[!, :Resp]))
mean_MinN = Float32(mean(sdf[!, :MinN]))

sdf[!, :Resp] .= sdf[!, :Resp] #./ mean_Resp
sdf[!, :MinN] .= sdf[!, :MinN] #./ mean_MinN

mean(sdf[!, :Resp])
mean(sdf[!, :MinN])

using EasyHybrid
ds_keyed = to_keyedArray(Float32.(sdf))

struct CUECNParams <: AbstractHybridModel
    hybrid::EasyHybrid.ParameterContainer
end

parameters = (
    #   name     = (default,    lower,   upper)         # description
    k0 = (0.1f-2,      1f-9,   1f-4),       
    gamma0 = (0.5f-2,     1f-3,   0.1f0),  
    CUE = (0.3f0,     1f-3,   1f0),  
    CNmic = (6f0,     1f0,   20f0),  
    scale = (1f0,     1f-3,   1f3),  
)

parameter_container = build_parameters(parameters, CUECNParams)

function CUE_CN(; CS, CB, CN_CS, k0, gamma0, CUE, CNmic, scale)

    # Calculate respiration
    Resp = (1.0f0 .- CUE) .* k0 .* CS .* CB .^ gamma0 .* scale
    # Calculate mineralization
    MinN = k0 .* CS .* CB .^ gamma0 .* (1.0f0 ./ CN_CS .- CUE ./ CNmic)  
    

    # Return results
    return (; Resp, MinN, CUE, CNmic, k0, gamma0, scale)

end   

# -----------------------------------------------------------------------------
# Neural parameters
# -----------------------------------------------------------------------------
neural_param_names = [:CUE, :CNmic]
global_param_names = [:scale, :k0, :gamma0]
targets = [:Resp, :MinN]

hybrid_model = constructHybridModel(
    predictors,
    forcing,
    targets,
    CUE_CN,
    parameter_container,
    neural_param_names,
    global_param_names,
    scale_nn_outputs = true,
    hidden_layers = [256, 128, 64, 32],
    activation = tanh,
    input_batchnorm = true
)

out = train(
    hybrid_model,
    ds_keyed,
    ();
    nepochs        = 1000,
    batchsize      = 32,
    opt            = AdamW(0.01),
    loss_types     = [:mse, :nse],
    training_loss  = :nse,
    yscale         = identity,
    agg            = mean,
    shuffleobs = true
)

out.train_diffs

θ_pred = out.train_obs_pred[!, Symbol(string(:Resp, "_pred"))]
θ_obs = out.train_obs_pred[!, :Resp]

fig = poplot(θ_pred, θ_obs, "Resp")

θ_pred = out.train_obs_pred[!, Symbol(string(:MinN, "_pred"))]
θ_obs = out.train_obs_pred[!, :MinN]

poplot!(fig, θ_pred, θ_obs, "MinN", 1, 2)
