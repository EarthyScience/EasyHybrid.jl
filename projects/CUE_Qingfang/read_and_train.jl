using Pkg

# -----------------------------------------------------------------------------
# Project Setup
# -----------------------------------------------------------------------------
project_path = "projects/CUE_Qingfang"
Pkg.activate(project_path)

# Only instantiate if Manifest.toml is missing
manifest_path = joinpath(project_path, "Manifest.toml")
if !isfile(manifest_path)
    Pkg.develop(path = pwd())
    Pkg.instantiate()
end

using DataFrames
using XLSX

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
data_file = joinpath(project_path, "data", "GlobalCUE_18O_July2025.xlsx")
xf = DataFrame(XLSX.readtable(data_file, "Sheet2", infer_eltypes = true))

# Print column names and types
for col in names(xf)
    println(col, ":      ", typeof(xf[!, col]))
end

# -----------------------------------------------------------------------------
# Targets, Forcing, and Predictors Definition
# -----------------------------------------------------------------------------
targets   = [:CUE, :Growth, :Respiration]
forcing   = []
predictors = [
    :MAT, :pH, :Clay, :Sand, :Silt, :TN, :CN, :MAP, :PET, :NPP, :CUE, :Growth, :Uptake
]

# -----------------------------------------------------------------------------
# Data Processing and Creation of KeyedArray
# -----------------------------------------------------------------------------
col_to_select = unique([predictors... , forcing... , targets...])

# Select columns and drop rows with any NaN values
sdf = copy(xf[!, col_to_select])
dropmissing!(sdf)

for col in names(sdf)
    T = eltype(sdf[!, col])
    if T <: Union{Missing, Real} || T <: Real
        sdf[!, col] = Float64.(coalesce.(sdf[!, col], NaN))
    end
end

using EasyHybrid
ds_keyed = to_keyedArray(Float32.(sdf))

# -----------------------------------------------------------------------------
# Parameter Container for the Mechanistic Model
# -----------------------------------------------------------------------------
parameters = (
    #   name     = (default,    lower,   upper)         # description
    Growth      = (500.0f0,      1f-5,   7000.0f0),       # Growth
    Respiration = (1200.0f0,     1f-5,   12000.0f0),      # Respiration
)

function CUE_simple(; Growth, Respiration)
    CUE = Growth ./ (Respiration .+ Growth)
    return (; CUE, Growth, Respiration)
end

# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
o_def = CUE_simple(; Growth = ds_keyed(:Growth), Respiration = ds_keyed(:Respiration))

using WGLMakie

fig1 = Figure()
fig1

ax1 = WGLMakie.Axis(fig1[1, 1], xlabel = "Growth", ylabel = "CUE")
scatter!(ax1, o_def.Growth, o_def.CUE)
scatter!(ax1, ds_keyed(:Growth), ds_keyed(:CUE), color = :red)
# TODO: why some mismatches?

ax2 = WGLMakie.Axis(fig1[2, 1], xlabel = "Respiration", ylabel = "CUE")
scatter!(ax2, o_def.Respiration, o_def.CUE)
scatter!(ax2, ds_keyed(:Respiration), ds_keyed(:CUE), color = :red)

# -----------------------------------------------------------------------------
# Hybrid Model Construction and Training (Simple)
# -----------------------------------------------------------------------------
neural_param_names = [:Growth, :Respiration]
global_param_names = []

hybrid_model = constructHybridModel(
    predictors,
    forcing,
    targets,
    CUE_simple,
    parameters,
    neural_param_names,
    global_param_names;
    scale_nn_outputs = true,
    hidden_layers    = [15, 15],
    activation       = sigmoid,
    input_batchnorm  = true
)

out = train(
    hybrid_model,
    ds_keyed,
    ();
    nepochs        = 100,
    batchsize      = 32,
    opt            = AdamW(0.01),
    loss_types     = [:mse, :nse],
    training_loss  = :nse,
    yscale         = identity,
    agg            = mean,
    shuffleobs     = true
)

# =============================================================================
# NOW WITH Q10
# =============================================================================

# -----------------------------------------------------------------------------
# Targets, Forcing, and Predictors Definition (Q10)
# -----------------------------------------------------------------------------
targets   = [:CUE, :Growth, :Respiration]
forcing   = [:MAT]
predictors = [
    :pH, :Clay, :Sand, :Silt, :TN, :CN, :MAP, :PET, :NPP, :CUE, :Growth, :Uptake
]

# -----------------------------------------------------------------------------
# Parameter Container for the Mechanistic Model (Q10)
# -----------------------------------------------------------------------------
parametersQ10 = (
    #    name           (default,   lower,    upper)         # description
    Growth         = (500.0f0,     1f-5,   7000.0f0),       # Growth
    Respiration    = (1200.0f0,    1f-5,   12000.0f0),      # Respiration
    Q10Growth      = (2.0f0,       1f0,   5.0f0),         # Q10Growth
    Q10Respiration = (2.0f0,       1f0,   5.0f0),         # Q10Respiration
)

function fQ10(T, T_ref, Q10)
    return Q10 .* 0.1 .* (T .- T_ref)
end

function CUE_Q10(; MAT, Growth, Respiration, Q10Growth, Q10Respiration)
    GrowthTemp      = Growth      .* fQ10(MAT, 15.f0, Q10Growth)
    RespirationTemp = Respiration .* fQ10(MAT, 15.f0, Q10Respiration)
    CUE = Growth ./ (Respiration .+ Growth)
    return (; CUE, Growth, Respiration, Q10Growth, Q10Respiration)
end

# -----------------------------------------------------------------------------
# Hybrid Model Construction and Training (Q10)
# -----------------------------------------------------------------------------
neural_param_names = [:Growth, :Respiration]
global_param_names = [:Q10Growth, :Q10Respiration]

hybrid_model = constructHybridModel(
    predictors,
    forcing,
    targets,
    CUE_Q10,
    parametersQ10,
    neural_param_names,
    global_param_names;
    scale_nn_outputs = true,
    hidden_layers    = [16, 8],
    activation       = sigmoid,
    input_batchnorm  = true
)

out = train(
    hybrid_model,
    ds_keyed,
    ();
    nepochs        = 100,
    batchsize      = 128,
    opt            = AdamW(0.01),
    loss_types     = [:mse, :nse],
    training_loss  = :nse,
    yscale         = identity,
    agg            = mean,
    shuffleobs     = true
)

θ_pred = out.val_obs_pred[!, Symbol(string(:CUE, "_pred"))]
θ_obs = out.val_obs_pred[!, :CUE]
EasyHybrid.poplot(θ_pred, θ_obs, "CUE")

θ_pred = out.val_obs_pred[!, Symbol(string(:Growth, "_pred"))]
θ_obs = out.val_obs_pred[!, :Growth]
EasyHybrid.poplot(θ_pred, θ_obs, "Growth")

θ_pred = out.val_obs_pred[!, Symbol(string(:Respiration, "_pred"))]
θ_obs = out.val_obs_pred[!, :Respiration]
EasyHybrid.poplot(θ_pred, θ_obs, "Respiration")

