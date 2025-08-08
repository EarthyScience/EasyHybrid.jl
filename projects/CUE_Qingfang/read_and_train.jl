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
using EasyHybrid

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
data_file = joinpath(project_path, "data", "GlobalCUE_18O_July2025.xlsx")
df = DataFrame(XLSX.readtable(data_file, "Sheet2", infer_eltypes = true))

# Print column names and types
for col in names(df)
    println(col, ":      ", typeof(df[!, col]))
end

# -----------------------------------------------------------------------------
# Targets, Forcing, and Predictors Definition
# -----------------------------------------------------------------------------
targets   = [:CUE, :Growth, :Respiration]
forcing   = []
predictors = [
    :MAT, :pH, :Clay, :Sand, :Silt, :TN, :CN, :MAP, :PET, :NPP
]

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
o_def = CUE_simple(; Growth = df[!, :Growth], Respiration = df[!, :Respiration])

using WGLMakie

fig1 = Figure()
fig1

ax1 = WGLMakie.Axis(fig1[1, 1], xlabel = "Growth", ylabel = "CUE")
scatter!(ax1, df[!, :Growth], o_def.CUE, label = "with function")
scatter!(ax1, df[!, :Growth], df[!, :CUE], color = :red, label = "from data")
# TODO: why some mismatches?

ax2 = WGLMakie.Axis(fig1[2, 1], xlabel = "Respiration", ylabel = "CUE")
scatter!(ax2, df[!, :Respiration], o_def.CUE, label = "with function")
scatter!(ax2, df[!, :Respiration], df[!, :CUE], color = :red, label = "from data")
axislegend(ax1; position=:rt)

# -----------------------------------------------------------------------------
# Hybrid Model Construction and Training (Simple)
# -----------------------------------------------------------------------------
neural_param_names = [:Growth, :Respiration]
global_param_names = []

hybrid_model_simple = constructHybridModel(
    predictors,
    forcing,
    targets,
    CUE_simple,
    parameters,
    neural_param_names,
    global_param_names;
    scale_nn_outputs = true,
    hidden_layers    = [15, 15],
    activation       = sigmoid, # tanh, relu, swish
    input_batchnorm  = true
)

out_simple = train(
    hybrid_model_simple,
    df,
    ();
    nepochs        = 1000,
    batchsize      = 64,
    opt            = AdamW(0.1, (0.9, 0.999), 0.01),
    loss_types     = [:nse, :mse],
    training_loss  = :nse,
    yscale         = identity, # log
    agg            = mean,
    shuffleobs     = true,
    patience       = 50
)

EasyHybrid.poplot(out_simple)

# =============================================================================
# NOW WITH Q10
# =============================================================================

# -----------------------------------------------------------------------------
# Targets, Forcing, and Predictors Definition (Q10)
# -----------------------------------------------------------------------------
targets   = [:CUE, :Growth, :Respiration]
forcing   = [:MAT]
predictors = [
    :pH, :Clay, :Sand, :Silt, :TN, :CN, :MAP, :PET, :NPP
]

# -----------------------------------------------------------------------------
# Parameter Container for the Mechanistic Model (Q10)
# -----------------------------------------------------------------------------
parametersQ10 = (
    #    name           (default,   lower,    upper)         # description
    Growth_Tref         = (500.0f0,     1f-5,   7000.0f0),       # Growth
    Respiration_Tref    = (1200.0f0,    1f-5,   12000.0f0),      # Respiration
    Q10Growth      = (2.0f0,       1f0,   5.0f0),         # Q10Growth
    Q10Respiration = (2.0f0,       1f0,   5.0f0),         # Q10Respiration
)

function fQ10(T, T_ref, Q10)
    return Q10 .* 0.1 .* (T .- T_ref)
end

function CUE_Q10(; MAT, Growth_Tref, Respiration_Tref, Q10Growth, Q10Respiration)
    Growth      = Growth_Tref      .* fQ10(MAT, 15.f0, Q10Growth)
    Respiration = Respiration_Tref .* fQ10(MAT, 15.f0, Q10Respiration)
    CUE = Growth ./ (Respiration .+ Growth)
    return (; CUE, Growth, Respiration, Q10Growth, Q10Respiration, Growth_Tref, Respiration_Tref)
end

# -----------------------------------------------------------------------------
# Hybrid Model Construction and Training (Q10)
# -----------------------------------------------------------------------------
neural_param_names = [:Growth_Tref, :Respiration_Tref]
global_param_names = [:Q10Growth, :Q10Respiration]

hybrid_model_Q10 = constructHybridModel(
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
    input_batchnorm  = true,
    start_from_default = true
)

out_Q10 = train(
    hybrid_model_Q10,
    df,
    ();
    nepochs        = 1000,
    batchsize      = 64,
    opt            = AdamW(0.1, (0.9, 0.999), 0.01),
    loss_types     = [:nse, :mse],
    training_loss  = :nse,
    yscale         = identity,
    agg            = mean,
    shuffleobs     = true,
    monitor_names  = [:Q10Growth, :Q10Respiration],
    patience       = 50
)

EasyHybrid.poplot(out_Q10)




neural_param_names = [:Growth_Tref, :Rm_Tref]
global_param_names = [:a]

targets   = [:CUE, :Growth, :Respiration]
forcing   = [:MAT]
predictors = [
    :pH, :Clay, :Sand, :Silt, :TN, :CN, :MAP, :PET, :NPP
]

# -----------------------------------------------------------------------------
# Parameter Container for the Mechanistic Model (Q10)
# -----------------------------------------------------------------------------
parametersQ10_Rm = (
    #    name           (default,   lower,    upper)         # description
    Growth_Tref         = (500.0f0,     1f-5,   7000.0f0),       # Growth
    Rm_Tref    = (500.0f0,    1f-5,   12000.0f0),      # Respiration
    Q10      = (2.0f0,       1f0,   5.0f0),         # Q10
    a = (1.0f0, 0.01f0, 10.0f0) # scaling factor for Rg

)


function CUE_Q10_Rm(; MAT, Growth_Tref, Rm_Tref, a, Q10)
    Growth      = Growth_Tref      .* fQ10(MAT, 15.f0, Q10)
    Rg_Tref              = Growth .* a
    Rg = Rg_Tref .* fQ10(MAT, 15.f0, Q10)
    Rm = Rm_Tref .* fQ10(MAT, 15.f0, Q10)
    Respiration = Rg_Tref .+ Rm_Tref
    CUE = Growth ./ (Respiration .+ Growth)
    return (; CUE, Growth, Respiration, Rg_Tref, Q10, Rm_Tref, a, Rm, Rg)
end



hybrid_model_Q10_Rm = constructHybridModel(
    predictors,
    forcing,
    targets,
    CUE_Q10_Rm,
    parametersQ10_Rm,
    neural_param_names,
    global_param_names;
    scale_nn_outputs = true,
    hidden_layers    = [16, 8],
    activation       = sigmoid,
    input_batchnorm  = true,
    start_from_default = true
)

out_Q10 = train(
    hybrid_model_Q10_Rm,
    df,
    ();
    nepochs        = 1000,
    batchsize      = 64,
    opt            = AdamW(0.01, (0.9, 0.999), 0.01),
    loss_types     = [:nse, :mse],
    training_loss  = :nse,
    yscale         = identity,
    agg            = mean,
    shuffleobs     = true,
    monitor_names  = [:a, :Rm],
    patience       = 50
)
