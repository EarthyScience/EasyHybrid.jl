using Pkg
project_path = "projects/CUE_Qingfang"
Pkg.activate(project_path)
# Only instantiate if Manifest.toml is missing
manifest_path = joinpath(project_path, "Manifest.toml")
if !isfile(manifest_path)
    Pkg.develop(path=pwd())
    Pkg.instantiate()
end


using DataFrames
using XLSX

xf = DataFrame(XLSX.readtable(joinpath(@__DIR__, "data", "GlobalCUE_18O_July2025.xlsx"), "Sheet2", infer_eltypes=true))


for i in names(xf) 
    println(i, ":      ", typeof(xf[!,i]))
end

# =============================================================================
# Targets, Forcing and Predictors definition
# =============================================================================

# Select target and forcing variables and predictors
targets = [:CUE, :Growth, :Respiration]
forcing = []

# Define predictors as NamedTuple - this automatically determines neural parameter names
predictors = [:MAT, :pH, :Clay, :Sand, :Silt, :TN, :CN, :MAP, :PET, :NPP, :CUE, :Growth, :Uptake]

# =============================================================================
# More Data Processing and Creation of KeyedArray
# =============================================================================

col_to_select = unique([predictors..., forcing..., targets...])

# select columns and drop rows with any NaN values
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

# =============================================================================
# Parameter container for the mechanistic model
# =============================================================================

# Parameter structure for FluxPartModel
struct CUESimpleParams <: AbstractHybridModel 
    hybrid::EasyHybrid.ParameterContainer
end

# Define parameter structure with bounds
parameters = (
    #            default                  lower                     upper                description
    Growth   = ( 500.f0,                  1f-5,                   7000.f0 ),            # Growth
    Respiration   = ( 1200.0f0,                  1f-5,                   12000.f0 ),            # Respiration
)

parameter_container = build_parameters(parameters, CUESimpleParams)

function CUE_simple(; Growth, Respiration)

    CUE = Growth./(Respiration .+  Growth)
    
    return (;CUE, Growth, Respiration)
end



o_def = CUE_simple(; Growth=ds_keyed(:Growth), Respiration=ds_keyed(:Respiration))

using WGLMakie
#WGLMakie.activate!(inline=false)
fig1 = Figure()
fig1
ax = WGLMakie.Axis(fig1[1, 1], xlabel="Growth", ylabel="CUE")
scatter!(ax, o_def.Growth, o_def.CUE)
scatter!(ax, ds_keyed(:Growth), ds_keyed(:CUE), color=:red)
# TODO why some mismatches?

ax = WGLMakie.Axis(fig1[2, 1], xlabel="Respiration", ylabel="CUE")
scatter!(ax, o_def.Respiration, o_def.CUE)
scatter!(ax, ds_keyed(:Respiration), ds_keyed(:CUE), color=:red)


neural_param_names = [:Growth, :Respiration]
global_param_names = []

# Create the hybrid model using the unified constructor
hybrid_model = constructHybridModel(
    predictors,
    forcing,
    targets,
    CUE_simple,
    parameter_container,
    neural_param_names,
    global_param_names,
    scale_nn_outputs=true,
    hidden_layers = [15, 15],
    activation = sigmoid,
    input_batchnorm = true,
)


# Train
out = train(hybrid_model, ds_keyed, (); nepochs=100, batchsize=32, opt=AdamW(0.01), loss_types=[:mse, :nse], training_loss=:nse);
