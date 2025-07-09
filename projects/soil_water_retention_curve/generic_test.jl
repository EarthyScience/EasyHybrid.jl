using Pkg
Pkg.activate("projects/soil_water_retention_curve")
Pkg.develop(path=pwd())
Pkg.instantiate()

using Revise

using EasyHybrid
using GLMakie
using Statistics

# ? move the `csv` file into the `BulkDSOC/data` folder (create folder)
@__DIR__
df_o = CSV.read(joinpath(@__DIR__, "./data/Norouzi_et_al_2024_WRR_Final.csv"), DataFrame, normalizenames=true)

df = copy(df_o)
df.h = 10 .^ df.pF

# Rename :WC to :θ in the DataFrame
rename!(df, :WC => :θ)

ds_keyed = to_keyedArray(Float32.(df))

ds_keyed(:θ)
ds_keyed(:h)




# 1) Define your mechanistic wrapper that matches mech_fun(h, p1, p2, …)
mech_fun(h, θ_s, α, n, m, h_r, h_0) = mFXW_theta(h, θ_s, h_r, h_0, α, n, m)

# 2) Build a small NN that outputs 4 parameters: θ_s, α, n, m
NN = Chain(
  Dense( 5, 16, relu ),      # say you have 5 predictor features
  Dense(16,  4)              # output size = length(nn_names) = 4
)

# 3) Specify which keys in your dataset:
targets = :θ
forcing = :h
predictors = [:sand, :clay, :silt, :BD, :OC]   

# 4) List all mechanistic parameter names (order must match mech_fun signature)
param_names = [:θ_s, :α, :n, :m, :h_r, :h_0]

# 5) Tell the wrapper which of those come from the NN
nn_names = [:θ_s, :α, :n, :m]

init_globals = (h_r = 1f0, h_0 = 1f0)

# 6) Provide initials for the globals (h_r, h_0)
model = constructHybridModel(
  NN,
  predictors,
  forcing,
  targets,
  mech_fun,
  param_names,
  nn_names,
  init_globals
)


# 7) Initialize parameters & states
rng = Random.default_rng()
ps  = LuxCore.initialparameters(rng, model)
st  = LuxCore.initialstates(rng, model)

# 8) Forward‐pass on a batch `ds_k` (e.g. from DataLoader)
output, new_state = model(ds_keyed, ps, st)


using ModelParameters

# Define the parameter container with bounds
Base.@kwdef struct FXWParams
    θ_s::Param = Param(0.45, bounds = (0.0, 1.0))
    h_r::Param = Param(0.05, bounds = (0.0, 0.5))
    h_0::Param = Param(100.0, bounds = (0.0, 1000.0))
    α::Param   = Param(0.04, bounds = (0.001, 0.5))
    n::Param   = Param(1.56, bounds = (1.0, 5.0))
    m::Param   = Param(1.0 - 1/1.56, bounds = (0.0, 1.0))
end

function mFXW_theta2(h, p::FXWParams)
    S_e = Se_FXW(h, p.h_r.val, p.h_0.val, p.α.val, p.n.val, p.m.val)
    θ   = p.θ_s.val .* S_e
    return θ
end

mFXW_theta2(ds_keyed(:h), FXWParams())

mdl = Model(FXWParams())     # Flattened view for optimizers, plotting, etc.
vec = collect(mdl[:val])

using Parameters
@with_kw struct FXWParams2
    θ_s::Param = Param(0.45, bounds = (0.0, 1.0))
    h_r::Param = Param(0.05, bounds = (0.0, 0.5))
    h_0::Param = Param(100.0, bounds = (0.0, 1000.0))
    α::Param   = Param(0.04, bounds = (0.001, 0.5))
    n::Param   = Param(1.56, bounds = (1.0, 5.0))
    m::Param   = Param(1.0 - 1/1.56, bounds = (0.0, 1.0))
end

function mFXW_theta2(h, p::FXWParams2)
    @unpack_FXWParams2 p
    θ = θ_s.val .* Se_FXW(h, h_r.val, h_0.val, α.val, n.val, m.val)
    return θ
end

mFXW_theta2(ds_keyed(:h), FXWParams2())





using ComponentArrays

# 1) module‐level helper
function build_cm(values::NamedTuple) where T<:Real
    param_names     = collect(keys(values))
    bound_names = (:default, :lower, :upper)
    data = [ values[p][i] for p in param_names, i in 1:length(bound_names) ]
    row_ax = ComponentArrays.Axis(param_names)
    col_ax = ComponentArrays.Axis(bound_names)
    return ComponentArray(data, row_ax, col_ax)
end

# 2) the struct with inner constructor
mutable struct FXWParams10
    table::ComponentMatrix{Float64, Matrix{Float64}} 

    # 3) inner constructor lives inside the struct block
    function FXWParams10()
        values = (
 #"columns" are default, lower, upper
          θ_s = (0.45,    0.0,   1.0),
          h_r = (0.05,    0.0,   0.5),
          h_0 = (100.0,   0.0, 1000.0),
          α   = (0.04,    0.001, 0.05),
          n   = (1.56,    1.0,   5.0),
          m   = (0.36,    0.0,   1.0)
        )
        cm = build_cm(values)
        new(cm)             # wrap it up
    end
end

ca =FXWParams10()

ca2 = ca.table

ca2[:, :default]

using PrettyTables

pretty_table(
    ca2;
    header     = collect(keys(ca2.axes[2])),
    row_labels = collect(keys(ca2.axes[1])),
    alignment  = [:r, :r, :r]
)


function mFXW_theta3(h, p::FXWParams10)
    p = p.table[:, :default]
    θ = p.θ_s .* Se_FXW(h, p.h_r, p.h_0, p.α, p.n, p.m)
    return θ
end

mFXW_theta3(ds_keyed(:h), ca)

using ComponentArrays, PrettyTables

function build_cm(values::Dict{Symbol,Tuple{T,T,T}}) where T<:Real
    params     = collect(keys(values))
    bounds     = (:default, :lower, :upper)
    data       = [ values[p][i] for p in params, i in 1:length(bounds) ]
    row_ax     = ComponentArrays.Axis(params...)
    col_ax     = ComponentArrays.Axis(bounds...)
    return ComponentMatrix(data, row_ax, col_ax)  # correct 2D constructor
end

mutable struct FXWParams11
    table::ComponentMatrix{Float64}
    function FXWParams11()
        vals = Dict(
          :θ_s => (0.45, 0.0, 1.0),
          :h_r => (0.05, 0.0, 0.5),
          :h_0 => (100.0,0.0,1000.0),
          :α   => (0.04, 0.001,0.05),
          :n   => (1.56, 1.0, 5.0),
          :m   => (0.36, 0.0, 1.0)
        )
        new(build_cm(vals))
    end
end

# Usage
p   = FXWParams11()
ca2 = p.table

pretty_table(
    ca2;
    header     = collect(keys(ca2.axes[2])),
    row_labels = collect(keys(ca2.axes[1])),
    alignment  = [:l, :r, :r, :r]
)