export HybridModel15, constructHybridModel8, scale_single_param, AbstractHybridModel, build_hybrid, HybridModel6, default, lower, upper, HybridModel6

abstract type AbstractHybridModel end

mutable struct HybridModel6{NT<:NamedTuple, T} <: AbstractHybridModel
    values::NT
    table::T

    function HybridModel6(values::NT) where {NT<:NamedTuple}
        table = EasyHybrid.build_cm(values)
        new{NT,typeof(table)}(values, table)
    end
end

function build_hybrid(paras::NamedTuple, f::DataType)
    ca = EasyHybrid.HybridModel6(paras)
    return f(ca)
end


# ───────────────────────────────────────────────────────────────────────────
# 1) New HybridModel that holds FXWParams10 directly
struct HybridModel15
    NN           :: Chain
    predictors   :: Vector{Symbol}
    forcing      :: Vector{Symbol}
    targets      :: Vector{Symbol}
    mech_fun     :: Function
    params       :: AbstractHybridModel          # now hold the full ComponentMatrix
    nn_names     :: Vector{Symbol}       # which parameters come from the NN
    global_names :: Vector{Symbol}       # which parameters are estimated globally
    fixed_names  :: Vector{Symbol}       # which parameters are fixed
end

# 2) Constructor: dispatch on FXWParams10 instead of separate pieces
function constructHybridModel8(
    predictors,
    forcing,
    targets,
    mech_fun,
    params,
    nn_names,
    global_names
)
    all_names = pnames(params)
    @assert all(n in all_names for n in nn_names) "nn_names ⊆ param_names"
    # if empty predictors do not construct NN
    if length(predictors) > 0
        NN = Chain(Dense(length(predictors), 64, tanh), Dense(64, 128, tanh), Dense(128, length(nn_names), tanh))
    else
        NN = Chain()
    end
    fixed_names = [ n for n in all_names if !(n in [nn_names..., global_names...]) ]
    return HybridModel15(NN, predictors, forcing, targets, mech_fun, params, nn_names, global_names, fixed_names)
end

# ───────────────────────────────────────────────────────────────────────────
# 3) Initial parameters: scalars come from params.table[:, :default]
function LuxCore.initialparameters(rng::AbstractRNG, m::HybridModel15)
    ps_nn, _ = LuxCore.setup(rng, m.NN)
    # start with the NN weights
    nt = (; ps = ps_nn)
    # then append each global parameter as a 1‐vector of Float32
    if !isempty(m.global_names)
        for g in m.global_names
            random_val = rand(rng, Float32)
            nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([random_val],)))
        end
    end
    return nt
end


function LuxCore.initialstates(rng::AbstractRNG, m::HybridModel15)
    _, st_nn = LuxCore.setup(rng, m.NN)
    # start with the NN weights
    nt = (;)
    # then append each global parameter as a 1‐vector of Float32
    if !isempty(m.fixed_names)
        for f in m.fixed_names  
            default_val = default(m.params)[f]
            nt = merge(nt, NamedTuple{(f,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
        end
    end
    nt = (; st = st_nn, fixed = nt)
    return nt
end

function default(p::AbstractHybridModel)
    p.hybrid.table[:, :default]
end

function lower(p::AbstractHybridModel)
    p.hybrid.table[:, :lower]
end

function upper(p::AbstractHybridModel)
    p.hybrid.table[:, :upper]
end

pnames(p::AbstractHybridModel) = keys(p.hybrid.table.axes[1])

"""
    scale_single_param(name, raw_val, table)

Scale a single parameter using the sigmoid scaling function.
"""
function scale_single_param(name, raw_val, hm::AbstractHybridModel)
    ℓ = lower(hm)[name]
    u = upper(hm)[name]
    return ℓ .+ (u .- ℓ) .* sigmoid.(raw_val)
end


# ───────────────────────────────────────────────────────────────────────────
# 4) Forward pass: exactly as before, but drop init_globals
function (m::HybridModel15)(ds_k, ps, st)

    # 1) get features
    p     = ds_k(m.predictors) 
    x     = Array(ds_k(m.forcing))[1, :]

    pms = m.params

    # 2) run NN → B×P

    # scale global parameters (handle empty case)
    if !isempty(m.global_names)
        global_vals = Tuple(
                scale_single_param(g, ps[g], pms)
                for g in m.global_names
            )
        glob_ps = NamedTuple{Tuple(m.global_names), Tuple{typeof.(global_vals)...}}(global_vals)
    else
        glob_ps = NamedTuple()
    end

    # scale NN parameters (handle empty case)
    if !isempty(m.nn_names)
        nn_out, st_NN = LuxCore.apply(m.NN, p, ps.ps, st.st)
        nn_cols = eachrow(nn_out)
        nn_ps   = NamedTuple(zip(m.nn_names, nn_cols))
        scaled_nn_vals = Tuple(
            scale_single_param(name, nn_ps[name], pms)
            for name in m.nn_names
        )
        scaled_nn_ps   = NamedTuple(zip(m.nn_names, scaled_nn_vals))
    else
        scaled_nn_ps = NamedTuple()
        st_NN = st.st
    end

    # pick fixed parameters (handle empty case)
    if !isempty(m.fixed_names)
        fixed_vals = Tuple(st.fixed[f] for f in m.fixed_names)
        fixed_ps = NamedTuple{Tuple(m.fixed_names), Tuple{typeof.(fixed_vals)...}}(fixed_vals)
    else
        fixed_ps = NamedTuple()
    end

    a = merge(scaled_nn_ps, glob_ps, fixed_ps)

    #println(a)
    #println("\n")
    #println(typeof(values(a)))

    # 6) physics
    y_pred = m.mech_fun(x, a.θ_s, a.h_r, a.h_0, a.log_α, a.log_nm1, a.log_m)

    out = (;θ = y_pred, a = a)

    st_new = (; st = st_NN, fixed = st.fixed)

    return out, (; st = st_new)
end