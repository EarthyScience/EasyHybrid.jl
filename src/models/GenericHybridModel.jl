export HybridModel, constructHybridModel, scale_single_param, AbstractHybridModel, build_hybrid, ParameterContainer, default, lower, upper

abstract type AbstractHybridModel end

mutable struct ParameterContainer{NT<:NamedTuple, T} <: AbstractHybridModel
    values::NT
    table::T

    function ParameterContainer(values::NT) where {NT<:NamedTuple}
        table = EasyHybrid.build_parameter_matrix(values)
        new{NT,typeof(table)}(values, table)
    end
end

function build_hybrid(paras::NamedTuple, f::DataType)
    ca = EasyHybrid.ParameterContainer(paras)
    return f(ca)
end


# ───────────────────────────────────────────────────────────────────────────
# Main Hybrid Model Structure
struct HybridModel
    NN    :: Chain
    predictors       :: Vector{Symbol}
    forcing          :: Vector{Symbol}
    targets          :: Vector{Symbol}
    mechanistic_model  :: Function
    parameters       :: AbstractHybridModel          # holds the full ComponentMatrix
    neural_param_names   :: Vector{Symbol}       # which parameters come from the NN
    global_param_names :: Vector{Symbol}       # which parameters are estimated globally
    fixed_param_names  :: Vector{Symbol}       # which parameters are fixed
end

# Constructor with clearer naming
function constructHybridModel(
    predictors,
    forcing,
    targets,
    mechanistic_model,
    parameters,
    neural_param_names,
    global_param_names
)
    all_names = pnames(parameters)
    @assert all(n in all_names for n in neural_param_names) "neural_param_names ⊆ param_names"
    # if empty predictors do not construct NN
    if length(predictors) > 0
        NN = Chain(Dense(length(predictors), 64, tanh), Dense(64, 128, tanh), Dense(128, length(neural_param_names), tanh))
    else
        NN = Chain()
    end
    fixed_param_names = [ n for n in all_names if !(n in [neural_param_names..., global_param_names...]) ]
    return HybridModel(NN, predictors, forcing, targets, mechanistic_model, parameters, neural_param_names, global_param_names, fixed_param_names)
end

# ───────────────────────────────────────────────────────────────────────────
# Initial parameters: scalars come from parameters.table[:, :default]
function LuxCore.initialparameters(rng::AbstractRNG, m::HybridModel)
    ps_nn, _ = LuxCore.setup(rng, m.NN)
    # start with the NN weights
    nt = (; ps = ps_nn)
    # then append each global parameter as a 1‐vector of Float32
    if !isempty(m.global_param_names)
        for g in m.global_param_names
            random_val = rand(rng, Float32)
            nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([random_val],)))
        end
    end
    return nt
end


function LuxCore.initialstates(rng::AbstractRNG, m::HybridModel)
    _, st_nn = LuxCore.setup(rng, m.NN)
    # start with the NN weights
    nt = (;)
    # then append each global parameter as a 1‐vector of Float32
    if !isempty(m.fixed_param_names)
        for f in m.fixed_param_names  
            default_val = default(m.parameters)[f]
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
    scale_single_param(name, raw_val, parameters)

Scale a single parameter using the sigmoid scaling function.
"""
function scale_single_param(name, raw_val, hm::AbstractHybridModel)
    ℓ = lower(hm)[name]
    u = upper(hm)[name]
    return ℓ .+ (u .- ℓ) .* sigmoid.(raw_val)
end


# ───────────────────────────────────────────────────────────────────────────
# Forward pass: exactly as before, but drop init_globals
function (m::HybridModel)(ds_k, ps, st)

    # 1) get features
    predictors = ds_k(m.predictors) 
    forcing_data = Array(ds_k(m.forcing))[1, :]

    parameters = m.parameters

    # 2) run NN → B×P

    # scale global parameters (handle empty case)
    if !isempty(m.global_param_names)
        global_vals = Tuple(
                scale_single_param(g, ps[g], parameters)
                for g in m.global_param_names
            )
        global_params = NamedTuple{Tuple(m.global_param_names), Tuple{typeof.(global_vals)...}}(global_vals)
    else
        global_params = NamedTuple()
    end

    # scale NN parameters (handle empty case)
    if !isempty(m.neural_param_names)
        nn_out, st_NN = LuxCore.apply(m.NN, predictors, ps.ps, st.st)
        nn_cols = eachrow(nn_out)
        nn_params   = NamedTuple(zip(m.neural_param_names, nn_cols))
        scaled_nn_vals = Tuple(
            scale_single_param(name, nn_params[name], parameters)
            for name in m.neural_param_names
        )
        scaled_nn_params   = NamedTuple(zip(m.neural_param_names, scaled_nn_vals))
    else
        scaled_nn_params = NamedTuple()
        st_NN = st.st
    end

    # pick fixed parameters (handle empty case)
    if !isempty(m.fixed_param_names)
        fixed_vals = Tuple(st.fixed[f] for f in m.fixed_param_names)
        fixed_params = NamedTuple{Tuple(m.fixed_param_names), Tuple{typeof.(fixed_vals)...}}(fixed_vals)
    else
        fixed_params = NamedTuple()
    end

    all_params = merge(scaled_nn_params, global_params, fixed_params)

    # 6) physics
    y_pred = m.mechanistic_model(forcing_data; all_params...)

    out = (;θ = y_pred, parameters = all_params)

    st_new = (; st = st_NN, fixed = st.fixed)

    return out, (; st = st_new)
end