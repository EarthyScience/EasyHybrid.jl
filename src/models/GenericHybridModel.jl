export HybridModel, scale_single_param, AbstractHybridModel, build_hybrid, ParameterContainer, default, lower, upper, hard_sigmoid, inv_hard_sigmoid, inv_sigmoid
export HybridParams

# Import necessary components for neural networks
using Lux: BatchNorm
using Lux: sigmoid

# Define the hard sigmoid activation function
function hard_sigmoid(x)
    return clamp.(0.2 .* x .+ 0.5, 0.0, 1.0)
end

# Inverse of `hard_sigmoid` on the linear region (0, 1).
# Saturated inputs (y ≤ 0 or y ≥ 1) are extrapolated linearly since the
# clamp makes the forward map non-invertible there.
function inv_hard_sigmoid(y)
    return (y .- 0.5) ./ 0.2
end

abstract type AbstractHybridModel end

mutable struct ParameterContainer{NT <: NamedTuple, T} <: AbstractHybridModel
    values::NT
    table::T

    function ParameterContainer(values::NT) where {NT <: NamedTuple}
        table = build_parameter_matrix(values)
        return new{NT, typeof(table)}(values, table)
    end
end

"""
    HybridParams{M<:Function}

A little parametric stub for “the params of function `M`.”  
All of your function‐based models become `HybridParams{typeof(f)}`.
"""
struct HybridParams{M <: Function} <: AbstractHybridModel
    hybrid::ParameterContainer
end

# ───────────────────────────────────────────────────────────────────────────
# Unified Hybrid Model Structure (optimized for performance)
struct HybridModel{T, P} <: LuxCore.AbstractLuxContainerLayer{(:NNs,)}
    NNs::T
    predictors::P
    forcing::Vector{Symbol}
    targets::Vector{Symbol}
    mechanistic_model::Function
    parameters::AbstractHybridModel
    neural_param_names::Vector{Symbol}
    global_param_names::Vector{Symbol}
    fixed_param_names::Vector{Symbol}
    scale_nn_outputs::Bool
    start_from_default::Bool
    config::NamedTuple
end

# Unified constructor that dispatches based on predictors type
function HybridModel(
        predictors::Vector{Symbol},
        forcing,
        targets,
        mechanistic_model,
        parameters,
        neural_param_names,
        global_param_names;
        hidden_layers::Union{Vector{Int}, Chain} = [32, 32],
        activation = tanh,
        scale_nn_outputs = false,
        input_batchnorm = false,
        start_from_default = true,
        kwargs...
    )

    if !isa(parameters, AbstractHybridModel)
        parameters = build_parameters(parameters, mechanistic_model)
    end

    all_names = pnames(parameters)
    @assert all(n in all_names for n in neural_param_names) "neural_param_names ⊆ param_names"

    # if empty predictors do not construct NN
    if length(predictors) > 0 && length(neural_param_names) > 0

        in_dim = length(predictors)
        out_dim = length(neural_param_names)

        NN = prepare_hidden_chain(
            hidden_layers, in_dim, out_dim;
            activation = activation,
            input_batchnorm = input_batchnorm
        )
    else
        NN = Chain()
    end

    fixed_param_names = [ n for n in all_names if !(n in [neural_param_names..., global_param_names...]) ]

    # capture the configuration used for construction
    config = (;
        hidden_layers,
        activation,
        scale_nn_outputs,
        input_batchnorm,
        start_from_default,
        kwargs...,
    )

    return HybridModel(NN, predictors, forcing, targets, mechanistic_model, parameters, neural_param_names, global_param_names, fixed_param_names, scale_nn_outputs, start_from_default, config)
end

function HybridModel(
        predictors::NamedTuple,
        forcing,
        targets,
        mechanistic_model,
        parameters,
        global_param_names;
        hidden_layers::Union{Vector{Int}, Chain, NamedTuple} = [32, 32],
        activation::Union{Function, NamedTuple} = tanh,
        scale_nn_outputs = false,
        input_batchnorm = false,
        start_from_default = true,
        kwargs...
    )

    if !isa(parameters, AbstractHybridModel)
        parameters = build_parameters(parameters, mechanistic_model)
    end

    all_names = pnames(parameters)
    neural_param_names = collect(keys(predictors))
    # Create neural networks based on predictors
    NNs = NamedTuple()
    for (nn_name, preds) in pairs(predictors)
        # Create a simple NN for each predictor set
        in_dim = length(preds)
        out_dim = 1
        if hidden_layers isa NamedTuple
            if activation isa NamedTuple
                nn = prepare_hidden_chain(
                    hidden_layers[nn_name], in_dim, out_dim;
                    activation = activation[nn_name],
                    input_batchnorm = input_batchnorm
                )
            else
                nn = prepare_hidden_chain(
                    hidden_layers[nn_name], in_dim, out_dim;
                    activation = activation,
                    input_batchnorm = input_batchnorm
                )
            end
        else
            nn = prepare_hidden_chain(
                hidden_layers, in_dim, out_dim;
                activation = activation,
                input_batchnorm = input_batchnorm
            )
        end
        NNs = merge(NNs, NamedTuple{(nn_name,), Tuple{typeof(nn)}}((nn,)))
    end

    fixed_param_names = [ n for n in all_names if !(n in [neural_param_names..., global_param_names...]) ]

    # capture the configuration used for construction
    config = (;
        hidden_layers,
        activation,
        scale_nn_outputs,
        input_batchnorm,
        start_from_default,
        kwargs...,
    )

    return HybridModel(NNs, predictors, forcing, targets, mechanistic_model, parameters, neural_param_names, global_param_names, fixed_param_names, scale_nn_outputs, start_from_default, config)
end

function HybridModel(
        ; predictors,
        forcing,
        targets,
        mechanistic_model,
        parameters,
        neural_param_names = nothing,
        global_param_names,
        kwargs...
    )
    if predictors isa Vector{Symbol}
        @assert neural_param_names !== nothing "Provide neural_param_names for Vector predictors"
        return HybridModel(
            predictors, forcing, targets, mechanistic_model, parameters,
            neural_param_names, global_param_names; kwargs...
        )
    elseif predictors isa NamedTuple
        return HybridModel(
            predictors, forcing, targets, mechanistic_model, parameters,
            global_param_names; kwargs...
        )
    else
        throw(ArgumentError("predictors must be Vector{Symbol} or NamedTuple, got $(typeof(predictors))"))
    end
end

# ───────────────────────────────────────────────────────────────────────────
# Helper functions for NN initialization
function _init_nn_params(rng::AbstractRNG, m::HybridModel{<:Any, <:NamedTuple})
    return map(nn -> LuxCore.setup(rng, nn)[1], m.NNs)
end

function _init_nn_params(rng::AbstractRNG, m::HybridModel{<:Any, <:Vector})
    ps_nn, _ = LuxCore.setup(rng, m.NNs)
    return (; ps = ps_nn)
end

# Initial parameters for HybridModel
function LuxCore.initialparameters(rng::AbstractRNG, m::HybridModel)
    nt = _init_nn_params(rng, m)

    # Then append each global parameter as a 1-vector of Float32
    if !isempty(m.global_param_names)
        if m.start_from_default
            for g in m.global_param_names
                default_val = scale_single_param_minmax(g, m.parameters)
                nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
            end
        else
            for g in m.global_param_names
                random_val = rand(rng, Float32)
                nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([random_val],)))
            end
        end
    end

    return nt
end

function _init_nn_states(rng::AbstractRNG, m::HybridModel{<:Any, <:NamedTuple})
    return map(nn -> LuxCore.setup(rng, nn)[2], m.NNs)
end

function _init_nn_states(rng::AbstractRNG, m::HybridModel{<:Any, <:Vector})
    _, st_nn = LuxCore.setup(rng, m.NNs)
    return (; st_nn = st_nn)
end

# Initial states for HybridModel
function LuxCore.initialstates(rng::AbstractRNG, m::HybridModel)
    nn_states_nt = _init_nn_states(rng, m)
    nt = (;)

    # Then append each fixed parameter as a 1-vector of Float32
    if !isempty(m.fixed_param_names)
        for f in m.fixed_param_names
            default_val = default(m.parameters)[f]
            nt = merge(nt, NamedTuple{(f,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
        end
    end

    return merge(nn_states_nt, (; fixed = nt))
end

function default(p::AbstractHybridModel)
    return p.hybrid.table[:, :default]
end

function lower(p::AbstractHybridModel)
    return p.hybrid.table[:, :lower]
end

function upper(p::AbstractHybridModel)
    return p.hybrid.table[:, :upper]
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

inv_sigmoid(y) = log.(y ./ (1 .- y))

""" 
    scale_single_param_minmax(name, hm::AbstractHybridModel)

Scale a single parameter using the minmax scaling function.
"""
function scale_single_param_minmax(name, hm::AbstractHybridModel)
    ℓ = lower(hm)[name]
    u = upper(hm)[name]
    return inv_sigmoid.((default(hm)[name] .- ℓ) ./ (u .- ℓ))
end


# ───────────────────────────────────────────────────────────────────────────
# ───────────────────────────────────────────────────────────────────────────
function _run_nn(m::HybridModel{<:Any, <:NamedTuple}, ds_k::Tuple, ps, st)
    applied = map(LuxCore.apply, m.NNs, ds_k[1], ps, st)
    nn_outputs = map(first, applied)
    nn_states = map(last, applied)

    scaled_vals = Tuple(
        begin
                val = eachslice(nn_outputs[nn_name]; dims = 1)[1]
                m.scale_nn_outputs ? scale_single_param(param_name, val, m.parameters) : val
            end
            for (nn_name, param_name) in zip(keys(m.NNs), m.neural_param_names)
    )
    scaled_nn_params = NamedTuple{Tuple(m.neural_param_names)}(scaled_vals)

    return scaled_nn_params, nn_states, (; nn_outputs = nn_outputs)
end

function _run_nn(m::HybridModel{<:Any, <:Vector}, ds_k::Tuple, ps, st)
    if !isempty(m.neural_param_names)
        nn_out, st_nn = LuxCore.apply(m.NNs, ds_k[1], ps.ps, st.st_nn)
        nn_cols = eachslice(nn_out, dims = 1)
        nn_params = NamedTuple(zip(m.neural_param_names, nn_cols))

        if m.scale_nn_outputs
            scaled_nn_vals = Tuple(
                scale_single_param(name, nn_params[name], m.parameters)
                    for name in m.neural_param_names
            )
        else
            scaled_nn_vals = Tuple(nn_params[name] for name in m.neural_param_names)
        end
        scaled_nn_params = NamedTuple(zip(m.neural_param_names, scaled_nn_vals))
    else
        scaled_nn_params = NamedTuple()
        st_nn = st.st_nn
    end
    return scaled_nn_params, (; st_nn = st_nn), (;)
end

# Forward pass for HybridModel (optimized, using multiple dispatch for NN run)
function (m::HybridModel)(ds_k::Tuple, ps, st)
    parameters = m.parameters

    # 1) Scale global parameters (handle empty case)
    if !isempty(m.global_param_names)
        global_vals = Tuple(
            scale_single_param(g, ps[g], parameters)
                for g in m.global_param_names
        )
        global_params = NamedTuple{Tuple(m.global_param_names), Tuple{typeof.(global_vals)...}}(global_vals)
    else
        global_params = NamedTuple()
    end

    # 2) Run neural network(s)
    scaled_nn_params, st_new_nns, out_extra = _run_nn(m, ds_k, ps, st)

    # 3) Pick fixed parameters (handle empty case)
    if !isempty(m.fixed_param_names)
        fixed_vals = Tuple(st.fixed[f] for f in m.fixed_param_names)
        fixed_params = NamedTuple{Tuple(m.fixed_param_names), Tuple{typeof.(fixed_vals)...}}(fixed_vals)
    else
        fixed_params = NamedTuple()
    end

    # 4) merge all parameters
    all_params = merge(scaled_nn_params, global_params, fixed_params)

    # 5) unpack forcing data
    forcing_data = ds_k[2]
    all_kwargs = merge(forcing_data, all_params)

    # 6) Apply mechanistic model
    y_pred = m.mechanistic_model(; all_kwargs...)

    out = (; y_pred..., parameters = all_params, out_extra...)
    st_new = (; st_new_nns..., fixed = st.fixed)

    return out, st_new
end

function (m::HybridModel)(ds_k, ps, st)
    # Forward pass fallback when ds_k is not explicitly typed as Tuple
    return m(Tuple(ds_k), ps, st)
end

function (m::HybridModel)(df::DataFrame, ps, st)
    @warn "Only makes sense in test mode, not training!"

    # Process numeric or missing-containing columns
    for col in names(df)
        what_type = eltype(df[!, col])
        if what_type <: Union{Missing, Real} || what_type <: Real
            df[!, col] = Float32.(coalesce.(df[!, col], NaN))
        end
    end

    all_data = to_keyedArray(df)
    x, _ = prepare_data(m, all_data)
    out, _ = m(x, ps, LuxCore.testmode(st))
    dfnew = copy(df)
    n_samples = x[1] isa NamedTuple ? size(first(values(x[1])), 2) : size(x[1], 2)
    for k in keys(out)
        if length(out[k]) == n_samples
            dfnew[!, String(k) * "_pred"] = out[k]
        end
    end
    return dfnew
end
