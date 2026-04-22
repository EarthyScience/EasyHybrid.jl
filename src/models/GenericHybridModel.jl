export SingleNNHybridModel, MultiNNHybridModel, constructHybridModel, scale_single_param, AbstractHybridModel, build_hybrid, ParameterContainer, default, lower, upper, hard_sigmoid, inv_sigmoid
export HybridParams

# Import necessary components for neural networks
using Lux: BatchNorm
using Lux: sigmoid

# Define the hard sigmoid activation function
function hard_sigmoid(x)
    return clamp.(0.2 .* x .+ 0.5, 0.0, 1.0)
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
# Single NN Hybrid Model Structure (optimized for performance)
struct SingleNNHybridModel{NP, GP, FP, M<:Function} <: LuxCore.AbstractLuxContainerLayer{(:NN,)}
    NN::Chain
    predictors::Vector{Symbol}
    forcing::Vector{Symbol}
    targets::Vector{Symbol}
    mechanistic_model::M
    parameters::AbstractHybridModel
    neural_param_names::Val{NP}
    global_param_names::Val{GP}
    fixed_param_names::Val{FP}
    scale_nn_outputs::Bool
    start_from_default::Bool
    config::NamedTuple
end

function SingleNNHybridModel(
    NN, predictors, forcing, targets,
    mechanistic_model, parameters,
    neural_param_names, global_param_names, fixed_param_names,
    scale_nn_outputs, start_from_default, config
)
    return SingleNNHybridModel(
        NN, predictors, forcing, targets,
        mechanistic_model, parameters,
        Val(Tuple(neural_param_names)),
        Val(Tuple(global_param_names)),
        Val(Tuple(fixed_param_names)),
        scale_nn_outputs, start_from_default, config
    )
end

# Multi-NN Hybrid Model Structure (optimized for performance)
struct MultiNNHybridModel{NP, GP, FP, M<:Function, NNT<:NamedTuple} <: LuxCore.AbstractLuxContainerLayer{(:NNs,)}
    NNs::NNT
    predictors::NamedTuple
    forcing::Vector{Symbol}
    targets::Vector{Symbol}
    mechanistic_model::M
    parameters::AbstractHybridModel
    neural_param_names::Val{NP}
    global_param_names::Val{GP}
    fixed_param_names::Val{FP}
    scale_nn_outputs::Bool
    start_from_default::Bool
    config::NamedTuple
end

function MultiNNHybridModel(
    NNs, predictors, forcing, targets,
    mechanistic_model, parameters,
    neural_param_names, global_param_names, fixed_param_names,
    scale_nn_outputs, start_from_default, config
)
    return MultiNNHybridModel(
        NNs, predictors, forcing, targets,
        mechanistic_model, parameters,
        Val(Tuple(neural_param_names)),
        Val(Tuple(global_param_names)),
        Val(Tuple(fixed_param_names)),
        scale_nn_outputs, start_from_default, config
    )
end

# Unified constructor that dispatches based on predictors type
function constructHybridModel(
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

    return SingleNNHybridModel(NN, predictors, forcing, targets, mechanistic_model, parameters, neural_param_names, global_param_names, fixed_param_names, scale_nn_outputs, start_from_default, config)
end

function constructHybridModel(
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

    return MultiNNHybridModel(NNs, predictors, forcing, targets, mechanistic_model, parameters, neural_param_names, global_param_names, fixed_param_names, scale_nn_outputs, start_from_default, config)
end

function constructHybridModel(
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
        return constructHybridModel(
            predictors, forcing, targets, mechanistic_model, parameters,
            neural_param_names, global_param_names; kwargs...
        )
    elseif predictors isa NamedTuple
        return constructHybridModel(
            predictors, forcing, targets, mechanistic_model, parameters,
            global_param_names; kwargs...
        )
    else
        throw(ArgumentError("predictors must be Vector{Symbol} or NamedTuple, got $(typeof(predictors))"))
    end
end

# ───────────────────────────────────────────────────────────────────────────
# Initial parameters for SingleNNHybridModel
function LuxCore.initialparameters(rng::AbstractRNG, m::SingleNNHybridModel)
    ps_nn, _ = LuxCore.setup(rng, m.NN)
    nt = (; ps = ps_nn)

    global_names = _unwrap(m.global_param_names)
    if !isempty(global_names)
        vals = if m.start_from_default
            ntuple(length(global_names)) do i
                [Float32(scale_single_param_minmax(global_names[i], m.parameters))]
            end
        else
            ntuple(length(global_names)) do i
                [rand(rng, Float32)]
            end
        end
        nt = merge(nt, NamedTuple{global_names}(vals))
    end

    return nt
end

# Initial parameters for MultiNNHybridModel
function LuxCore.initialparameters(rng::AbstractRNG, m::MultiNNHybridModel)
    nn_params = map(nn -> first(LuxCore.setup(rng, nn)), m.NNs)
    nt = (; nn_params...)

    global_names = _unwrap(m.global_param_names)
    if !isempty(global_names)
        vals = if m.start_from_default
            ntuple(length(global_names)) do i
                [Float32(scale_single_param_minmax(global_names[i], m.parameters))]
            end
        else
            ntuple(length(global_names)) do i
                [rand(rng, Float32)]
            end
        end
        nt = merge(nt, NamedTuple{global_names}(vals))
    end
    return nt
end

# Initial states for SingleNNHybridModel
function LuxCore.initialstates(rng::AbstractRNG, m::SingleNNHybridModel)
    _, st_nn = LuxCore.setup(rng, m.NN)

    fixed_names = _unwrap(m.fixed_param_names)
    fixed = if !isempty(fixed_names)
        vals = ntuple(length(fixed_names)) do i
            [Float32(default(m.parameters)[fixed_names[i]])]
        end
        NamedTuple{fixed_names}(vals)
    else
        NamedTuple()
    end

    return (; st_nn = st_nn, fixed = fixed)
end

# Initial states for MultiNNHybridModel
function LuxCore.initialstates(rng::AbstractRNG, m::MultiNNHybridModel)
    nn_states = map(nn -> last(LuxCore.setup(rng, nn)), m.NNs)

    fixed_names = _unwrap(m.fixed_param_names)
    fixed = if !isempty(fixed_names)
        vals = ntuple(length(fixed_names)) do i
            [Float32(default(m.parameters)[fixed_names[i]])]
        end
        NamedTuple{fixed_names}(vals)
    else
        NamedTuple()
    end

    return merge(nn_states, (fixed = fixed,))
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
# Forward pass for SingleNNHybridModel (optimized, no branching)
function (m::SingleNNHybridModel)(ds_k, ps, st)
    predictors   = @inbounds ds_k[1]
    forcing_data = @inbounds ds_k[2]
    parameters   = m.parameters

    # 1) Global params
    global_params = if !isempty(_unwrap(m.global_param_names))
        _build_global_params(m, ps, parameters, m.global_param_names)
    else
        NamedTuple()
    end

    # 2) NN params
    nn_out, st_nn = LuxCore.apply(m.NN, predictors, ps.ps, st.st_nn)
    nn_out = nn_out::Matrix{eltype(predictors)}
    scaled_nn_params = if !isempty(_unwrap(m.neural_param_names))
        _build_nn_params(m, nn_out, parameters, m.neural_param_names)
    else
        NamedTuple()
    end

    # 3) Fixed params
    fixed_params = if !isempty(_unwrap(m.fixed_param_names))
        _build_fixed_params(m, st, m.fixed_param_names)
    else
        NamedTuple()
    end

    # 4) Merge & call mechanistic model
    all_params = merge(scaled_nn_params, global_params)
    all_params = merge(all_params, fixed_params)
    all_kwargs = merge(forcing_data, all_params)

    y_pred = m.mechanistic_model(; all_kwargs...)
    out    = merge(y_pred, (parameters = all_params,))

    st_new = (st_nn = st_nn, fixed = st.fixed)
    return out, st_new
end

_unwrap(::Val{names}) where {names} = names

function _build_global_params(m, ps, parameters, ::Val{names}) where {names}
    vals = ntuple(length(names)) do i
        scale_single_param(names[i], ps[names[i]], parameters)
    end
    return NamedTuple{names}(vals)
end

function _build_nn_params(m, nn_out, parameters, ::Val{names}) where {names}
    N = length(names)
    vals = if m.scale_nn_outputs
        ntuple(N) do i
            scale_single_param(names[i], view(nn_out, i, :), parameters)
        end
    else
        ntuple(i -> view(nn_out, i, :), N)
    end
    return NamedTuple{names}(vals)
end

function _build_fixed_params(m, st, ::Val{names}) where {names}
    vals = ntuple(length(names)) do i
        st.fixed[names[i]]
    end
    return NamedTuple{names}(vals)
end

function (m::SingleNNHybridModel)(df::DataFrame, ps, st)
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
    for k in keys(out)
        if length(out[k]) == size(x, 2)
            dfnew[!, String(k) * "_pred"] = out[k]
        end
    end
    return dfnew
end

# Forward pass for MultiNNHybridModel (optimized, no branching)
function (m::MultiNNHybridModel)(ds_k::Tuple, ps, st)
    parameters   = m.parameters
    forcing_data = ds_k[2]

    # Run all NNs in one type-stable map over the NamedTuple
    nn_results = map(keys(m.NNs)) do nn_name
        LuxCore.apply(m.NNs[nn_name], ds_k[1][nn_name], ps[nn_name], st[nn_name])
    end
    nn_outputs = map(first, nn_results)  # NamedTuple of raw outputs
    nn_states  = map(last,  nn_results)  # NamedTuple of states

    # Scale NN outputs
    scaled_nn_params = _build_multi_nn_params(m, nn_outputs, parameters, m.neural_param_names)

    # Global params
    global_params = if !isempty(_unwrap(m.global_param_names))
        _build_global_params(m, ps, parameters, m.global_param_names)
    else
        NamedTuple()
    end

    # Fixed params
    fixed_params = if !isempty(_unwrap(m.fixed_param_names))
        _build_fixed_params(m, st, m.fixed_param_names)
    else
        NamedTuple()
    end

    all_params = merge(scaled_nn_params, global_params, fixed_params)
    all_kwargs = merge(forcing_data, all_params)

    y_pred = m.mechanistic_model(; all_kwargs...)
    out    = merge(y_pred, (parameters = all_params, nn_outputs = nn_outputs))
    st_new = merge(nn_states, (fixed = st.fixed,))

    return out, st_new
end

function _build_multi_nn_params(m, nn_outputs, parameters, ::Val{names}) where {names}
    vals = if m.scale_nn_outputs
        ntuple(length(names)) do i
            scale_single_param(names[i], view(nn_outputs[names[i]], 1, :), parameters)
        end
    else
        ntuple(length(names)) do i
            view(nn_outputs[names[i]], 1, :)
        end
    end
    return NamedTuple{names}(vals)
end

function (m::MultiNNHybridModel)(df::DataFrame, ps, st)
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
    for k in keys(out)
        if length(out[k]) == size(x, 2)
            dfnew[!, String(k) * "_pred"] = out[k]
        end
    end
    return dfnew
end
