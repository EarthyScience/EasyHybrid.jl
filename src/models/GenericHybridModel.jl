export SingleNNHybridModel, MultiNNHybridModel, constructHybridModel, scale_single_param, AbstractHybridModel, build_hybrid, ParameterContainer, default, lower, upper, hard_sigmoid
export HybridParams

# Import necessary components for neural networks
using Lux: BatchNorm
using Lux: sigmoid

# Define the hard sigmoid activation function
function hard_sigmoid(x)
    return clamp.(0.2 .* x .+ 0.5, 0.0, 1.0)
end

abstract type AbstractHybridModel end

mutable struct ParameterContainer{NT<:NamedTuple, T} <: AbstractHybridModel
    values::NT
    table::T

    function ParameterContainer(values::NT) where {NT<:NamedTuple}
        table = EasyHybrid.build_parameter_matrix(values)
        new{NT,typeof(table)}(values, table)
    end
end

"""
    HybridParams{M<:Function}

A little parametric stub for “the params of function `M`.”  
All of your function‐based models become `HybridParams{typeof(f)}`.
"""
struct HybridParams{M<:Function} <: AbstractHybridModel
    hybrid::ParameterContainer
end

# ───────────────────────────────────────────────────────────────────────────
# Single NN Hybrid Model Structure (optimized for performance)
struct SingleNNHybridModel
    NN              :: Chain
    predictors      :: Vector{Symbol}
    forcing         :: Vector{Symbol}
    targets         :: Vector{Symbol}
    mechanistic_model :: Function
    parameters      :: AbstractHybridModel
    neural_param_names :: Vector{Symbol}
    global_param_names :: Vector{Symbol}
    fixed_param_names  :: Vector{Symbol}
    scale_nn_outputs  :: Bool
    start_from_default :: Bool
end

# Multi-NN Hybrid Model Structure (optimized for performance)
struct MultiNNHybridModel
    NNs             :: NamedTuple
    predictors      :: NamedTuple
    forcing         :: Vector{Symbol}
    targets         :: Vector{Symbol}
    mechanistic_model :: Function
    parameters      :: AbstractHybridModel
    neural_param_names :: Vector{Symbol}
    global_param_names :: Vector{Symbol}
    fixed_param_names  :: Vector{Symbol}
    scale_nn_outputs  :: Bool
    start_from_default :: Bool
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
    if length(predictors) > 0

        in_dim  = length(predictors)
        out_dim = length(neural_param_names)
    
        NN = prepare_hidden_chain(hidden_layers, in_dim, out_dim;
                                  activation = activation,
                                  input_batchnorm = input_batchnorm)
    else
        NN = Chain()
    end
    
    fixed_param_names = [ n for n in all_names if !(n in [neural_param_names..., global_param_names...]) ]
    
    return SingleNNHybridModel(NN, predictors, forcing, targets, mechanistic_model, parameters, neural_param_names, global_param_names, fixed_param_names, scale_nn_outputs, start_from_default)
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
        in_dim  = length(preds)
        out_dim = 1
        if hidden_layers isa NamedTuple
            if activation isa NamedTuple
                nn = prepare_hidden_chain(hidden_layers[nn_name], in_dim, out_dim;
                                          activation = activation[nn_name],
                                          input_batchnorm = input_batchnorm)
            else
                nn = prepare_hidden_chain(hidden_layers[nn_name], in_dim, out_dim;
                                          activation = activation,
                                          input_batchnorm = input_batchnorm)
            end
        else
            nn = prepare_hidden_chain(hidden_layers, in_dim, out_dim;
                                      activation = activation,
                                      input_batchnorm = input_batchnorm)
        end
        NNs = merge(NNs, NamedTuple{(nn_name,), Tuple{typeof(nn)}}((nn,)))
    end
    
    fixed_param_names = [ n for n in all_names if !(n in [neural_param_names..., global_param_names...]) ]
    
    return MultiNNHybridModel(NNs, predictors, forcing, targets, mechanistic_model, parameters, neural_param_names, global_param_names, fixed_param_names, scale_nn_outputs, start_from_default)
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

# Initial parameters for MultiNNHybridModel
function LuxCore.initialparameters(rng::AbstractRNG, m::MultiNNHybridModel)
    # Setup parameters for each neural network
    nn_params = NamedTuple()
    for (nn_name, nn) in pairs(m.NNs)
        ps_nn, _ = LuxCore.setup(rng, nn)
        nn_params = merge(nn_params, NamedTuple{(nn_name,), Tuple{typeof(ps_nn)}}((ps_nn,)))
    end
    
    # Start with the NN weights
    nt = (; nn_params...)
    
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

# Initial states for SingleNNHybridModel
function LuxCore.initialstates(rng::AbstractRNG, m::SingleNNHybridModel)
    _, st_nn = LuxCore.setup(rng, m.NN)
    nt = (;)
    
    # Then append each fixed parameter as a 1-vector of Float32
    if !isempty(m.fixed_param_names)
        for f in m.fixed_param_names  
            default_val = default(m.parameters)[f]
            nt = merge(nt, NamedTuple{(f,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
        end
    end
    
    nt = (; st = st_nn, fixed = nt)
    return nt
end

# Initial states for MultiNNHybridModel
function LuxCore.initialstates(rng::AbstractRNG, m::MultiNNHybridModel)
    # Setup states for each neural network
    nn_states = NamedTuple()
    for (nn_name, nn) in pairs(m.NNs)
        _, st_nn = LuxCore.setup(rng, nn)
        nn_states = merge(nn_states, NamedTuple{(nn_name,), Tuple{typeof(st_nn)}}((st_nn,)))
    end
    
    # Start with the NN states
    nt = (;)
    
    # Then append each fixed parameter as a 1-vector of Float32
    if !isempty(m.fixed_param_names)
        for f in m.fixed_param_names  
            default_val = default(m.parameters)[f]
            nt = merge(nt, NamedTuple{(f,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
        end
    end

    nt = (; nn_states..., fixed = nt)
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
function (m::SingleNNHybridModel)(ds_k::KeyedArray, ps, st)
    # 1) get features
    predictors = ds_k(m.predictors) 

    parameters = m.parameters

    # 2) scale global parameters (handle empty case)
    if !isempty(m.global_param_names)
        global_vals = Tuple(
                scale_single_param(g, ps[g], parameters)
                for g in m.global_param_names
            )
        global_params = NamedTuple{Tuple(m.global_param_names), Tuple{typeof.(global_vals)...}}(global_vals)
    else
        global_params = NamedTuple()
    end

    # 3) scale NN parameters (handle empty case)
    if !isempty(m.neural_param_names)
        nn_out, st_NN = LuxCore.apply(m.NN, predictors, ps.ps, st.st)
        nn_cols = eachrow(nn_out)
        nn_params   = NamedTuple(zip(m.neural_param_names, nn_cols))
        
        # Use appropriate scaling based on setting
        if m.scale_nn_outputs
            scaled_nn_vals = Tuple(
                scale_single_param(name, nn_params[name], parameters)
                for name in m.neural_param_names
            )
        else
            scaled_nn_vals = Tuple(nn_params[name] for name in m.neural_param_names)
        end
        scaled_nn_params   = NamedTuple(zip(m.neural_param_names, scaled_nn_vals))
    else
        scaled_nn_params = NamedTuple()
        st_NN = st.st
    end

    # 4) pick fixed parameters (handle empty case)
    if !isempty(m.fixed_param_names)
        fixed_vals = Tuple(st.fixed[f] for f in m.fixed_param_names)
        fixed_params = NamedTuple{Tuple(m.fixed_param_names), Tuple{typeof.(fixed_vals)...}}(fixed_vals)
    else
        fixed_params = NamedTuple()
    end

    # 5) unpack forcing data
    forcing_data = unpack_keyedarray(ds_k, m.forcing)

    # 6) merge all parameters
    all_params = merge(scaled_nn_params, global_params, fixed_params)
    all_kwargs = merge(forcing_data, all_params)

    # 7) physics
    y_pred = m.mechanistic_model(; all_kwargs...)

    out = (;y_pred..., parameters = all_params)
    st_new = (; st = st_NN, fixed = st.fixed)

    return out, (; st = st_new)
end

function (m::SingleNNHybridModel)(df::DataFrame, ps, st)
    @warn "Only makes sense in test mode, not training!"

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
function (m::MultiNNHybridModel)(ds_k::KeyedArray, ps, st)

    parameters = m.parameters

    # 2) Scale global parameters (handle empty case)
    if !isempty(m.global_param_names)
        global_vals = Tuple(
                scale_single_param(g, ps[g], parameters)
                for g in m.global_param_names
            )
        global_params = NamedTuple{Tuple(m.global_param_names), Tuple{typeof.(global_vals)...}}(global_vals)
    else
        global_params = NamedTuple()
    end

    # 3) Run each neural network and collect outputs
    nn_outputs = NamedTuple()
    nn_states = NamedTuple()
    
    for (nn_name, nn) in pairs(m.NNs)
        predictors = m.predictors[nn_name]
        nn_out, st_nn = LuxCore.apply(nn, ds_k(predictors), ps[nn_name], st[nn_name])
        nn_outputs = merge(nn_outputs, NamedTuple{(nn_name,), Tuple{typeof(nn_out)}}((nn_out,)))
        nn_states = merge(nn_states, NamedTuple{(nn_name,), Tuple{typeof(st_nn)}}((st_nn,)))
    end
    
    # 4) Scale neural network parameters using the mapping
    scaled_nn_params = NamedTuple()
    for (nn_name, param_name) in zip(keys(m.NNs), m.neural_param_names)
        nn_cols = eachrow(nn_outputs[nn_name])
        
        # Create parameter for this NN
        nn_param = NamedTuple{(param_name,), Tuple{typeof(nn_cols[1])}}((nn_cols[1],))
        
        # Conditionally apply scaling based on scale_nn_outputs setting
        if m.scale_nn_outputs
            scaled_nn_val = scale_single_param(param_name, nn_param[param_name], parameters)
        else
            scaled_nn_val = nn_param[param_name]  # Use raw NN output without scaling
        end
        
        nn_scaled_param = NamedTuple{(param_name,), Tuple{typeof(scaled_nn_val)}}((scaled_nn_val,))
        
        # Merge with existing scaled parameters
        scaled_nn_params = merge(scaled_nn_params, nn_scaled_param)
    end

    # 5) Pick fixed parameters (handle empty case)
    if !isempty(m.fixed_param_names)
        fixed_vals = Tuple(st.fixed[f] for f in m.fixed_param_names)
        fixed_params = NamedTuple{Tuple(m.fixed_param_names), Tuple{typeof.(fixed_vals)...}}(fixed_vals)
    else
        fixed_params = NamedTuple()
    end

    all_params = merge(scaled_nn_params, global_params, fixed_params)

    # 6) unpack forcing data

    forcing_data = unpack_keyedarray(ds_k, m.forcing)
    all_kwargs = merge(forcing_data, all_params)
    
    # 7) Apply mechanistic model
    y_pred = m.mechanistic_model(; all_kwargs...)

    out = (;y_pred..., parameters = all_params, nn_outputs = nn_outputs)

    st_new = (; nn_states..., fixed = st.fixed)

    return out, (; st = st_new)
end

function (m::MultiNNHybridModel)(df::DataFrame, ps, st)
    @warn "Only makes sense in test mode, not training!"

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
