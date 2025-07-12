export MultiNNHybridModel, constructMultiNNHybridModel, scale_single_param, AbstractHybridModel, build_hybrid, ParameterContainer, default, lower, upper, hard_sigmoid

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

function build_hybrid(paras::NamedTuple, f::DataType)
    ca = EasyHybrid.ParameterContainer(paras)
    return f(ca)
end

# ───────────────────────────────────────────────────────────────────────────
# Multi-Neural Network Hybrid Model Structure
struct MultiNNHybridModel
    NNs           :: NamedTuple  # Multiple neural networks
    predictors    :: NamedTuple  # Predictors for each NN
    forcing       :: Vector{Symbol}
    targets       :: Vector{Symbol}
    mechanistic_model :: Function
    parameters    :: AbstractHybridModel          # holds the full ComponentMatrix
    neural_param_names :: Vector{Symbol}       # which parameters come from the NNs (auto-determined)
    global_param_names :: Vector{Symbol}       # which parameters are estimated globally
    fixed_param_names  :: Vector{Symbol}       # which parameters are fixed
    scale_nn_outputs  :: Bool                 # whether to scale NN outputs (default: true)
end

# Generic constructor for multi-NN hybrid model - similar to constructHybridModel
function constructMultiNNHybridModel(
    predictors,             # NamedTuple of predictor vectors for each NN
    forcing,                # forcing variables
    targets,                # target variables
    mechanistic_model,      # mechanistic model function
    parameters,             # parameter container
    neural_param_names,     # which parameters come from NNs (auto-determined from predictors)
    global_param_names;     # which parameters are estimated globally
    scale_nn_outputs = true # whether to scale NN outputs
)
    all_names = pnames(parameters)
    @assert all(n in all_names for n in neural_param_names) "neural_param_names ⊆ param_names"
    
    # Check that predictor names match neural parameter names
    @assert collect(keys(predictors)) == neural_param_names "predictor names must match neural parameter names"
    
    # Create neural networks based on predictors
    NNs = NamedTuple()
    for (nn_name, preds) in pairs(predictors)
        # Create a simple NN for each predictor set
        nn = Chain(
            BatchNorm(length(preds), affine=false),
            Dense(length(preds), 15, sigmoid), 
            Dense(15, 15, sigmoid), 
            Dense(15, 1, x -> x^2)  # Output 1 parameter per NN with positive activation
        )
        NNs = merge(NNs, NamedTuple{(nn_name,), Tuple{typeof(nn)}}((nn,)))
    end
    
    fixed_param_names = [ n for n in all_names if !(n in [neural_param_names..., global_param_names...]) ]
    
    return MultiNNHybridModel(NNs, predictors, forcing, targets, mechanistic_model, parameters, neural_param_names, global_param_names, fixed_param_names, scale_nn_outputs)
end

# ───────────────────────────────────────────────────────────────────────────
# Initial parameters: scalars come from parameters.table[:, :default]
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
        for g in m.global_param_names
            random_val = rand(rng, Float32)
            nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([random_val],)))
        end
    end
    
    return nt
end

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
    return ℓ .+ (u .- ℓ) .* hard_sigmoid.(raw_val)
end

# ───────────────────────────────────────────────────────────────────────────
# Forward pass for multi-NN hybrid model
function (m::MultiNNHybridModel)(ds_k, ps, st)
    # 1) Get features for each neural network
    nn_inputs = NamedTuple()
    for (nn_name, predictors) in pairs(m.predictors)
        nn_inputs = merge(nn_inputs, NamedTuple{(nn_name,), Tuple{typeof(ds_k(predictors))}}((ds_k(predictors),)))
    end
    
    forcing_data = ds_k(m.forcing)
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
    
    #println("st: ", st)
    for (nn_name, nn) in pairs(m.NNs)
        nn_out, st_nn = LuxCore.apply(nn, nn_inputs[nn_name], ps[nn_name], st[nn_name])
        nn_outputs = merge(nn_outputs, NamedTuple{(nn_name,), Tuple{typeof(nn_out)}}((nn_out,)))
        nn_states = merge(nn_states, NamedTuple{(nn_name,), Tuple{typeof(st_nn)}}((st_nn,)))
    end
    
    # 4) Scale neural network parameters using the mapping
    scaled_nn_params = NamedTuple()
    for (nn_name, param_name) in zip(keys(m.NNs), m.neural_param_names)
        nn_output = nn_outputs[nn_name]
        nn_cols = eachrow(nn_output)
        
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

    # 6) Apply mechanistic model
    y_pred = m.mechanistic_model(forcing_data; all_params...)

    out = (;y_pred..., parameters = all_params, nn_outputs = nn_outputs)

    st_new = (; nn_states..., fixed = st.fixed)
    #st_new = (; st = nn_states)

    #println("st_new: ", st_new)

    return out, (; st = st_new)
end

function Base.display(hm::MultiNNHybridModel)
    println("Neural Networks:")
    for (name, nn) in pairs(hm.NNs)
        println("  $name: ", nn)
    end
    
    println("Predictors:")
    for (name, preds) in pairs(hm.predictors)
        println("  $name: ", preds)
    end
    
    println("neural parameters: ", hm.neural_param_names)
    println("global parameters: ", hm.global_param_names)
    println("fixed parameters: ", hm.fixed_param_names)
    println("scale NN outputs: ", hm.scale_nn_outputs)

    println("parameter defaults and bounds:")
    display(hm.parameters)
end 