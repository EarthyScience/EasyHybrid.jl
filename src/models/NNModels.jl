export SingleNNModel, MultiNNModel, constructNNModel

using Lux, LuxCore
using ..EasyHybrid: hard_sigmoid

# Pure Neural Network Models (no mechanistic component)

struct SingleNNModel
    NN               :: Chain
    predictors       :: Vector{Symbol}
    targets          :: Vector{Symbol}
    scale_nn_outputs :: Bool
end

"""
    prepare_hidden_chain(hidden_layers, in_dim, out_dim; activation, input_batchnorm=false)

Construct a neural network `Chain` for use in NN models.

# Arguments
- `hidden_layers::Union{Vector{Int}, Chain}`: 
    - If a `Vector{Int}`, specifies the sizes of each hidden layer. 
      For example, `[32, 16]` creates two hidden layers with 32 and 16 units, respectively.
    - If a `Chain`, the user provides a pre-built chain of hidden layers (excluding input/output layers).
- `in_dim::Int`: Number of input features (input dimension).
- `out_dim::Int`: Number of output features (output dimension).
- `activation`: Activation function to use in hidden layers (default: `tanh`).
- `input_batchnorm::Bool`: If `true`, applies a `BatchNorm` layer to the input (default: `false`).

# Returns
- A `Chain` object representing the full neural network, with the following structure:
    - Optional input batch normalization (if `input_batchnorm=true`)
    - Input layer: `Dense(in_dim, h₁, activation)` where `h₁` is the first hidden size
    - Hidden layers: either user-supplied `Chain` or constructed from `hidden_layers`
    - Output layer: `Dense(hₖ, out_dim)` where `hₖ` is the last hidden size

where `h₁` is the first hidden size and `hₖ` the last.
"""
function prepare_hidden_chain(
    hidden_layers::Union{Vector{Int}, Chain},
    in_dim::Int,
    out_dim::Int;
    activation = tanh,
    input_batchnorm = false # apply batchnorm to input as an easy way for normalization
)
    if hidden_layers isa Chain
        # user gave a chain of hidden layers only
        first_h = hidden_layers[1].out_dims
        last_h  = hidden_layers[end].out_dims

        return Chain(
            input_batchnorm ? BatchNorm(in_dim, affine=false) : identity,
            Dense(in_dim, first_h, activation),
            hidden_layers.layers...,    
            Dense(last_h, out_dim)
        )
    else
        # user gave a vector of hidden‐layer sizes
        hs = hidden_layers
        # build the hidden‐to‐hidden part
        hidden_chain = length(hs) > 1 ? 
            Chain((Dense(hs[i], hs[i+1], activation) for i in 1:length(hs)-1)...) :
            Chain()
        return Chain(
            input_batchnorm ? BatchNorm(in_dim, affine=false) : identity,
            Dense(in_dim, hs[1], activation),
            hidden_chain.layers...,
            Dense(hs[end], out_dim)
        )
    end
end

"""
    constructNNModel(predictors, targets; hidden_layers, activation, scale_nn_outputs)

Main constructor: `hidden_layers` can be either
  • a `Vector{Int}` of sizes, or
  • a `Chain` of hidden-layer `Dense` blocks.
"""
function constructNNModel(
    predictors::Vector{Symbol},
    targets::Vector{Symbol};
    hidden_layers::Union{Vector{Int}, Chain} = [32, 16, 16],
    activation = tanh,
    scale_nn_outputs::Bool = true,
    input_batchnorm = false
)
    in_dim  = length(predictors)
    out_dim = length(targets)

    NN = prepare_hidden_chain(hidden_layers, in_dim, out_dim;
                              activation = activation,
                              input_batchnorm = input_batchnorm)

    return SingleNNModel(NN, predictors, targets, scale_nn_outputs)
end

# MultiNNModel remains as before
struct MultiNNModel
    NNs             :: NamedTuple
    predictors      :: NamedTuple
    targets         :: Vector{Symbol}
    scale_nn_outputs  :: Bool
end

function constructNNModel(
    predictors::NamedTuple,
    targets;
    scale_nn_outputs = true
)
    @assert collect(keys(predictors)) == targets "predictor names must match targets"
    NNs = NamedTuple()
    for (nn_name, preds) in pairs(predictors)
        nn = Chain(
            BatchNorm(length(preds), affine=false),
            Dense(length(preds), 15, sigmoid),
            Dense(15, 15, sigmoid),
            Dense(15, 1, x -> x^2)
        )
        NNs = merge(NNs, NamedTuple{(nn_name,), Tuple{typeof(nn)}}((nn,)))
    end
    return MultiNNModel(NNs, predictors, targets, scale_nn_outputs)
end

# LuxCore initial parameters for SingleNNModel
function LuxCore.initialparameters(rng::AbstractRNG, m::SingleNNModel)
    ps_nn, _ = LuxCore.setup(rng, m.NN)
    nt = (; ps = ps_nn)
    return nt
end

# LuxCore initial parameters for MultiNNModel
function LuxCore.initialparameters(rng::AbstractRNG, m::MultiNNModel)
    nn_params = NamedTuple()
    for (nn_name, nn) in pairs(m.NNs)
        ps_nn, _ = LuxCore.setup(rng, nn)
        nn_params = merge(nn_params, NamedTuple{(nn_name,), Tuple{typeof(ps_nn)}}((ps_nn,)))
    end
    nt = (; nn_params...)
    return nt
end

# LuxCore initial states for SingleNNModel
function LuxCore.initialstates(rng::AbstractRNG, m::SingleNNModel)
    _, st_nn = LuxCore.setup(rng, m.NN)
    nt = (; st = st_nn)
    return nt
end

# LuxCore initial states for MultiNNModel
function LuxCore.initialstates(rng::AbstractRNG, m::MultiNNModel)
    nn_states = NamedTuple()
    for (nn_name, nn) in pairs(m.NNs)
        _, st_nn = LuxCore.setup(rng, nn)
        nn_states = merge(nn_states, NamedTuple{(nn_name,), Tuple{typeof(st_nn)}}((st_nn,)))
    end
    nt = (; nn_states...)
    return nt
end

# Forward pass for SingleNNModel
function (m::SingleNNModel)(ds_k, ps, st)
    predictors = ds_k(m.predictors)
    nn_out, st_NN = LuxCore.apply(m.NN, predictors, ps.ps, st.st)
    nn_cols = eachrow(nn_out)
    nn_params = NamedTuple(zip(m.targets, nn_cols))
    if m.scale_nn_outputs
        scaled_nn_vals = Tuple(hard_sigmoid(nn_params[name]) for name in m.targets)
    else
        scaled_nn_vals = Tuple(nn_params[name] for name in m.targets)
    end
    scaled_nn_params = NamedTuple(zip(m.targets, scaled_nn_vals))

    out = (; scaled_nn_params...)
    st_new = (; st = st_NN)
    return out, (; st = st_new)
end

# Forward pass for MultiNNModel
function (m::MultiNNModel)(ds_k, ps, st)
    nn_inputs = NamedTuple()
    for (nn_name, predictors) in pairs(m.predictors)
        nn_inputs = merge(nn_inputs, NamedTuple{(nn_name,), Tuple{typeof(ds_k(predictors))}}((ds_k(predictors),)))
    end
    nn_outputs = NamedTuple()
    nn_states = NamedTuple()
    for (nn_name, nn) in pairs(m.NNs)
        nn_out, st_nn = LuxCore.apply(nn, nn_inputs[nn_name], ps[nn_name], st[nn_name])
        nn_outputs = merge(nn_outputs, NamedTuple{(nn_name,), Tuple{typeof(nn_out)}}((nn_out,)))
        nn_states = merge(nn_states, NamedTuple{(nn_name,), Tuple{typeof(st_nn)}}((st_nn,)))
    end
    scaled_nn_params = NamedTuple()
    for (nn_name, target_name) in zip(keys(m.NNs), m.targets)
        nn_output = nn_outputs[nn_name]
        nn_cols = eachrow(nn_output)
        nn_param = NamedTuple{(target_name,), Tuple{typeof(nn_cols[1])}}((nn_cols[1],))
        if m.scale_nn_outputs
            scaled_nn_val = hard_sigmoid(nn_param[target_name])
        else
            scaled_nn_val = nn_param[target_name]
        end
        nn_scaled_param = NamedTuple{(target_name,), Tuple{typeof(scaled_nn_val)}}((scaled_nn_val,))
        scaled_nn_params = merge(scaled_nn_params, nn_scaled_param)
    end
    out = (; scaled_nn_params..., nn_outputs = nn_outputs)
    st_new = (; nn_states...)
    return out, (; st = st_new)
end

# Display functions
function Base.display(m::SingleNNModel)
    println("Neural Network: ", m.NN)
    println("Predictors: ", m.predictors)
    println("scale NN outputs: ", m.scale_nn_outputs)
end

function Base.display(m::MultiNNModel)
    println("Neural Networks:")
    for (name, nn) in pairs(m.NNs)
        println("  $name: ", nn)
    end
    println("Predictors:")
    for (name, preds) in pairs(m.predictors)
        println("  $name: ", preds)
    end
    println("scale NN outputs: ", m.scale_nn_outputs)
end 