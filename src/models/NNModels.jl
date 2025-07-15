export SingleNNModel, MultiNNModel, constructNNModel

using Lux, LuxCore
using ..EasyHybrid: hard_sigmoid

# Pure Neural Network Models (no mechanistic component)

struct SingleNNModel
    NN              :: Chain
    predictors      :: Vector{Symbol}
    targets         :: Vector{Symbol}
    scale_nn_outputs  :: Bool
end

struct MultiNNModel
    NNs             :: NamedTuple
    predictors      :: NamedTuple
    targets         :: Vector{Symbol}
    scale_nn_outputs  :: Bool
end

# Constructor for SingleNNModel
function constructNNModel(
    predictors::Vector{Symbol},
    targets;
    scale_nn_outputs = true
)
    NN = Chain(Dense(length(predictors), 64, tanh), Dense(64, 128, tanh), Dense(128, length(targets), tanh))
    return SingleNNModel(NN, predictors, targets, scale_nn_outputs)
end

# Constructor for MultiNNModel
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
    st_new = (; st = st_NN.st)
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