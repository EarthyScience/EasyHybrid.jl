export LoggingLoss

"""
    LoggingLoss

A structure to define a logging loss function for hybrid models.
It allows for multiple loss types and an aggregation function to be specified.

# Arguments:
- `loss_types::Vector{Symbol}`: A vector of loss types to compute, e.g., `[:mse, :mae]`.
- `training_loss::Symbol`: The loss type to use during training, e.g., `:mse`.
- `agg::Function`: A function to aggregate the losses, e.g., `sum` or `mean`.
- `train_mode::Bool`: A flag indicating whether the model is in training mode. If `true`, it uses `training_loss`; otherwise, it uses `loss_types`.
"""
Base.@kwdef struct LoggingLoss{T<:Function}
    loss_types::Vector{Symbol} = [:mse]
    training_loss::Symbol = :mse
    agg::T = sum
    train_mode::Bool = true
end

"""
    lossfn(HM::LuxCore.AbstractLuxContainerLayer, x, (y_t, y_nan), ps, st, logging::LoggingLoss)

Arguments:
- `HM::LuxCore.AbstractLuxContainerLayer`: The hybrid model to compute the loss for.
- `x`: Input data for the model.
- `(y_t, y_nan)`: Tuple containing the target values and a mask for NaN values.
- `ps`: Parameters of the model.
- `st`: State of the model.
- `logging::LoggingLoss`: Logging configuration for the loss function.
"""
function lossfn(HM::LuxCore.AbstractLuxContainerLayer, x, (y_t, y_nan), ps, st, logging::LoggingLoss)
    targets = HM.targets
    ŷ, y, y_nan, st = get_predictions_targets(HM, x, (y_t, y_nan), ps, st, targets)
    if logging.train_mode
        return compute_loss(ŷ, y, y_nan, targets, logging.training_loss, logging.agg), st
    else
        return compute_loss(ŷ, y, y_nan, targets, logging.loss_types, logging.agg), st
    end
end

function lossfn(HM::Union{SingleNNHybridModel, MultiNNHybridModel, SingleNNModel, MultiNNModel}, x, (y_t, y_nan), ps, st, logging::LoggingLoss)
    targets = HM.targets
    ŷ, y, y_nan, st = get_predictions_targets(HM, x, (y_t, y_nan), ps, st, targets)
    if logging.train_mode
        return compute_loss(ŷ, y, y_nan, targets, logging.training_loss, logging.agg), st
    else
        return compute_loss(ŷ, y, y_nan, targets, logging.loss_types, logging.agg), st
    end
end


"""
    get_predictions_targets(HM, x, (y_t, y_nan), ps, st, targets)
Get predictions and targets from the hybrid model and return them along with the NaN mask.
"""
function get_predictions_targets(HM, x, (y_t, y_nan), ps, st, targets)
    ŷ, st = HM(x, ps, st) #TODO the output st can contain more than st, e.g. Rb is that what we want?
    y = y_t(HM.targets)
    y_nan = y_nan(HM.targets)
    return ŷ, y, y_nan, st #TODO has to be done otherwise e.g. Rb is passed as a st and messes up the training
end
function compute_loss(ŷ, y, y_nan, targets, training_loss::Symbol, agg::Function)
    losses = [loss_fn(ŷ[k], y(k), y_nan(k), Val(training_loss)) for k in targets]
    return agg(losses)
end
function compute_loss(ŷ, y, y_nan, targets, loss_types::Vector{Symbol}, agg::Function)
    out_loss_types = [
        begin
            losses = [loss_fn(ŷ[k], y(k), y_nan(k), Val(loss_type)) for k in targets]
            agg_loss = agg(losses)
            NamedTuple{(targets..., Symbol(agg))}([losses..., agg_loss])
        end
        for loss_type in loss_types]
    return NamedTuple{Tuple(loss_types)}([out_loss_types...])
end

"""
    compute_loss(ŷ, y, y_nan, targets, training_loss::Symbol, agg::Function)
    compute_loss(ŷ, y, y_nan, targets, loss_types::Vector{Symbol}, agg::Function)

Compute the loss for the given predictions and targets using the specified training loss (or vector of losses) type and aggregation function.

# Arguments:
- `ŷ`: Predicted values.
- `y`: Target values.
- `y_nan`: Mask for NaN values.
- `targets`: The targets for which the loss is computed.
- `training_loss::Symbol`: The loss type to use during training, e.g., `:mse`.
- `loss_types::Vector{Symbol}`: A vector of loss types to compute, e.g., `[:mse, :mae]`.
- `agg::Function`: The aggregation function to apply to the computed losses, e.g., `sum` or `mean`.

Returns a single loss value if `training_loss` is provided, or a NamedTuple of losses for each type in `loss_types`.
"""
function compute_loss end