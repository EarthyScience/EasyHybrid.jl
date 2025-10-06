export LoggingLoss

"""
    LoggingLoss

A structure to define a logging loss function for hybrid models.

# Arguments:
- `loss_types`: A vector of loss specifications (Symbol, Function or Tuple)
  e.g. `[:mse, custom_loss, (weighted_loss, (0.5,))]`
- `training_loss`: The loss to use during training
  e.g. `:mse` or `custom_loss` or `(weighted_loss, (0.5,))`
- `agg`: Function to aggregate losses, e.g. `sum` or `mean`
- `train_mode`: If true, uses training_loss; otherwise uses loss_types
"""
Base.@kwdef struct LoggingLoss{T<:Function}
    loss_types = [:mse]
    training_loss = :mse
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
        loss_value = compute_loss(ŷ, y, y_nan, targets, logging.training_loss, logging.agg)
        return loss_value, st
    else
        loss_value = compute_loss(ŷ, y, y_nan, targets, logging.loss_types, logging.agg)
        return loss_value, st, ŷ
    end
end

function lossfn(HM::Union{SingleNNHybridModel, MultiNNHybridModel, SingleNNModel, MultiNNModel}, x, (y_t, y_nan), ps, st, logging::LoggingLoss)
    targets = HM.targets
    ŷ, y, y_nan, st = get_predictions_targets(HM, x, (y_t, y_nan), ps, st, targets)
    if logging.train_mode
        loss_value = compute_loss(ŷ, y, y_nan, targets, logging.training_loss, logging.agg)
        return loss_value, st
    else
        loss_value = compute_loss(ŷ, y, y_nan, targets, logging.loss_types, logging.agg)
        return loss_value, st, ŷ
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

function get_predictions_targets(
    HM, 
    x::AbstractDimArray, 
    ys::Tuple{<:AbstractDimArray,<:AbstractDimArray}, 
    ps, st, targets
)
    y_t, y_nan = ys
    ŷ, st = HM(x, ps, st)
    y     = y_t[col=At(targets)]
    y_nan = y_nan[col=At(targets)]
    return ŷ, y, y_nan, st
end

function compute_loss(ŷ, y, y_nan, targets, loss_spec, agg::Function)
    losses = [_apply_loss(ŷ[k], y(k), y_nan(k), loss_spec) for k in targets]
    return agg(losses)
end

function compute_loss(ŷ, y::AbstractDimArray, y_nan::AbstractDimArray, targets, loss_spec, agg::Function)
    losses = [_apply_loss(ŷ[k], y[col=At(k)], y_nan[col=At(k)], loss_spec) for k in targets]
    return agg(losses)
end
function _apply_loss(ŷ, y, y_nan, loss_spec::Symbol)
    return loss_fn(ŷ, y, y_nan, Val(loss_spec))
end

function _apply_loss(ŷ, y, y_nan, loss_spec::Function)
    return loss_fn(ŷ, y, y_nan, loss_spec)
end

function _apply_loss(ŷ, y, y_nan, loss_spec::Tuple)
    return loss_fn(ŷ, y, y_nan, loss_spec)
end

function compute_loss(ŷ, y, y_nan, targets, loss_types::Vector, agg::Function)
    out_loss_types = [
        begin
            losses = [_apply_loss(ŷ[k], y(k), y_nan(k), loss_type) for k in targets]
            agg_loss = agg(losses)
            NamedTuple{(targets..., Symbol(agg))}([losses..., agg_loss])
        end
        for loss_type in loss_types]
        _names = [_loss_name(lt) for lt in loss_types]
    return NamedTuple{Tuple(_names)}([out_loss_types...])
end
function compute_loss(ŷ, y::AbstractDimArray, y_nan::AbstractDimArray, targets, loss_types::Vector, agg::Function)
    out_loss_types = [
        begin
            losses = [_apply_loss(ŷ[k], y[col=At(k)], y_nan[col=At(k)], loss_type) for k in targets]
            agg_loss = agg(losses)
            NamedTuple{(targets..., Symbol(agg))}([losses..., agg_loss])
        end
        for loss_type in loss_types]
        _names = [_loss_name(lt) for lt in loss_types]
    return NamedTuple{Tuple(_names)}([out_loss_types...])
end

# Helper to generate meaningful names for loss types
function _loss_name(loss_spec::Symbol)
    return loss_spec
end

function _loss_name(loss_spec::Function)
    return nameof(typeof(loss_spec))
end

function _loss_name(loss_spec::Tuple)
    return nameof(typeof(loss_spec[1]))
end

"""
    compute_loss(ŷ, y, y_nan, targets, training_loss, agg::Function)
    compute_loss(ŷ, y, y_nan, targets, loss_types::Vector, agg::Function)

Compute the loss for the given predictions and targets using the specified training loss (or vector of losses) type and aggregation function.

# Arguments:
- `ŷ`: Predicted values.
- `y`: Target values.
- `y_nan`: Mask for NaN values.
- `targets`: The targets for which the loss is computed.
- `training_loss`: The loss type to use during training, e.g., `:mse`.
- `loss_types::Vector`: A vector of loss types to compute, e.g., `[:mse, :mae]`.
- `agg::Function`: The aggregation function to apply to the computed losses, e.g., `sum` or `mean`.

Returns a single loss value if `training_loss` is provided, or a NamedTuple of losses for each type in `loss_types`.
"""
function compute_loss end