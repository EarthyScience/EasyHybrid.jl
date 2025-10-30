export LoggingLoss

const LossSpec = Union{Symbol, Function, Tuple}

"""
    LoggingLoss

A structure to define a logging loss function for hybrid models.

# Arguments
- `loss_types`: A vector of loss specifications (Symbol, Function or Tuple)
  - Symbol: predefined loss, e.g. `:mse`
  - Function: custom loss function, e.g. `custom_loss`
  - Tuple: function with args/kwargs:
    - `(f, args)`: positional args, e.g. `(weighted_loss, (0.5,))`
    - `(f, kwargs)`: keyword args, e.g. `(scaled_loss, (scale=2.0,))`
    - `(f, args, kwargs)`: both, e.g. `(complex_loss, (0.5,), (scale=2.0,))`
- `training_loss`: The loss specification to use during training (same format as above)
- `agg`: Function to aggregate losses across targets, e.g. `sum` or `mean`
- `train_mode`: If true, uses `training_loss`; otherwise uses `loss_types`.

# Examples
```julia
# Simple predefined loss
logging = LoggingLoss(
    loss_types=[:mse, :mae],
    training_loss=:mse
)

# Custom loss function
custom_loss(ŷ, y) = mean(abs2, ŷ .- y)
logging = LoggingLoss(
    loss_types=[:mse, custom_loss],
    training_loss=custom_loss
)

# With arguments/kwargs
weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
scaled_loss(ŷ, y; scale=1.0) = scale * mean(abs2, ŷ .- y)
logging = LoggingLoss(
    loss_types=[:mse, (weighted_loss, (0.5,)), (scaled_loss, (scale=2.0,))],
    training_loss=(weighted_loss, (0.5,))
)
```
"""
struct LoggingLoss{T<:Function, L<:AbstractVector{<:LossSpec}, TL<:LossSpec}
    loss_types::L
    training_loss::TL
    agg::T
    train_mode::Bool

    function LoggingLoss{T,L,TL}(;
        loss_types = [:mse],
        training_loss = :mse,
        agg::T = sum,
        train_mode::Bool = true
    ) where {T<:Function, L<:AbstractVector{<:LossSpec}, TL<:LossSpec}
        lt = Vector{LossSpec}(loss_types)
        new{T, typeof(lt), typeof(training_loss)}(lt, training_loss, agg, train_mode)
    end
end

# Outer constructor
function LoggingLoss(;
    loss_types = [:mse],
    training_loss = :mse,
    agg::F = sum,
    train_mode::Bool = true
) where {F<:Function}
    LoggingLoss{F, Vector{LossSpec}, typeof(training_loss)}(;
        loss_types = loss_types,
        training_loss = training_loss,
        agg = agg,
        train_mode = train_mode
    )
end

"""
    lossfn(HM, x, (y_t, y_nan), ps, st, logging::LoggingLoss)

Main loss function for hybrid models that handles both training and evaluation modes.

# Arguments
- `HM`: The hybrid model (AbstractLuxContainerLayer or specific model type)
- `x`: Input data for the model
- `(y_t, y_nan)`: Tuple containing target values and NaN mask functions/arrays
- `ps`: Model parameters
- `st`: Model state
- `logging`: LoggingLoss configuration

# Returns
- In training mode (`logging.train_mode = true`):
  - `(loss_value, st)`: Single loss value and updated state
- In evaluation mode (`logging.train_mode = false`):
  - `(loss_values, st, ŷ)`: NamedTuple of losses, state and predictions
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

# Arguments
- `HM`: The hybrid model
- `x`: Input data
- `(y_t, y_nan)`: Tuple of target values and NaN mask (functions or arrays)
- `ps`: Model parameters
- `st`: Model state
- `targets`: Target variable names

# Returns
- `ŷ`: Model predictions
- `y`: Target values
- `y_nan`: NaN mask
- `st`: Updated model state
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

function _apply_loss(ŷ, y, y_nan, loss_spec::Symbol)
    return loss_fn(ŷ, y, y_nan, Val(loss_spec))
end

function _apply_loss(ŷ, y, y_nan, loss_spec::Function)
    return loss_fn(ŷ, y, y_nan, loss_spec)
end

function _apply_loss(ŷ, y, y_nan, loss_spec::Tuple)
    return loss_fn(ŷ, y, y_nan, loss_spec)
end

"""
    _apply_loss(ŷ, y, y_nan, loss_spec)

Helper function to apply the appropriate loss function based on the specification type.

# Arguments
- `ŷ`: Predictions for a single target
- `y`: Target values for a single target
- `y_nan`: NaN mask for a single target
- `loss_spec`: Loss specification (Symbol, Function, or Tuple)

# Returns
- Computed loss value
"""
function _apply_loss end

function compute_loss(ŷ, y, y_nan, targets, loss_spec, agg::Function)
    losses = [_apply_loss(ŷ[k], y(k), y_nan(k), loss_spec) for k in targets]
    return agg(losses)
end

function compute_loss(ŷ, y::Tuple, y_nan, targets, loss_spec, agg::Function)
    y_obs, y_sigma = y
    losses = [_apply_loss(ŷ[k], (y_obs(k), y_sigma(k)), y_nan(k), loss_spec) for k in targets]
    return agg(losses)
end

function compute_loss(ŷ, y::AbstractDimArray, y_nan::AbstractDimArray, targets, loss_spec, agg::Function)
    losses = [_apply_loss(ŷ[k], y[col=At(k)], y_nan[col=At(k)], loss_spec) for k in targets]
    return agg(losses)
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

function compute_loss(ŷ, y::Tuple, y_nan, targets, loss_types::Vector, agg::Function)
    y_obs, y_sigma = y
    out_loss_types = [
        begin
            losses = [_apply_loss(ŷ[k], (y_obs(k), y_sigma(k)), y_nan(k), loss_type) for k in targets]
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

"""
    compute_loss(ŷ, y, y_nan, targets, training_loss, agg::Function)
    compute_loss(ŷ, y, y_nan, targets, loss_types::Vector, agg::Function)
    compute_loss(ŷ, y::Tuple, y_nan, targets, training_loss, agg::Function)
    compute_loss(ŷ, y::Tuple, y_nan, targets, loss_types::Vector, agg::Function)
    compute_loss(ŷ, y::AbstractDimArray, y_nan::AbstractDimArray, targets, training_loss, agg::Function)
    compute_loss(ŷ, y::AbstractDimArray, y_nan::AbstractDimArray, targets, loss_types::Vector, agg::Function)


Compute the loss for the given predictions and targets using the specified training loss (or vector of losses) type and aggregation function.

# Arguments:
- `ŷ`: Predicted values.
- `y`: Target values (as a function, tuple `(y, y_sigma)`, or AbstractDimArray).
- `y_nan`: Mask for NaN values.
- `targets`: The targets for which the loss is computed.
- `training_loss`: The loss type to use during training, e.g., `:mse` or a custom function.
- `loss_types::Vector`: A vector of loss types to compute, e.g., `[:mse, :mae]`.
- `agg::Function`: The aggregation function to apply to the computed losses, e.g., `sum` or `mean`.

Returns a single loss value if `training_loss` is provided, or a NamedTuple of losses for each type in `loss_types`.
"""
function compute_loss end

# Helper to generate meaningful names for loss types
function _loss_name(loss_spec::Symbol)
    return loss_spec
end

function _loss_name(loss_spec::Function)
    raw_name = nameof(typeof(loss_spec))
    clean_name = Symbol(replace(string(raw_name), "#" => ""))
    return clean_name
end

function _loss_name(loss_spec::Tuple)
    return _loss_name(loss_spec[1])
end

"""
    _loss_name(loss_spec::Symbol|Function|Tuple)

Helper function to generate a meaningful name for a loss specification
"""
function _loss_name end