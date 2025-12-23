"""
    compute_loss(HM, x, (y_t, y_nan), ps, st, logging::LoggingLoss)

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
function compute_loss(
        HM::LuxCore.AbstractLuxContainerLayer, ps, st, (x, (y_t, y_nan));
        logging::LoggingLoss
    )

    targets = HM.targets
    ext_loss = extra_loss(logging)
    if logging.train_mode
        ŷall, st = HM(x, ps, st)

        ŷ = ŷall(window = axiskeys(y_t, :window))

        loss_value = _compute_loss(ŷ, y_t, y_nan, targets, training_loss(logging), logging.agg)
        # Add extra_loss if provided
        if ext_loss !== nothing
            extra_loss_value = ext_loss(ŷ)
            loss_value = logging.agg([loss_value, extra_loss_value...])
        end
        stats = NamedTuple()
    else
        ŷall, _ = HM(x, ps, LuxCore.testmode(st))

        ŷ = ŷall(window = axiskeys(y_t, :window))

        loss_value = _compute_loss(ŷ, y_t, y_nan, targets, loss_types(logging), logging.agg)
        # Add extra_loss entries if provided
        if ext_loss !== nothing
            extra_loss_values = ext_loss(ŷ)
            agg_extra_loss_value = logging.agg(extra_loss_values)
            loss_value = (; loss_value..., extra_loss = (; extra_loss_values..., Symbol(logging.agg) => agg_extra_loss_value))
        end
        stats = (; ŷ...)
    end
    return loss_value, st, stats
end

function _compute_loss(ŷ, y, y_nan, targets, loss_spec, agg::Function)
    losses = assemble_loss(ŷ, y, y_nan, targets, loss_spec)
    return agg(losses)
end

function _compute_loss(ŷ, y, y_nan, targets, loss_types::Vector, agg::Function)
        
    @show typeof(ŷ)
    @show typeof(y)
    out_loss_types = [
        begin
                losses = assemble_loss(ŷ, y, y_nan, targets, loss_type)
                agg_loss = agg(losses)
                NamedTuple{(targets..., Symbol(agg))}([losses..., agg_loss])
            end
            for loss_type in loss_types
    ]
    _names = [_loss_name(lt) for lt in loss_types]
    return NamedTuple{Tuple(_names)}([out_loss_types...])
end

"""
    _compute_loss(ŷ, y, y_nan, targets, loss_spec, agg::Function)
    _compute_loss(ŷ, y, y_nan, targets, loss_types::Vector, agg::Function)

Compute the loss for the given predictions and targets using the specified training loss (or vector of losses) type and aggregation function.

# Arguments:
- `ŷ`: Predicted values.
- `y`: Target values.
- `y_nan`: Mask for NaN values.
- `targets`: The targets for which the loss is computed.
- `loss_spec`: The loss type to use during training, e.g., `:mse`.
- `loss_types::Vector`: A vector of loss types to compute, e.g., `[:mse, :mae]`.
- `agg::Function`: The aggregation function to apply to the computed losses, e.g., `sum` or `mean`.

Returns a single loss value if `loss_spec` is provided, or a NamedTuple of losses for each type in `loss_types`.
"""
function _compute_loss end

function assemble_loss(ŷ, y, y_nan, targets, loss_spec)
    return [
        _apply_loss(ŷ[target], _get_target_y(y, target), _get_target_nan(y_nan, target), loss_spec)
            for target in targets
    ]
end

function assemble_loss(ŷ, y, y_nan, targets, loss_spec::PerTarget)
    @assert length(targets) == length(loss_spec.losses) "Length of targets and PerTarget losses tuple must match"
    losses = [
        _apply_loss(
                ŷ,
                _get_target_y(y, target),
                _get_target_nan(y_nan, target),
                target,
                loss_t
            ) for (target, loss_t) in zip(targets, loss_spec.losses)
    ]
    return losses
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
function _apply_loss(ŷ, y, y_nan, target, loss_spec)
    return _apply_loss(ŷ[target], y, y_nan, loss_spec)
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

_get_target_y(y, target) = y(target)
_get_target_y(y::AbstractDimArray, target) = y[col = At(target)] # assumes the DimArray uses :col indexing
_get_target_y(y::AbstractDimArray, targets::Vector) = y[col = At(targets)] # for multiple targets

function _get_target_y(y::Tuple, target)
    y_obs, y_sigma = y
    sigma = y_sigma isa Number ? y_sigma : y_sigma(target)
    y_obs_val = _get_target_y(y_obs, target)
    return (y_obs_val, sigma)
end


"""
    _get_target_y(y, target)
Helper function to extract target-specific values from `y`, handling cases where `y` can be a tuple of `(y_obs, y_sigma)`.
"""
function _get_target_y end

_get_target_nan(y_nan, target) = y_nan(target)
_get_target_nan(y_nan::AbstractDimArray, target) = y_nan[col = At(target)] # assumes the DimArray uses :col indexing
_get_target_nan(y_nan::AbstractDimArray, targets::Vector) = y_nan[col = At(targets)] # for multiple targets

"""
    _get_target_nan(y_nan, target)

Helper function to extract target-specific values from `y_nan`.
"""
function _get_target_nan end

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
