export LoggingLoss, DataAndPhysicsLoss, Physics
import Base: +

"""
    LPPP(loss)
Prior-penalty-physics loss

Wrapper to indicate a physics-based loss that operates on the full prediction ŷ.
Physics losses are computed once per batch, not per target.
"""
struct LPPP{L}
    loss::L
end

"""
    LDataPPP(data_loss, physics_loss)

A container for an optional data-driven loss and one or more physics-based losses.
The `physics_loss` can be a single `Physics` instance or a tuple of them.
"""
struct LDataPPP{D, P<:Tuple}
    ℒ_data::D
    ℒ_phys::P

    function LDataPPP(ℒ_data, ℒ_phys::Tuple)
        all(p -> p isa Physics, ℒ_phys) || throw(ArgumentError("All elements in physics_loss must be of type Physics."))
        new{typeof(ℒ_data), typeof(ℒ_phys)}(ℒ_data, ℒ_phys)
    end
end

LDataPPP(; ℒ_data = nothing, ℒ_phys = ()) = LDataPPP(ℒ_data, ℒ_phys)

"""
    +(loss1, loss2)

Overloads the `+` operator to combine data and physics losses into a `DataAndPhysicsLoss` object.

# Examples
```julia
:mse + Physics(smoothness_loss)
custom_loss + Physics(conservation_loss)
:mse + Physics(smoothness_loss) + Physics(conservation_loss)
```
"""
const DataLossSpec = Union{Symbol, Function, Tuple}

# Data loss + Physics loss
+(data_loss::DataLossSpec, physics_loss::LPPP) = LDataPPP(data_loss, (physics_loss,))

# Physics loss + Data loss
+(physics_loss::LPPP, data_loss::DataLossSpec) = LDataPPP(data_loss, (physics_loss,))

# Two physics losses
+(p1::LPPP, p2::LPPP) = LDataPPP(nothing, (p1, p2))

# DataAndPhysicsLoss + Physics loss
+(dp::LDataPPP, p::LPPP) = LDataPPP(dp.data_loss, (dp.physics_loss..., p))
+(p::LPPP, dp::LDataPPP) = LDataPPP(dp.data_loss, (p, dp.physics_loss...))

"""
    PerTarget(losses)

A wrapper to indicate that a tuple of losses should be applied on a per-target basis.
"""
struct PerTarget{T<:Tuple}
    losses::T
end

const LossSpec = Union{Symbol, Function, Tuple, LPPP, PerTarget, LDataPPP}

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
function lossfn(HM::Union{LuxCore.AbstractLuxContainerLayer, SingleNNHybridModel, MultiNNHybridModel, SingleNNModel, MultiNNModel}, x, (y_t, y_nan), ps, st, logging::LoggingLoss)
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
    ŷ, st = HM(x, ps, st)
    y = _get_target_y(y_t, targets)
    y_nan = _get_target_nan(y_nan, targets)
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

function _apply_loss(ŷ, y, y_nan, loss_spec::LPPP)
    return loss_fn(ŷ, y, y_nan, loss_spec)
end


function _apply_loss(ŷ, y, y_nan, target, ℒ_mix::LDataPPP{D, P}) where {D, P}
    data_loss = ℒ_mix.ℒ_data === nothing ? 0.0f0 : _apply_loss(ŷ[target], y, y_nan, ℒ.ℒ_data)
    phys_loss = ℒ_mix.ℒ_phys === nothing ? 0.0f0 : sum(𝓁 -> _apply_loss(ŷ, y, y_nan, 𝓁), ℒ_mix.ℒ_phys; init=0.0f0)
    return data_loss + phys_loss
end

function _apply_loss(ŷ_all, y, y_nan, target, ℒ_single)
    return _apply_loss(ŷ_all[target], y, y_nan, ℒ_single)
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
    losses = assemble_loss(ŷ, y, y_nan, targets, loss_spec)
    return agg(losses)
end

function compute_loss(ŷ, y, y_nan, targets, loss_types::Vector, agg::Function)
    out_loss_types = [
        begin
            losses = assemble_loss(ŷ, y, y_nan, targets, loss_type)
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

_get_target_y(y, target) = y(target)
_get_target_y(y::AbstractDimArray, target) = y[col=At(target)]
_get_target_y(y::AbstractDimArray, targets::Vector) = y[col=At(targets)]

function _get_target_y(y::Tuple, target)
    y_obs, y_sigma = y
    sigma = y_sigma isa Number ? y_sigma : y_sigma(target)
    y_obs_val = _get_target_y(y_obs, target)
    return (y_obs_val, sigma)
end

"""
    _get_target_y(y, target)
Helper function to extract target-specific values from `y`, handling cases where `y` may be a tuple of `(y_obs, y_sigma)`.
"""
function _get_target_y end

_get_target_nan(y_nan, target) = y_nan(target)
_get_target_nan(y_nan::AbstractDimArray, target) = y_nan[col=At(target)]
_get_target_nan(y_nan::AbstractDimArray, targets::Vector) = y_nan[col=At(targets)]

"""
    _get_target_nan(y_nan, target)

Helper function to extract target-specific values from `y_nan`.
"""
function _get_target_nan end

function assemble_loss(ŷ, y, y_nan, targets, ℒ_mix::LDataPPP{D, P}) where {D, P}
    data_losses = if ℒ_mix.ℒ_data != nothing
        [_apply_loss(ŷ[target], _get_target_y(y, target), _get_target_nan(y_nan, target), ℒ.ℒ_data)
            for target in targets]
    end
    phys_losses = [_apply_loss(ŷ, nothing, nothing, 𝓁) for 𝓁 in ℒ_mix.ℒ_phys]
    return vcat(data_losses..., phys_losses...)
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

"""
    assemble_loss(ŷ, y, y_nan, targets, loss_spec)

Helper function to assemble a vector of losses for each target based on the provided loss specification.
    
# Arguments
- `ŷ`: Predictions for all targets.
- `y`: Target values (can be a function, tuple, or AbstractDimArray).
- `y_nan`: NaN mask (function or array).
- `targets`: List of target names.
- `loss_spec`: Loss specification (Symbol, Function, or Tuple).

# Returns
- Vector of losses for each target.
"""
function assemble_loss end

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

function _loss_name(loss_spec::DataAndPhysicsLoss)
    data_name = loss_spec.data_loss === nothing ? "" : _loss_name(loss_spec.data_loss)
    if isempty(loss_spec.physics_loss)
        return Symbol(data_name)
    end
    num_physics = length(loss_spec.physics_loss)
    return Symbol(data_name, "_plus_", num_physics, "_physics")
end

"""
    _loss_name(loss_spec::Symbol|Function|Tuple)

Helper function to generate a meaningful name for a loss specification
"""
function _loss_name end