export LoggingLoss
export loss_types, training_loss, extra_loss

abstract type LossSpec end

struct SymbolicLoss{S} <: LossSpec
    name::Symbol
end
SymbolicLoss(s::Symbol) = SymbolicLoss{s}(s)

struct FunctionLoss <: LossSpec
    f::Function
end

struct ParameterizedLoss <: LossSpec
    f::Function
    args::Tuple
    kwargs::NamedTuple
end

ParameterizedLoss(f::Function) = ParameterizedLoss(f, (), NamedTuple())
ParameterizedLoss(f::Function, args::Tuple) = ParameterizedLoss(f, args, NamedTuple())
ParameterizedLoss(f::Function, kwargs::NamedTuple) = ParameterizedLoss(f, (), kwargs)

struct ExtraLoss{F} <: LossSpec
    f::F
end

"""
    ParamLoss(f)

Wrapper for a *full-context* training loss with signature
`f(ŷ, y, y_nan, ps, targets, parameters)`. Unlike a classic custom loss
`f(ŷ_masked, y_masked)`, it receives the full (unmasked) predictions `ŷ`, targets
`y`, NaN masks `y_nan`, the raw parameters `ps`, the `targets` names, and the model
`parameters` (i.e. `ŷ.parameters`: NN-predicted, global and fixed values), and must
return a scalar loss.

Users normally do not construct this directly: passing a function with this
6-argument signature to `training_loss` auto-detects it (see [`_accepts_params`](@ref)).
It is the natural choice when the loss needs a learned quantity such as a noise
scale `parameters.sigma`, e.g. a Gaussian negative log-likelihood.
"""
struct ParamLoss <: LossSpec
    f::Function
end

"""
    PerTarget(losses)

A wrapper to indicate that a tuple of losses should be applied on a per-target basis.
"""
struct PerTarget{T <: Tuple}
    losses::T
end

Base.length(pt::PerTarget) = length(pt.losses)
Base.getindex(pt::PerTarget, i::Int) = pt.losses[i]

Base.iterate(pt::PerTarget) = iterate(pt.losses)
Base.iterate(pt::PerTarget, state) = iterate(pt.losses, state)
Base.first(pt::PerTarget) = first(pt.losses)
Base.last(pt::PerTarget) = last(pt.losses)
Base.keys(pt::PerTarget) = keys(pt.losses)
Base.eltype(::Type{PerTarget{T}}) where {T <: Tuple} = eltype(T)

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
- `extra_loss`: Optional function `(ŷ, ps; kwargs...) -> NamedTuple` (or splattable collection) added to training loss (default: `nothing`)
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
struct LoggingLoss{L <: Union{LossSpec, PerTarget}, E <: LossSpec, T <: Function, TM}
    loss_types::Vector{LossSpec}
    training_loss::L
    extra_loss::E
    agg::T
    train_mode::Bool
end

function LoggingLoss(;
        loss_types = [:mse],
        training_loss = :mse,
        extra_loss = nothing,
        agg::F = sum,
        train_mode::Bool = true
    ) where {F <: Function}

    lt = map(_to_loss_spec, loss_types)
    tl = _to_loss_spec(training_loss)
    el = _to_extra_loss_spec(extra_loss)

    return LoggingLoss{typeof(tl), typeof(el), F, train_mode}(lt, tl, el, agg, train_mode)
end


_to_loss_spec(s::Symbol) = SymbolicLoss{s}(s)
_to_loss_spec(f::Function) = _accepts_params(f) ? ParamLoss(f) : FunctionLoss(f)
_to_loss_spec(ls::LossSpec) = ls

"""
    _accepts_params(f) -> Bool

Auto-detect whether a custom loss `f` uses the full-context signature
`f(ŷ, y, y_nan, ps, targets, parameters)` (6 positional args) instead of the
classic masked signature `f(ŷ_masked, y_masked)` (2 positional args). Returns
`true` only when `f` has a 6-argument method and no 2-argument method, so existing
2-arg losses keep their behavior. Functions that are ambiguous (e.g. varargs
matching both arities) fall back to the classic 2-arg path.
"""
_accepts_params(f) = hasmethod(f, NTuple{6, Any}) && !hasmethod(f, NTuple{2, Any})

_to_loss_spec(t::Tuple{<:Function, <:Tuple}) = ParameterizedLoss(t[1], t[2])
_to_loss_spec(t::Tuple{<:Function, <:NamedTuple}) = ParameterizedLoss(t[1], (), t[2])
_to_loss_spec(t::Tuple{<:Function, <:Tuple, <:NamedTuple}) = ParameterizedLoss(t[1], t[2], t[3])
_to_loss_spec(x) = throw(ArgumentError("Invalid loss specification: $x"))
_to_loss_spec(pt::Tuple) = PerTarget(map(_to_loss_spec, pt))

_to_extra_loss_spec(::Nothing) = ExtraLoss(nothing)
_to_extra_loss_spec(f::Function) = ExtraLoss(f)
_to_extra_loss_spec(el::ExtraLoss) = el

# unwrapping / getters
loss_name(ls::SymbolicLoss) = ls.name
loss_name(::LossSpec) = nothing

loss_spec(ls::SymbolicLoss) = ls.name
loss_spec(ls::FunctionLoss) = ls.f
loss_spec(ls::ParameterizedLoss) =
    isempty(ls.args) && isempty(ls.kwargs) ? ls.f :
    isempty(ls.kwargs) ? (ls.f, ls.args) :
    isempty(ls.args) ? (ls.f, ls.kwargs) :
    (ls.f, ls.args, ls.kwargs)

loss_spec(el::ExtraLoss) = el.f
loss_spec(pl::ParamLoss) = pl.f
loss_spec(pt::PerTarget) = PerTarget(map(loss_spec, pt.losses))

loss_types(logging::LoggingLoss) = map(loss_spec, logging.loss_types)
training_loss(logging::LoggingLoss) = loss_spec(logging.training_loss)
extra_loss(logging::LoggingLoss) = loss_spec(logging.extra_loss)
