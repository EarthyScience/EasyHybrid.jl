export scale_single_param, default, lower, upper, hard_sigmoid, inv_hard_sigmoid, inv_sigmoid, scale_single_param_minmax, scaletype

# Define the hard sigmoid activation function
function hard_sigmoid(x)
    return clamp.(0.2 .* x .+ 0.5, 0.0, 1.0)
end

# Inverse of `hard_sigmoid` on the linear region (0, 1).
# Saturated inputs (y ≤ 0 or y ≥ 1) are extrapolated linearly since the
# clamp makes the forward map non-invertible there.
function inv_hard_sigmoid(y)
    return (y .- 0.5) ./ 0.2
end

function default(p::ParameterContainer)
    return p.table[:, :default]
end

function lower(p::ParameterContainer)
    return p.table[:, :lower]
end

function upper(p::ParameterContainer)
    return p.table[:, :upper]
end

pnames(p::ParameterContainer) = keys(p.table.axes[1])

inv_sigmoid(y) = log.(y ./ (1 .- y))

# ---------------------------------------------------------------------------
# Parameter scaling / warps
#
# Every optimizable parameter is interpolated between its bounds `[ℓ, u]` in a
# warped space defined by a monotone function `g` (with inverse `g⁻¹`):
#
#     value = g⁻¹( g(ℓ) + (g(u) − g(ℓ)) · sigmoid(raw) )       # forward
#     raw   = logit( (g(d) − g(ℓ)) / (g(u) − g(ℓ)) )           # inverse (init)
#
# Warps (selected per-parameter via the optional 4th tuple element):
#   :linear  g = identity        default; uniform resolution in value
#   :log     g = log, g⁻¹ = exp  uniform in log(value); needs ℓ > 0
#   :logit   g = logit, g⁻¹ = σ  uniform in log-odds; needs 0 < ℓ < u < 1
#
# `_scale`/`_unscale` dispatch on `Val(scale)`, so each warp compiles to a
# type-stable method (constructing the `Val` is the single dynamic point — a
# function barrier). Adding a scale is one `_scale`/`_unscale` method pair.
# ---------------------------------------------------------------------------

_scale(::Val{:linear}, ℓ, u, raw) = ℓ .+ (u .- ℓ) .* sigmoid.(raw)
_scale(::Val{:log}, ℓ, u, raw) = exp.(log(ℓ) .+ (log(u) - log(ℓ)) .* sigmoid.(raw))
function _scale(::Val{:logit}, ℓ, u, raw)
    L = inv_sigmoid(ℓ)
    U = inv_sigmoid(u)
    return sigmoid.(L .+ (U - L) .* sigmoid.(raw))
end

_unscale(::Val{:linear}, ℓ, u, d) = inv_sigmoid.((d .- ℓ) ./ (u .- ℓ))
_unscale(::Val{:log}, ℓ, u, d) = inv_sigmoid.((log.(d) .- log(ℓ)) ./ (log(u) - log(ℓ)))
function _unscale(::Val{:logit}, ℓ, u, d)
    L = inv_sigmoid(ℓ)
    U = inv_sigmoid(u)
    return inv_sigmoid.((inv_sigmoid.(d) .- L) ./ (U - L))
end

"""
    scaletype(hm::ParameterContainer, name)

Return the scaling warp (`:linear`, `:log`, or `:logit`) selected for parameter `name`.
"""
scaletype(hm::ParameterContainer, name) = hm.scales[name]

"""
    scale_single_param(name, raw_val, hm::ParameterContainer)

Map an unconstrained `raw_val` into the parameter's bounds `[ℓ, u]`, using the
warp selected for `name` (`:linear`, `:log`, or `:logit`).
"""
function scale_single_param(name, raw_val, hm::ParameterContainer)
    return _scale(Val(scaletype(hm, name)), lower(hm)[name], upper(hm)[name], raw_val)
end

""" 
    scale_single_param_minmax(name, hm::ParameterContainer)

Inverse of [`scale_single_param`](@ref): map a parameter's default value back to
the unconstrained space used at initialization, using its warp.
"""
function scale_single_param_minmax(name, hm::ParameterContainer)
    return _unscale(Val(scaletype(hm, name)), lower(hm)[name], upper(hm)[name], default(hm)[name])
end
