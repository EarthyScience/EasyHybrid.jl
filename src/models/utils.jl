export scale_single_param, default, lower, upper, hard_sigmoid, inv_hard_sigmoid, inv_sigmoid, scale_single_param_minmax

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

"""
    scale_single_param(name, raw_val, parameters)

Scale a single parameter using the sigmoid scaling function.
"""
function scale_single_param(name, raw_val, hm::ParameterContainer)
    ℓ = lower(hm)[name]
    u = upper(hm)[name]
    return ℓ .+ (u .- ℓ) .* sigmoid.(raw_val)
end

inv_sigmoid(y) = log.(y ./ (1 .- y))

""" 
    scale_single_param_minmax(name, hm::AbstractHybridModel)

Scale a single parameter using the minmax scaling function.
"""
function scale_single_param_minmax(name, hm::ParameterContainer)
    ℓ = lower(hm)[name]
    u = upper(hm)[name]
    return inv_sigmoid.((default(hm)[name] .- ℓ) ./ (u .- ℓ))
end
