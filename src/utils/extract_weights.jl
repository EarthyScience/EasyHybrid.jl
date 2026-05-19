export extract_weights, weight_l2
using ComponentArrays

"""
    _is_named_leaf(name, child, key) -> Bool

Return `true` when `child` is an array leaf whose parent field is `key`
(e.g. a `Dense` layer's `weight` matrix).
"""
function _is_named_leaf(name::Symbol, child, key::Symbol)
    return name === key && child isa AbstractArray && isempty(propertynames(child))
end

"""
    extract_weights(ps; key=:weight) -> Vector{AbstractArray}

Walk the parameter tree `ps` (a `ComponentArray`, `NamedTuple`, or any nested
combination of them) and return all leaf arrays whose **immediate parent field
name** equals `key`.

Defaults to `:weight`, so you get the weight matrices of `Dense`/`Conv` layers
and skip biases, `BatchNorm` `scale`/`bias`, running statistics in `st`, and
any scalar global parameters.

The returned arrays are views/aliases into `ps`. When `ps` is the argument the
autodiff is differentiating w.r.t., gradients of any function of these views
flow back into the trainable weights.
"""
function extract_weights(ps; key::Symbol = :weight)
    out = AbstractArray[]
    _collect!(out, ps, key)
    return out
end

_collect!(_, _, ::Symbol) = nothing

function _collect!(out, node::Union{NamedTuple, ComponentArray}, key::Symbol)
    for name in propertynames(node)
        child = getproperty(node, name)
        if _is_named_leaf(name, child, key)
            push!(out, child)
        else
            _collect!(out, child, key)
        end
    end
    return nothing
end

"""
    weight_l2(ps; key=:weight) -> Real

Sum of squared Frobenius norms over all parameter arrays in `ps` whose
immediate parent field name is `key` (default `:weight`).

Unlike `sum(abs2, extract_weights(ps))`, this fuses the tree walk with the
reduction so it is safe to use inside Zygote-differentiated losses, e.g.:

```julia
extra_loss = (ŷ, ps) -> (; l2_Rb = λ * weight_l2(ps.Rb),)
```

When `ps` is the loss function argument, gradients flow into the weight arrays.
"""
function weight_l2(ps; key::Symbol = :weight)
    return _weight_l2(ps, key)
end

_weight_l2(::Any, ::Symbol) = 0f0

function _weight_l2(node::Union{NamedTuple, ComponentArray}, key::Symbol)
    s = 0f0
    for name in propertynames(node)
        child = getproperty(node, name)
        if _is_named_leaf(name, child, key)
            s = s + sum(abs2, child)
        else
            s = s + _weight_l2(child, key)
        end
    end
    return s
end
