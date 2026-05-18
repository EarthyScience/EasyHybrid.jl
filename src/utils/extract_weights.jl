export extract_weights
using ComponentArrays

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
        if name === key && child isa AbstractArray && isempty(propertynames(child))
            push!(out, child)
        else
            _collect!(out, child, key)
        end
    end
    return nothing
end