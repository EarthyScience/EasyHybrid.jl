export WrappedTuples

"""
    WrappedTuples(vec::Vector{<:NamedTuple})

Wraps a vector of named tuples to allow dot-access to each field as a vector.
"""
struct WrappedTuples{T<:AbstractVector{<:NamedTuple}} <: AbstractVector{NamedTuple}
    data::T
end

# Required methods for AbstractVector
Base.size(w::WrappedTuples) = (length(w.data), length(first(w.data)))
Base.getindex(w::WrappedTuples, i::Int) = w.data[i]
Base.getindex(w::WrappedTuples, r::AbstractRange) = WrappedTuples(w.data[r])
Base.IndexStyle(::Type{<:WrappedTuples}) = IndexLinear()
Base.length(w::WrappedTuples) = length(w.data)
Base.iterate(w::WrappedTuples, state=1) = state > length(w.data) ? nothing : (w.data[state], state+1)

# Dot-access to fields (e.g., wrapped.a)
function Base.getproperty(w::WrappedTuples, field::Symbol)
    if field === :data  # Avoid recursion on the internal field
        return getfield(w, :data)
    end
    # Extract field across all tuples
    return getfield.(w.data, field)
end

Base.keys(w::WrappedTuples) = propertynames(first(w.data))

# Enable tab-completion and introspection
function Base.propertynames(w::WrappedTuples, private::Bool=false)
    return (:data, ) ∪ propertynames(first(w.data), private)
end
function Base.Matrix(w::WrappedTuples)
    n, m = size(w)
    fields = propertynames(first(w.data))
    T = promote_type(map(f -> eltype(getproperty(w, f)), fields)...)
    mat = Array{T}(undef, n, m)
    for (j, f) in enumerate(fields)
        mat[:, j] .= getproperty(w, f)
    end
    return mat
end