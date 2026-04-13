module EasyHybridGPUArraysExt
using GPUArrays: AbstractGPUArray
using DimensionalData: AbstractDimArray, modify
using AxisKeys: KeyedArray, axiskeys
function Base.show(io::IO, mime::MIME"text/plain", A::AbstractDimArray{T,N,D,<:AbstractGPUArray}) where {T,N,D<:Tuple}
    backend = nameof(typeof(parent(A)))
    println(io, "GPU-backed ($backend):")
    return show(io, mime, modify(Array, A))
end

function Base.show(io::IO, mime::MIME"text/plain", A::KeyedArray{T,N,<:AbstractGPUArray}) where {T,N}
    backend = nameof(typeof(parent(A)))
    println(io, "GPU-backed ($backend):")
    return show(io, mime, KeyedArray(Array(parent(A)), axiskeys(A)))
end

end