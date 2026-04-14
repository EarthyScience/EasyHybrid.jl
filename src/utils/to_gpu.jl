export to_gpu
function to_gpu(A::AbstractDimArray)
    dev = gpu_device()
    return modify(dev, A)
end

function to_gpu(A::KeyedArray)
    dev = gpu_device()
    return KeyedArray(dev(parent(A)), axiskeys(A))
end