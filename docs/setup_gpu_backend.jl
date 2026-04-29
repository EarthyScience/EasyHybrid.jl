# Load the appropriate GPU backend for `MLDataDevices`, independent of machine.
#
# Tries the common GPU trigger packages in order:
#   - LuxCUDA  -> CUDADevice   (NVIDIA)
#   - AMDGPU   -> AMDGPUDevice (AMD ROCm)
#   - Metal    -> MetalDevice  (Apple Silicon)
#   - oneAPI   -> oneAPIDevice (Intel)
#
# The first backend that is both installed AND functional on this machine
# is kept. If none are available, we silently fall back to CPU and
# `gpu_device()` will return a `CPUDevice`.
#
# After `include`ing this file, the following are defined in the caller's
# namespace:
#   - `GPU_BACKEND_PKG`  : `Symbol` of the loaded package (or `nothing`)
#   - `GPU_DEVICE_TYPE`  : the `MLDataDevices` device type (e.g. `CUDADevice`,
#                          or `CPUDevice` as fallback)

using MLDataDevices

const _GPU_TRIGGERS = [
    (:LuxCUDA, :CUDADevice),
    (:AMDGPU, :AMDGPUDevice),
    (:Metal, :MetalDevice),
    (:oneAPI, :oneAPIDevice),
]

function _load_gpu_backend()
    for (pkg, dev) in _GPU_TRIGGERS
        try
            @eval Main using $pkg
        catch err
            @debug "Skipping $pkg (not installed / failed to load)" exception = err
            continue
        end

        # `using $pkg` may load an `MLDataDevices` package extension that
        # adds new methods (e.g. a real `functional(::Type{CUDADevice})`).
        # Those methods live in a newer world age than this function, so
        # we must dispatch via `invokelatest` or we'll get the stale
        # (pre-extension) method that reports `false`.
        DeviceT = Base.invokelatest(getfield, MLDataDevices, dev)
        is_functional = Base.invokelatest(MLDataDevices.functional, DeviceT)

        if is_functional
            @info "GPU backend loaded: $pkg → $dev"
            return pkg, DeviceT
        else
            @warn "$pkg loaded but $dev is not functional on this machine; trying next backend."
        end
    end

    @info "No functional GPU backend found; falling back to CPU."
    return nothing, CPUDevice
end

const GPU_BACKEND_PKG, GPU_DEVICE_TYPE = _load_gpu_backend()
