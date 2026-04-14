## GPU Acceleration

GPU training is supported, but you must install the backend package for your hardware first.

:::code-group

```julia [NVIDIA GPUs]
using Pkg
Pkg.add("LuxCUDA")
# or
Pkg.add(["CUDA", "cuDNN"])
```

```julia [AMD ROCm GPUs]
using Pkg
Pkg.add("AMDGPU")
# If your use case fails, consider Reactant.jl for GPU support.
```

```julia [Metal M-Series GPUs]
using Pkg
Pkg.add("Metal")
```

```julia [Intel GPUs]
using Pkg
Pkg.add("oneAPI")
```

:::

Then run the following to access a device:

:::code-group

```julia [NVIDIA GPUs]
using Lux, LuxCUDA
gpu_device()
```

```julia [AMD ROCm GPUs]
using Lux, AMDGPU
gpu_device()
```

```julia [Metal M-Series GPUs]
using Lux, Metal
gpu_device()
```

```julia [Intel GPUs]
using Lux, oneAPI
gpu_device()
```

:::

In your training call, pass `arch = GPU()`. For example:

```julia
using EasyHybrid, Metal

train(...; arch = GPU())
```

That is all you need. Your hybrid model will now train on the GPU.