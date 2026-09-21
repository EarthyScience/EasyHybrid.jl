# Lux.jl, ComponentArrays & Model Rules

## Functional Architecture (Lux.jl)
- EasyHybrid.jl uses [Lux.jl](https://github.com/LuxDL/Lux.jl) for neural network components.
- Respect the strict separation between parameters (`ps`) and states (`st`): `y, st = model(x, ps, st)`.
- Never store mutable state or parameters inside layer structs.
- Ensure all custom layers implement standard Lux interfaces (`LuxCore.AbstractExplicitLayer`, `LuxCore.initialparameters`, `LuxCore.initialstates`).

## ComponentArrays & Parameter Handling
- Parameters in EasyHybrid models are structured as `ComponentArray`s to allow named indexing (`ps.nn`, `ps.mechanistic`, etc.) while maintaining seamless vector representation for optimizers (`Optimization.jl`).
- When manipulating parameters, ensure shape and key consistency.

## Automatic Differentiation & Pure Functions
- Loss functions, model forwards, and mechanistic evaluation must be pure and differentiable with `Zygote.jl` and `ForwardDiff.jl`.
- Avoid mutating arrays in-place (`x[i] = ...`) inside functions that participate in AD backward passes. Use array transformations, comprehensions, or `ChainRulesCore` custom rrules when necessary.

## Dimensional Data & AxisKeys
- When handling multidimensional time-series or spatio-temporal datasets, respect `DimArray` (DimensionalData.jl) and `KeyedArray` (AxisKeys.jl) metadata and dimension orders.
