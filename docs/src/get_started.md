# EasyHybrid.jl

Simply do

```@example modelName
using EasyHybrid
@hybrid YourHybridModelName α
# α is the newly introduced physical parameter to be learned!
```

see your new model by typing `?YourHybridModelName`.

::: details you should see something like:

```@example modelName
@doc YourHybridModelName
```
:::


::: details What is `@habrid` doing?

Defines your model as a sub-type of an `LuxCore.AbstractLuxContainerLayer`. Namely,

```julia
struct YourHybridModelName{D, T1, T2, T3, T4} <: LuxCore.AbstractLuxContainerLayer{(:NN, :predictors, :forcing, :targets, :α)}
  NN
  predictors
  forcing
  targets
  α
  function YourHybridModelName(NN::D, predictors::T1, forcing::T2, targets::T3, α::T4) where {D, T1, T2, T3, T4}
      new{D, T1, T2, T3, T4}(NN, collect(predictors), collect(targets), collect(forcing), [α])
  end
end
```

sets initial parameters and states

::: code-group

```julia [initial parameters]
function LuxCore.initialparameters(::AbstractRNG, layer::YourHybridModelName)
  ps, _ = LuxCore.setup(Random.default_rng(), layer.NN)
  return (; ps, α = layer.α,) # these parameters are trainable! [!code warning]
end
```

```julia [initial states]
function LuxCore.initialstates(::AbstractRNG, layer::RespirationRbQ10)
  _, st = LuxCore.setup(Random.default_rng(), layer.NN)
  return (; st) # none of the possible additional arguments/variables here are trainable! [!code warning]
end
```

:::

````@docs
@hybrid
````

Hence, after specifying the `physical parameters`, you only need to describe how this model operates.

## Definition
Model definition, how does the actual model operates!

```julia
function (hm::YourHybridModelName)(dataset_keyed, ps, st)
  # data selection
  p = dataset_keyed(hm.predictors)
  x = Array(dataset_keyed(hm.forcing))
  
  # output from Neural network application 
  β, st = LuxCore.apply(hm.NN, p, ps.ps, st.st) # [!code highlight]

  # equation! this is where you should focus, your model!
  ŷ = β .* ps.α .^(0.1f0 * (x .- 15.0f0)) # [!code warning]

  return (; ŷ), (; β, st) # always output predictions and states as two tuples
end
```

## Loss function

We provide a generic loss function, if you need further adjustments then define a specific one for your hybrid model.

::: details define a custom loss

What is your approach for optimization?

```julia
function lossfn(HM::YourHybridModelName, ds_p, (ds_t, ds_t_nan), ps, st)
  targets = HM.targets
  ŷ, _ = HM(ds_p, ps, st)
  y = ds_t(targets)
  y_nan = ds_t_nan(targets)

  loss = 0.0
  for k in targets
      loss += mean(abs2, (ŷ[k][y_nan(k)] .- y(k)[y_nan(k)]))
  end
  return loss
end
```
:::