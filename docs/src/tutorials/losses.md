## Losses and LoggingLoss

```@example loss
using EasyHybrid
using EasyHybrid: compute_loss
```

````@docs; canonical=false
EasyHybrid.compute_loss
````

::: warning

- `y_nan` is a boolean mask (or function returning a mask per target) used to ignore missing values.
- For uncertainty-aware losses, pass target values as `(y_vals, y_sigma)` and write custom losses to accept that tuple.

:::

### Simple usage

Predefined metrics

```@example loss
# toy data
ŷ = Dict(:t1 => [1.0, 2.0], :t2 => [0.5, 1.0])
y(t) = t == :t1 ? [1.1, 1.9] : [0.4, 1.1]
y_nan(t) = trues(2)
targets = [:t1, :t2]
```

```@ansi loss
# total MSE across targets
mse_total = compute_loss(ŷ, y, y_nan, targets, :mse, sum)

# multiple metrics in a NamedTuple
losses = compute_loss(ŷ, y, y_nan, targets, [:mse, :mae], sum)
```

### Intermediate: custom functions, args, kwargs

Custom losses receive masked predictions and masked targets:

```@example loss
custom_loss(ŷ, y) = mean(abs2, ŷ .- y)
weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
scaled_loss(ŷ, y; scale=1.0) = scale * mean(abs2, ŷ .- y)
complex_loss(ŷ, y, w; scale=1.0) = scale * w * mean(abs2, ŷ .- y);
```

Use variants:

```@ansi loss
compute_loss(ŷ, y, y_nan, targets, custom_loss, sum)
compute_loss(ŷ, y, y_nan, targets, (weighted_loss, (0.5,)), sum)
compute_loss(ŷ, y, y_nan, targets, (scaled_loss, (scale=2.0,)), sum)
compute_loss(ŷ, y, y_nan, targets, (complex_loss, (0.5,), (scale=2.0,)), sum)
```

### Advanced: uncertainty-aware losses

Signal uncertainty by providing targets as `(y_vals, y_sigma)` and write the loss to accept that tuple:

```@example loss
function custom_loss_uncertainty(ŷ, y_and_sigma)
    y_vals, σ = y_and_sigma
    return mean(((ŷ .- y_vals).^2) ./ (σ .^2 .+ 1e-6))
end
```

Top-level usage (both `y` and `y_sigma` can be functions or containers):

```@example loss
y_sigma(t) = t == :t1 ? [0.1, 0.2] : [0.2, 0.1]
loss = compute_loss(ŷ, (y, y_sigma), y_nan, targets,
    custom_loss_uncertainty, sum)
```

::: info Behavior

- `compute_loss` packs per-target `(y_vals_target, σ_target)` tuples and forwards them to `loss_fn`.
- Predefined metrics use only `y_vals` when a `(y, σ)` tuple is supplied. (TODO)

:::


::: tip Tips and quick reference

- Prefer `f(ŷ_masked, y_masked)` for custom losses; `y_masked` may be a vector or `(y, σ)`.
- Use `Val(:metric)` only for predefined `loss_fn` variants.
- Quick calls:
  - Predefined: `compute_loss(..., :mse, sum)`
  - Custom: `compute_loss(..., custom_loss, sum)`
  - Args: `compute_loss(..., (f, (arg1,arg2)), sum)`
  - Kwargs: `compute_loss(..., (f, (kw=val,)), sum)`
  - Uncertainty: `compute_loss(..., (y, y_sigma), ..., custom_loss_uncertainty, sum)`

:::

## LoggingLoss

The `LoggingLoss` helper aggregates per-target loss specifications for training and evaluation.

````@docs; canonical=false
LoggingLoss
````

Internally, in training we use `logging.training_loss` and in evaluation `logging.loss_types`.
Note that `LoggingLoss` can mix symbols and functions.