## Losses and LoggingLoss

```@example loss
using EasyHybrid
using EasyHybrid: _compute_loss
```

````@docs; canonical=false
EasyHybrid._compute_loss
````

::: warning

- `y_nan` is a boolean mask (or function returning a mask per target) used to ignore missing values.
- For uncertainty-aware losses, pass target values as `(y_vals, y_sigma)` and write custom losses to accept that tuple.

:::

::: tip Tips and quick reference

- Prefer `f(ŷ_masked, y_masked)` for custom losses; `y_masked` may be a vector or `(y, σ)`.
- Use `Val(:metric)` only for predefined `loss_fn` variants.
- Quick calls:
    - `_compute_loss(..., :mse, sum)`: predefined
    - `_compute_loss(..., custom_loss, sum)` : custom loss
    - `_compute_loss(..., (f, (arg1, arg2, )), sum)`: additional arguments 
    - `_compute_loss(..., (f, (kw=val,)), sum)`: with keyword arguments
    - `_compute_loss(..., (f, (arg1, ), (kw=val,)), sum)`: with additional arguments and keyword arguments
    - `_compute_loss(..., (y, y_sigma), ..., custom_loss_uncertainty, sum)`: with uncertainties

:::


### Simple usage

Predefined metrics

```@example loss
# synthetic data
ŷ = Dict(:t1 => [1.0, 2.0], :t2 => [0.5, 1.0])
y(t) = t == :t1 ? [1.1, 1.9] : [0.4, 1.1]
y_nan(t) = trues(2)
targets = [:t1, :t2]
```

```@ansi loss
mse_total = _compute_loss(ŷ, y, y_nan, targets, :mse, sum) # total MSE across targets
losses = _compute_loss(ŷ, y, y_nan, targets, [:mse, :mae], sum) # multiple metrics in a NamedTuple
```

### Custom functions, args, kwargs

Custom losses receive masked predictions and masked targets:

```@example loss
custom_loss(ŷ, y) = mean(abs2, ŷ .- y)
weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
scaled_loss(ŷ, y; scale=1.0) = scale * mean(abs2, ŷ .- y)
complex_loss(ŷ, y, w; scale=1.0) = scale * w * mean(abs2, ŷ .- y);
nothing # hide
```

Use variants:

```@ansi loss
_compute_loss(ŷ, y, y_nan, targets, custom_loss, sum)
_compute_loss(ŷ, y, y_nan, targets, (weighted_loss, (0.5,)), sum)
_compute_loss(ŷ, y, y_nan, targets, (scaled_loss, (scale=2.0,)), sum)
_compute_loss(ŷ, y, y_nan, targets, (complex_loss, (0.5,), (scale=2.0,)), sum)
```

### Uncertainty-aware losses

::: warning 

This will be supported soon!

:::

Signal uncertainty by providing targets as `(y_vals, y_sigma)` and write the loss to accept that tuple:

```julia
function custom_loss_uncertainty(ŷ, y_and_sigma)
    y_vals, σ = y_and_sigma
    return mean(((ŷ .- y_vals).^2) ./ (σ .^2 .+ 1e-6))
end
```

Top-level usage (both `y` and `y_sigma` can be functions or containers):

```julia
y_sigma(t) = t == :t1 ? [0.1, 0.2] : [0.2, 0.1]
loss = _compute_loss(ŷ, (y, y_sigma), y_nan, targets,
    custom_loss_uncertainty, sum)
```

::: info Behavior

- `_compute_loss` packs per-target `(y_vals_target, σ_target)` tuples and forwards them to `loss_fn`.
- Predefined metrics use only `y_vals` when a `(y, σ)` tuple is supplied. (TODO)

:::


## LoggingLoss

The `LoggingLoss` helper aggregates per-target loss specifications for training and evaluation.

````@docs; canonical=false
LoggingLoss
````

Internally, in training we use `logging.training_loss` and in evaluation `logging.loss_types`.
Note that `LoggingLoss` can mix symbols and functions.

## Loss → train

So, how do you specified your loss? and the additional metrics given by `loss_types`?

### default losses

You could select a different training or and a different vector for additional metrics

```julia
train(...;
    training_loss = :mae,
    loss_types = [:mse, :mae, :nse]
    )
```

### without additional arguments

Define your own custom function `fn(ŷ, y)` as above and pass it to the corresponding keyword argument:

```julia
train(...;
    training_loss = fn,
    loss_types = [fn, :mae, :nse]
    )
```

### with additional arguments

now your function will have additional arguments, i.e. `fn_args(ŷ, y, args...)`:

```julia
train(...;
    training_loss = (fn_args, (args...,)),
    loss_types = [(fn_args, (args...,)), :mae, :nse]
    )
```

and possible keyword arguments, i.e. `fn_args(ŷ, y, args...; kwargs...)`:

```julia
train(...;
    training_loss = (fn_args, (args...,), (kwargs...,)),
    loss_types = [(fn_args, (args...,), (kwargs...,)), :mae, :nse]
    )
```

### full-context losses (access to the whole `ŷ`, masks, `ps` and `parameters`)

A 'standard' custom loss `f(ŷ, y)` only receives *one target's masked* predictions
and observations. When the loss needs more — the full prediction NamedTuple, the
NaN masks, the raw parameters, or the model parameters — pass a function with the
6-argument signature `f(ŷ, y, y_nan, ps, targets, parameters)`. It is
**auto-detected** (no wrapper needed) and called with the *full, unmasked* `ŷ` and
`y`, the masks `y_nan`, the raw parameters `ps`, the target names, and
`parameters` (i.e. `ŷ.parameters`: every NN-predicted, global and fixed value).
You do the masking and aggregation yourself and return a scalar.

The typical use case is a Gaussian negative log-likelihood with a *learned* noise
scale `σ`. You do **not** need the mechanistic model to know about `σ`: just
declare it as a parameter and read `parameters.sigma` in the loss (the mechanistic
model only receives the kwargs it declares, so a loss-only `σ` is skipped there):

```julia
function gaussian_nll(ŷ, y, y_nan, ps, targets, parameters)
    total = zero(eltype(ŷ.reco))
    for t in targets
        m = y_nan[t]
        r = ŷ[t][m] .- y[t][m]
        σ = length(parameters.sigma) == 1 ? parameters.sigma[1] : parameters.sigma[m]
        total += sum(@. 0.5f0 * (r / σ)^2 + log(σ))
    end
    return total
end

train(...; training_loss = gaussian_nll)
```

The same loss works whether `σ` is declared as a global parameter (one value per
target) or an NN-predicted parameter (one value per observation) — only the
model construction changes. See the synthetic respiration tutorial for both.

::: warning

- The 6-argument function is detected only when it has no 2-argument method;
  classic `f(ŷ, y)` losses are unaffected.
- Full-context losses are only used for `training_loss`. Entries in `loss_types`
  (logging/metrics) still use the masked `f(ŷ_masked, y_masked)` form.
- Because the loss needs the *full* `ŷ`/`parameters`, it cannot use the bare
  2-argument `f(ŷ, y)` signature (that one is per-target and masked). Use the
  6-argument form and simply ignore the arguments you don't need.

:::