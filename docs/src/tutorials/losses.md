# Losses and LoggingLoss

Concise guide to EasyHybrid loss primitives: the low-level `loss_fn` API and the `LoggingLoss` helper that aggregates per-target loss specifications for training and evaluation. Examples progress from simple to advanced (uncertainty-aware).

## Core concepts

- `loss_fn(ŷ, y, y_nan, loss_spec)` — low-level loss API. `loss_spec` may be:
  - `Val(:mse)`, `Val(:rmse)`, `Val(:mae)`, `Val(:pearson)`, `Val(:r2)`, etc. — predefined metrics.
  - `f :: Function` — custom `f(ŷ_masked, y_masked)` (where `y_masked` is a vector or a `(y, σ)` tuple).
  - `(f, (args...))` — positional args forwarded as `f(ŷ, y, args...)`.
  - `(f, NamedTuple(...))` — kwargs forwarded as `f(ŷ, y; kwargs...)`.
  - `(f, (args...), NamedTuple(...))` — both.
- `compute_loss(...)` — convenience over targets: aggregates per-target calls to `loss_fn`.
- `LoggingLoss` — structure with:
  - `loss_types` (Vector of loss specs for evaluation),
  - `training_loss` (loss spec used during training),
  - `agg` (aggregation function across targets, e.g. `sum`, `mean`),
  - `train_mode` (toggle training vs eval behavior).

Notes:
- `y_nan` is a boolean mask (or function returning a mask per target) used to ignore missing values.
- For uncertainty-aware losses, pass target values as `(y_vals, y_sigma)` and write custom losses to accept that tuple.

---

## 1 — Simple usage

Predefined metrics and a basic `LoggingLoss`:

```julia
ŷ = Dict(:t1 => [1.0, 2.0], :t2 => [0.5, 1.0])
y(t) = t == :t1 ? [1.1, 1.9] : [0.4, 1.1]
y_nan(t) = trues(2)
targets = [:t1, :t2]

# total MSE across targets
mse_total = compute_loss(ŷ, y, y_nan, targets, :mse, sum)

# multiple metrics in a NamedTuple
losses = compute_loss(ŷ, y, y_nan, targets, [:mse, :mae], sum)
```

Create a `LoggingLoss`:

```julia
logging = LoggingLoss() # defaults
```

```julia
logging = LoggingLoss(loss_types=[:mse, :mae], training_loss=:mse, agg=sum, train_mode=true)
```

In training use `logging.training_loss`, in evaluation use `logging.loss_types`.

---

## 2 — Intermediate: custom functions, args, kwargs

Custom losses receive masked predictions and masked targets:

```julia
custom_loss(ŷ, y) = mean(abs2, ŷ .- y)
weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
scaled_loss(ŷ, y; scale=1.0) = scale * mean(abs2, ŷ .- y)
```

Use variants:

```julia
compute_loss(ŷ, y, y_nan, targets, custom_loss, sum)
compute_loss(ŷ, y, y_nan, targets, (weighted_loss, (0.5,)), sum)
compute_loss(ŷ, y, y_nan, targets, (scaled_loss, (scale=2.0,)), sum)
compute_loss(ŷ, y, y_nan, targets, (weighted_loss, (0.5,), (scale=2.0,)), sum)
```

`LoggingLoss` can mix symbols and functions:

```julia
logging = LoggingLoss(
    loss_types = [:mse, custom_loss, (weighted_loss, (0.5,)), (scaled_loss, (scale=2.0,))],
    training_loss = custom_loss,
    agg = mean,
    train_mode = false
)
```

## 3 — Advanced: uncertainty-aware losses

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
loss = compute_loss(ŷ, (y, y_sigma), y_nan, targets, custom_loss_uncertainty, sum)
```

Behavior:
- `compute_loss` packs per-target `(y_vals_target, σ_target)` tuples and forwards them to `loss_fn`.
<!-- TODO -->
- Predefined metrics use only `y_vals` when a `(y, σ)` tuple is supplied. 

## Tips and quick reference

- Prefer `f(ŷ_masked, y_masked)` for custom losses; `y_masked` may be a vector or `(y, σ)`.
- Use `Val(:metric)` only for predefined `loss_fn` variants.
- Quick calls:
  - Predefined: `compute_loss(..., :mse, sum)`
  - Custom: `compute_loss(..., custom_loss, sum)`
  - Args: `compute_loss(..., (f, (arg1,arg2)), sum)`
  - Kwargs: `compute_loss(..., (f, (kw=val,)), sum)`
  - Uncertainty: `compute_loss(..., (y, y_sigma), ..., custom_loss_uncertainty, sum)`