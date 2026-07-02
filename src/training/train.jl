export train

"""
    is_optimisers_rule(opt) -> Bool

Return `true` when `opt` originates from the `Optimisers.jl` package (e.g.
`Adam`, `AdamW`, `RMSProp`, `OptimiserChain`), **or** when `opt` is a
`NamedTuple` describing a per-branch optimizer (see [`is_per_branch_opt`](@ref)).

The check on single rules is by source package
(`nameof(parentmodule(typeof(opt))) === :Optimisers`) rather than by
`isa Optimisers.AbstractRule`, because in some package combinations
`Optim.jl` optimizers were observed to satisfy the `AbstractRule` test and
get misrouted to the `Lux.Training` loop.

The `Lux.Training`-based loop dispatches on this; everything else (including
`Optim.jl` and `Optimization.jl` optimizers) is routed through the
`Optimization.jl` driver.
"""
is_optimisers_rule(opt) =
    nameof(parentmodule(typeof(opt))) === :Optimisers ||
    is_per_branch_opt(opt)

"""
    is_per_branch_opt(opt) -> Bool

Return `true` when `opt` is a `NamedTuple` describing a per-branch
optimizer specification — i.e. one of:

  - a `NamedTuple` of `Optimisers.AbstractRule`s (e.g.
    `(; Rb = Adam(1e-3), Q10 = Descent(1e-2))`), or
  - a `NamedTuple` of pre-built optimizer state trees as returned by
    `Optimisers.setup(rule, ps_branch)`, or
  - a mix of the two.

Branches of the parameter tree not listed in `opt` fall back to the default
rule `Adam()` (see [`build_opt_state`](@ref)).

This is detected purely by `opt isa NamedTuple`; the per-branch dispatch is
checked again per-leaf when [`build_opt_state`](@ref) walks the spec.
"""
is_per_branch_opt(opt) = opt isa NamedTuple

"""
    build_opt_state(opt, ps::NamedTuple; default_rule = Optimisers.Adam())

Build the optimizer state tree consumed by `Lux.Training.TrainState` /
`Optimisers.update!`. Three forms of `opt` are accepted:

  1. `opt::Optimisers.AbstractRule` — single rule applied to the whole
     parameter tree (delegates to `Optimisers.setup(opt, ps)`).
  2. `opt::NamedTuple` of `Optimisers.AbstractRule`s — each rule is wired
     to the matching top-level branch of `ps` via
     `Optimisers.setup(rule, ps[name])`. Branches missing from `opt` use
     `default_rule`.
  3. `opt::NamedTuple` of pre-built state trees (already returned by a
     prior `Optimisers.setup`) — used as-is. Form 2 and 3 can be mixed in
     the same `NamedTuple`.

The returned state tree has the same top-level keys as `ps`.

# Example

```julia
ps, _ = LuxCore.setup(rng, hybrid_model)   # (; Rb = NN_ps, RUE = NN_ps, Q10 = [v])

# Form 2 — preferred, lets the framework call `Optimisers.setup`:
opt_state = build_opt_state(
    (; Rb = Adam(1e-3), RUE = Adam(1e-3), Q10 = Descent(1e-2)),
    ps,
)
```
"""
function build_opt_state(opt::Optimisers.AbstractRule, ps; default_rule = Optimisers.Adam())
    return Optimisers.setup(opt, ps)
end

function build_opt_state(opt::NamedTuple, ps::NamedTuple; default_rule = Optimisers.Adam())
    pairs_ = map(collect(keys(ps))) do k
        if haskey(opt, k)
            spec = opt[k]
            return k => spec isa Optimisers.AbstractRule ?
                Optimisers.setup(spec, getproperty(ps, k)) :
                spec
        else
            return k => Optimisers.setup(default_rule, getproperty(ps, k))
        end
    end
    extra = setdiff(collect(keys(opt)), collect(keys(ps)))
    isempty(extra) ||
        @warn "Per-branch optimizer keys not found in parameter tree, ignored: $(extra)"
    return NamedTuple(pairs_)
end

function _train(model, data, train_cfg::TrainConfig, data_cfg::DataConfig)
    validate_config(train_cfg)
    ext = load_makie_extension(train_cfg)
    seed!(train_cfg.random_seed)

    ((x_train, forcings_train), y_train), ((x_val, forcings_val), y_val) = prepare_splits(data, model, data_cfg)
    mask_train, _ = valid_mask(y_train)
    mask_val, _ = valid_mask(y_val)
    loader = build_loader(x_train, forcings_train, y_train, mask_train, train_cfg)
    ps, st, train_state = init_model_state(model, train_cfg)

    init = compute_initial_state(model, x_train, forcings_train, y_train, mask_train, x_val, forcings_val, y_val, mask_val, ps, st, train_cfg)
    history = TrainingHistory(init)
    stopper = EarlyStopping(init.l_val, ps, st, train_cfg)
    paths = resolve_paths(train_cfg)
    prog = build_progress(train_cfg)

    # @show train_cfg.agg
    # @show train_cfg.training_loss
    # @show get_loss_value_t(history, train_cfg.training_loss, Symbol("$(train_cfg.agg)"))
    # @show get_loss_value_v(history, train_cfg.val_loss, Symbol("$(train_cfg.agg)"))

    dashboard = init_dashboard(ext, history, train_cfg, y_train, y_val, model.targets)

    save_initial_state!(paths, model, ps, st, train_cfg)
    ps = ps |> train_cfg.gdev
    st = st |> train_cfg.gdev
    train_state = train_state |> train_cfg.gdev
    record_or_run(ext, paths, train_cfg) do io
        for epoch in 1:train_cfg.nepochs
            ps, st, train_state = run_epoch!(loader, model, ps, st, train_state, train_cfg)
            snapshot = evaluate_epoch(model, x_train, forcings_train, y_train, mask_train, x_val, forcings_val, y_val, mask_val, ps, st, epoch, init, train_cfg)

            update!(stopper, history, snapshot, ps, st, train_cfg)
            # save_epoch!(paths, model, ps, st, snapshot, train_cfg)
            update_dashboard!(dashboard, ext, history, io, train_cfg)
            # log_progress!(prog, init, snapshot, train_cfg)

            is_done(stopper) && break
        end
    end

    # save_dashboard_img!(dashboard, ext, paths, train_cfg, stopper.best_epoch)
    ps, st = best_or_final(stopper, ps, st, train_cfg)
    # save_final!(paths, model, ps, st, x_train, forcings_train, y_train, x_val, forcings_val, y_val, stopper, train_cfg)

    return build_results(model, history, stopper, ps, st, x_train, forcings_train, y_train, x_val, forcings_val, y_val, train_cfg)
end

"""
    _train(model, data, train_cfg, data_cfg, solve_kwargs)

Dispatcher used by `train(...)`: routes to the original 4-arg `_train` body
(the `Lux.Training` / `Optimisers.jl` loop) when `train_cfg.opt isa
Optimisers.AbstractRule`, or to `_train_optimization` (which delegates batch
iteration to `Optimization.jl`) otherwise. `solve_kwargs` are forwarded to
`solve(...)` on the `Optimization.jl` branch and warned about on the
`Optimisers.jl` branch.
"""
function _train(model, data, train_cfg::TrainConfig, data_cfg::DataConfig, solve_kwargs::NamedTuple)
    return if is_optimisers_rule(train_cfg.opt)
        if !isempty(solve_kwargs)
            @warn "Unknown kwargs ignored on the Optimisers.jl path: $(join(keys(solve_kwargs), ", "))"
        end
        _train(model, data, train_cfg, data_cfg)
    else
        _train_optimization(model, data, train_cfg, data_cfg, solve_kwargs)
    end
end

"""
    train(model, data; train_cfg::TrainConfig = TrainConfig(), data_cfg::DataConfig = DataConfig())
    train(model, data; kwargs...)

Train a hybrid model using the provided data.

Two equivalent calling styles are supported:

1. **Typed configs** — pass complete `TrainConfig` / `DataConfig` objects:
   ```julia
   train(model, data;
       train_cfg = TrainConfig(nepochs=100, batchsize=32),
       data_cfg  = DataConfig(split_data_at=0.8),
   )
   ```

2. **Flat kwargs** — pass `TrainConfig` / `DataConfig` field names directly:
   ```julia
   train(model, data; nepochs=100, batchsize=32, split_data_at=0.8)
   ```

The two styles can also be mixed; flat kwargs override the corresponding fields of
the supplied `train_cfg` / `data_cfg`:
```julia
train(model, data; train_cfg = TrainConfig(nepochs=100), nepochs = 10)  # nepochs = 10
```

Returns `nothing` if data preparation fails (zero-size dimension in training or validation data).

# Arguments
- `model`: The hybrid model to train.
- `data`: Training data, a single `DimArray`, a single `DataFrame`, or a single `KeyedArray`.

# Keyword Arguments
- `train_cfg`: Training configuration. See [`TrainConfig`](@ref) for all options.
- `data_cfg`: Data preparation configuration. See [`DataConfig`](@ref) for all options.
- Any other kwargs are forwarded as overrides to `train_cfg` / `data_cfg`.

# Returns
A [`TrainResults`](@ref) with the following fields:
- `train_losses`: Per-epoch training losses.
- `val_losses`: Per-epoch validation losses.
- `snapshots`: Model parameter snapshots taken during training.
- `train_obs_pred`: Observed vs. predicted values on the training set.
- `val_obs_pred`: Observed vs. predicted values on the validation set.
- `train_diffs`: Additional diagnostic variables computed on the training set.
- `val_diffs`: Additional diagnostic variables computed on the validation set.
- `ps`: Final (or best) model parameters.
- `st`: Final (or best) model state.
- `best_epoch`: Epoch at which the best validation loss was achieved.
- `best_loss`: Best validation loss recorded during training.
"""
function train(
        model, data;
        train_cfg::TrainConfig = TrainConfig(),
        data_cfg::DataConfig = DataConfig(),
        kwargs...,
    )
    train_cfg, data_cfg, solve_kwargs = override_configs(train_cfg, data_cfg, kwargs)
    return _train(model, data, train_cfg, data_cfg, solve_kwargs)
end

function valid_mask(y)
    nt = (;)
    isempty = true
    for (k, v) in pairs(y)
        k_mask = .!isnan.(v)
        if !all(k_mask .== false)
            isempty = false
        end
        nt = merge(nt, NamedTuple([k => .!isnan.(v)]))
    end
    return nt, isempty
end

function train(model, data, save_ps; kwargs...)
    target_names = model.targets
    merge_kwargs = (; kwargs..., target_names)
    train_cfg, data_cfg, solve_kwargs = kwargs_to_configs(save_ps, merge_kwargs)
    return _train(model, data, train_cfg, data_cfg, solve_kwargs)
end

function expand_sequence_kwargs(kwargs)
    haskey(kwargs, :sequence_kwargs) || return kwargs

    seq_kw = kwargs[:sequence_kwargs]

    # map old sequence_kwargs keys to new DataConfig field names
    key_map = Dict(
        :input_window => :sequence_length,
        :output_window => :sequence_output_window,
        :output_shift => :sequence_output_shift,
        :lead_time => :sequence_lead_time,
    )

    expanded = NamedTuple(
        get(key_map, k, k) => v for (k, v) in pairs(seq_kw)
    )

    remaining = NamedTuple(k => kwargs[k] for k in keys(kwargs) if k !== :sequence_kwargs)
    return merge(remaining, expanded)
end

"""
    kwargs_to_configs(save_ps, kwargs) -> (TrainConfig, DataConfig, NamedTuple)

Build a fresh `(TrainConfig, DataConfig)` pair from a flat collection of kwargs.
Kwargs are split between the two configs based on `fieldnames(TrainConfig)` and
`fieldnames(DataConfig)`; anything left over is returned as the third element
and forwarded to `solve(...)` on the `Optimization.jl` path (or warned about
on the `Optimisers.jl` path — see `_train`).

`save_ps` is the deprecated positional argument from `train(model, data, save_ps; ...)`;
when non-empty it is forwarded as `tracked_params` on the resulting `TrainConfig`.
"""
function kwargs_to_configs(save_ps, kwargs)
    train_keys = fieldnames(TrainConfig)
    data_keys = fieldnames(DataConfig)

    kwargs = rename_deprecated_kwargs(kwargs)
    kwargs = expand_sequence_kwargs(kwargs)       # ← unpack sequence_kwargs if present

    train_kwargs = NamedTuple(k => kwargs[k] for k in keys(kwargs) if k in train_keys)
    data_kwargs = NamedTuple(k => kwargs[k] for k in keys(kwargs) if k in data_keys)
    solve_kwargs = NamedTuple(k => kwargs[k] for k in keys(kwargs) if k ∉ train_keys && k ∉ data_keys)

    if !isempty(save_ps)
        @warn "`save_ps` is deprecated, use `TrainConfig(tracked_params=(...))` instead."
        train_kwargs = merge(train_kwargs, (; tracked_params = save_ps))
    end

    return TrainConfig(; train_kwargs...), DataConfig(; data_kwargs...), solve_kwargs
end

"""
    override_configs(train_cfg, data_cfg, kwargs) -> (TrainConfig, DataConfig, NamedTuple)

Return `(train_cfg′, data_cfg′, solve_kwargs)` where any field present in `kwargs`
overrides the corresponding field of `train_cfg`/`data_cfg`. Fields not mentioned
in `kwargs` are kept as-is. Anything left over is returned as `solve_kwargs` and
forwarded to `solve(...)` on the `Optimization.jl` path (or warned about on the
`Optimisers.jl` path — see `_train`).
"""
function override_configs(train_cfg::TrainConfig, data_cfg::DataConfig, kwargs)
    train_keys = fieldnames(TrainConfig)
    data_keys = fieldnames(DataConfig)

    kwargs = rename_deprecated_kwargs(kwargs)
    kwargs = expand_sequence_kwargs(kwargs)

    train_overrides = NamedTuple(k => kwargs[k] for k in keys(kwargs) if k in train_keys)
    data_overrides = NamedTuple(k => kwargs[k] for k in keys(kwargs) if k in data_keys)
    solve_kwargs = NamedTuple(k => kwargs[k] for k in keys(kwargs) if k ∉ train_keys && k ∉ data_keys)

    return override_config(train_cfg, train_overrides),
        override_config(data_cfg, data_overrides),
        solve_kwargs
end

"""
    override_config(cfg, overrides::NamedTuple)

Return a new `cfg` of the same type with the fields named in `overrides` replaced.
Works with any `@kwdef` struct (e.g. [`TrainConfig`](@ref), [`DataConfig`](@ref)).
"""
function override_config(cfg::T, overrides::NamedTuple) where {T}
    isempty(overrides) && return cfg
    base = NamedTuple(k => getfield(cfg, k) for k in fieldnames(T))
    return T(; merge(base, overrides)...)
end

const DEPRECATED_KWARG_NAMES = Dict(
    :file_name => :model_name,
    :hybrid_name => :model_name,
    :folder_to_save => :output_folder,
)

function rename_deprecated_kwargs(kwargs)
    pairs = map(keys(kwargs), values(kwargs)) do k, v
        if haskey(DEPRECATED_KWARG_NAMES, k)
            new_k = DEPRECATED_KWARG_NAMES[k]
            @warn "kwarg `$k` has been renamed to `$new_k`."
            new_k => v
        else
            k => v
        end
    end
    return NamedTuple(pairs)
end

function evaluate_acc(ghm, x, forcings, y, y_no_nan, ps, st, loss_types, training_loss, extra_loss, agg)
    # Metric/validation evaluation is a plain forward pass (no autodiff). Switch
    # to test mode so layers like BatchNorm use their running statistics instead
    # of batch statistics, and to avoid LuxLib's "training is set to Val{true}()
    # but is not being used within an autodiff call" warning.
    st = LuxCore.testmode(st)
    loss_val, sts, ŷ = compute_loss(ghm, ps, st, ((x, forcings), (y, y_no_nan)), logging = LoggingLoss(train_mode = false, loss_types = loss_types, training_loss = training_loss, extra_loss = extra_loss, agg = agg))
    return loss_val, sts, ŷ
end

function styled_values(nt; digits = 5, color = nothing, paddings = nothing)
    formatted = [
        begin
                value_str = @sprintf("%.*f", digits, v)
                padded = isnothing(paddings) ? value_str : rpad(value_str, paddings[i])
                isnothing(color) ? padded : styled"{$color:$padded}"
            end
            for (i, v) in enumerate(values(nt))
    ]
    return join(formatted, "  ")
end

function header_and_paddings(nt; digits = 5)
    min_val_width = digits + 2  # 1 for "0", 1 for ".", rest for digits
    paddings = map(k -> max(length(string(k)), min_val_width), keys(nt))
    headers = [rpad(string(k), w) for (k, w) in zip(keys(nt), paddings)]
    return headers, paddings
end

function get_ps_st(train_from::TrainResults)
    return train_from.ps, train_from.st
end

function get_ps_st(train_from::Tuple)
    return train_from
end

function WrappedTuples(vec::Vector{EpochSnapshot})
    nt_vec = map(
        s -> (;
            l_train = s.l_train,
            l_val = s.l_val,
            ŷ_train = s.ŷ_train,
            ŷ_val = s.ŷ_val,
        ), vec
    )
    return WrappedTuples(nt_vec)
end
