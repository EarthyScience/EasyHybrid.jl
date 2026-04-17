export train

"""
    train(model, data; train_cfg::TrainConfig = TrainConfig(), data_cfg::DataConfig = DataConfig())

Train a hybrid model using the provided data.

Returns `nothing` if data preparation fails (zero-size dimension in training or validation data).

# Arguments
- `model`: The hybrid model to train.
- `data`: Training data, a single `DimArray`, a single `DataFrame`, or a single `KeyedArray`.

# Keyword Arguments
- `train_cfg`: Training configuration. See [`TrainConfig`](@ref) for all options.
- `data_cfg`: Data preparation configuration. See [`DataConfig`](@ref) for all options.

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

# Example
```julia
cfg = TrainConfig(nepochs=100, batchsize=32)
result = train(myModel, myData; train_cfg=cfg)
```
"""
function train(model, data; train_cfg::TrainConfig = TrainConfig(), data_cfg::DataConfig = DataConfig())
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
    dashboard = init_dashboard(ext, init, train_cfg, y_train, y_val, model.targets)

    save_initial_state!(paths, model, ps, st, train_cfg)
    ps = ps |> train_cfg.gdev
    st = st |> train_cfg.gdev
    train_state = train_state |> train_cfg.gdev
    record_or_run(ext, paths, train_cfg) do io
        for epoch in 1:train_cfg.nepochs
            ps, st, train_state = run_epoch!(loader, model, ps, st, train_state, train_cfg)
            snapshot = evaluate_epoch(model, x_train, forcings_train, y_train, mask_train, x_val, forcings_val, y_val, mask_val, ps, st, init, train_cfg)

            update!(stopper, history, snapshot, ps, st, epoch, train_cfg)
            save_epoch!(paths, model, ps, st, snapshot, epoch, train_cfg)
            update_dashboard!(dashboard, ext, snapshot, epoch, io, train_cfg)
            log_progress!(prog, init, snapshot, epoch, train_cfg)

            is_done(stopper) && break
        end
    end

    save_dashboard_img!(dashboard, ext, paths, stopper.best_epoch, train_cfg)
    ps, st = best_or_final(stopper, ps, st, train_cfg)
    save_final!(paths, model, ps, st, x_train, forcings_train, y_train, x_val, forcings_val, y_val, stopper, train_cfg)

    return build_results(model, history, stopper, ps, st, x_train, forcings_train, y_train, x_val, forcings_val, y_val, train_cfg)
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
    Base.depwarn(
        """
        `train(model, data, save_ps; kwargs...)` is deprecated.
        Use the new API instead:

            train(model, data;
                train_cfg = TrainConfig(nepochs=100, ...),
                data_cfg  = DataConfig(split_data_at=0.8, ...)
            )

        See `?TrainConfig` and `?DataConfig` for all available options.
        """,
        :train
    )

    train_cfg, data_cfg = kwargs_to_configs(save_ps, kwargs)
    return train(model, data; train_cfg, data_cfg)
end

function expand_sequence_kwargs(kwargs)
    haskey(kwargs, :sequence_kwargs) || return kwargs

    seq_kw = kwargs[:sequence_kwargs]
    @warn "`sequence_kwargs` is deprecated, pass sequence options directly via `DataConfig` instead."

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

    # drop sequence_kwargs, merge expanded fields
    remaining = NamedTuple(k => kwargs[k] for k in keys(kwargs) if k !== :sequence_kwargs)
    return merge(remaining, expanded)
end

function kwargs_to_configs(save_ps, kwargs)
    train_keys = fieldnames(TrainConfig)
    data_keys = fieldnames(DataConfig)

    kwargs = rename_deprecated_kwargs(kwargs)
    kwargs = expand_sequence_kwargs(kwargs)       # ← unpack sequence_kwargs if present

    train_kwargs = NamedTuple(k => kwargs[k] for k in keys(kwargs) if k in train_keys)
    data_kwargs = NamedTuple(k => kwargs[k] for k in keys(kwargs) if k in data_keys)

    unknown = [k for k in keys(kwargs) if k ∉ train_keys && k ∉ data_keys]
    if !isempty(unknown)
        @warn "Unknown kwargs will be ignored: $(join(unknown, ", "))"
    end

    if !isempty(save_ps)
        @warn "`save_ps` is deprecated, use `TrainConfig(tracked_params=(...))` instead."
        train_kwargs = merge(train_kwargs, (; tracked_params = save_ps))
    end

    return TrainConfig(; train_kwargs...), DataConfig(; data_kwargs...)
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
