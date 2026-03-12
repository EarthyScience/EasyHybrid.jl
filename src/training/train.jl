export train, TrainResults
# beneficial for plotting based on type TrainResults?
struct TrainResults
    train_history
    val_history
    ps_history
    train_obs_pred
    val_obs_pred
    train_diffs
    val_diffs
    ps
    st
    best_epoch
    best_loss
end

function train(model, data; train_cfg::TrainConfig = TrainConfig(), data_cfg::DataConfig = DataConfig())
    validate_config(train_cfg)
    ext = load_makie_extension(train_cfg)
    seed!(train_cfg.random_seed)

    (x_train, y_train), (x_val, y_val) = prepare_splits(data, model, data_cfg)
    loader = build_loader(x_train, y_train, train_cfg)
    ps, st, opt_state = init_model_state(model, train_cfg)

    init = compute_initial_state(model, x_train, y_train, x_val, y_val, ps, st, train_cfg)
    history = TrainingHistory(init)
    stopper = EarlyStopping(init.l_val, ps, st, train_cfg.patience)
    paths = resolve_paths(train_cfg)
    prog = build_progress(train_cfg)
    dashboard = init_dashboard(ext, init, train_cfg)

    save_initial_state!(paths, model, ps, st)

    record_or_run(ext, paths, train_cfg) do io
        for epoch in 1:train_cfg.nepochs
            ps, st, opt_state = run_epoch!(loader, model, ps, st, opt_state, train_cfg)
            snapshot = evaluate_epoch(model, x_train, y_train, x_val, y_val, ps, st, init, train_cfg)

            update!(history, snapshot)
            update!(stopper, snapshot, ps, st, epoch, train_cfg)
            save_epoch!(paths, model, ps, st, snapshot, epoch)
            update_dashboard!(dashboard, ext, snapshot, epoch, io)
            log_progress!(prog, init, snapshot, epoch, train_cfg)

            is_done(stopper) && break
        end
    end

    save_dashboard_img!(dashboard, ext, paths, stopper.best_epoch)
    ps, st = best_or_final(stopper, ps, st, train_cfg)
    save_final!(paths, model, ps, st, x_train, y_train, x_val, y_val, stopper, train_cfg)

    return build_results(model, history, stopper, ps, st, x_train, y_train, x_val, y_val)
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

function kwargs_to_configs(save_ps, kwargs)
    train_keys = fieldnames(TrainConfig)
    data_keys = fieldnames(DataConfig)

    # rename old kwargs to new names before sorting
    kwargs = rename_deprecated_kwargs(kwargs)

    train_kwargs = filter(((k, v),) -> k in train_keys, kwargs)
    data_kwargs = filter(((k, v),) -> k in data_keys, kwargs)

    unknown = filter(((k, v),) -> k ∉ train_keys && k ∉ data_keys, kwargs)
    if !isempty(unknown)
        @warn "Unknown kwargs will be ignored: $(join(keys(unknown), ", "))"
    end

    # fold save_ps into tracked_params if provided
    train_kwargs = if !isempty(save_ps)
        @warn "`save_ps` is deprecated, use `TrainConfig(tracked_params=(...))` instead."
        merge(train_kwargs, (; tracked_params = save_ps))
    else
        train_kwargs
    end

    return TrainConfig(; train_kwargs...), DataConfig(; data_kwargs...)
end

const DEPRECATED_KWARG_NAMES = (
    :hybrid_name => :model_name,
    :folder_to_save => :output_folder,
)

function rename_deprecated_kwargs(kwargs)
    renamed = map(kwargs) do (k, v)
        if k in keys(DEPRECATED_KWARG_NAMES)
            new_k = DEPRECATED_KWARG_NAMES[k]
            @warn "kwarg `$k` has been renamed to `$new_k`."
            new_k => v
        else
            k => v
        end
    end
    return renamed
end
