struct TrainDashboard
    observables
    fixed_observations
    eval_metric
    agg
    target_names
    monitor_names
end

function init_dashboard(ext, init::EpochSnapshot, cfg::TrainConfig, y_train, y_val, target_names)
    isnothing(ext) && return nothing

    observables, fixed_observations = initialize_plotting_observables(
        init.ŷ_train,
        init.ŷ_val,
        y_train,
        y_val,
        init.l_train,
        init.l_val,
        cfg.loss_types[1],
        cfg.agg,
        target_names;
        monitor_names = cfg.monitor_names    # ← was missing
    )

    zoom_epochs = min(cfg.patience, 50)
    EasyHybrid.train_board(
        observables...,
        fixed_observations...,
        cfg.yscale,
        target_names,
        string(cfg.loss_types[1]);
        monitor_names = cfg.monitor_names,
        zoom_epochs
    )

    return TrainDashboard(
        observables,
        fixed_observations,
        cfg.loss_types[1],
        cfg.agg,
        target_names,
        cfg.monitor_names
    )
end

function update_dashboard!(dashboard, ext, snapshot::EpochSnapshot, epoch::Int, io, cfg::TrainConfig)
    isnothing(ext) && !cfg.save_training && return
    isnothing(dashboard) && return

    update_plotting_observables(
        dashboard.observables...,
        snapshot.l_train,
        snapshot.l_val,
        dashboard.eval_metric,
        dashboard.agg,
        snapshot.ŷ_train,
        snapshot.ŷ_val,
        dashboard.target_names,
        epoch;
        monitor_names = dashboard.monitor_names
    )

    if io !== nothing
        recordframe!(io)
    end
    return nothing
end

function save_dashboard_img!(dashboard, ext, paths::TrainingPaths, cfg::TrainConfig, best_epoch::Int)
    return if !isnothing(ext) && cfg.save_training
        save_fig(paths.history_img, dashboard_figure())
        @info "Dashboard saved to $(paths.history_img)"
    else
        nothing
    end
end

function record_or_run(f, ext, paths::TrainingPaths, cfg::TrainConfig)
    return if !isnothing(ext) && cfg.save_training
        record_history(dashboard_figure(), paths.history_video; framerate = 24) do io
            f(io)
        end
    else
        f(nothing)
    end
end
