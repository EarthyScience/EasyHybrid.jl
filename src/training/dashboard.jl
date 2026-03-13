struct TrainDashboard
    observables
    fixed_observations
    eval_metric
    agg
    target_names
    monitor_names
end

function init_dashboard(ext, init::EpochSnapshot, cfg::TrainConfig, target_names)
    isnothing(ext) && return nothing

    observables, fixed_observations = initialize_plotting_observables(
        init.ŷ_train,
        init.ŷ_val,
        init.y_train,
        init.y_val,
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

function update_dashboard!(dashboard, ext, snapshot::EpochSnapshot, epoch::Int, io)
    isnothing(ext) && return
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

    return recordframe!(io)
end

function save_dashboard_img!(dashboard, ext, paths::TrainingPaths, best_epoch::Int)
    isnothing(ext) && return

    save_fig(paths.history_img, dashboard_figure())
    return @info "Dashboard saved to $(paths.history_img)"
end

function record_or_run(f, ext, paths::TrainingPaths, cfg::TrainConfig)
    return if !isnothing(ext)
        record_history(dashboard_figure(), paths.history_video; framerate = 24) do io
            f(io)
        end
    else
        f(nothing)
    end
end
