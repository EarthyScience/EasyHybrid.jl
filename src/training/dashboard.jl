struct TrainDashboard
    ax
    plt
end

function init_dashboard(ext, history::TrainingHistory, cfg::TrainConfig, y_train, y_val, target_names)
    isnothing(ext) && return nothing

    fig, ax, plt = train_dashboard(history, cfg)
    return TrainDashboard(ax, plt)
end

function update_dashboard!(dashboard, ext, history::TrainingHistory, io, cfg::TrainConfig)
    isnothing(ext) && !cfg.save_training && return
    isnothing(dashboard) && return

    update_step_dashboard!(dashboard, history, cfg)

    if io !== nothing
        recordframe!(io)
    end
    return nothing
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
