struct TrainDashboard
    figures::Dict{Symbol, Any}
    axes::Dict{Symbol, Any}
    plots::Dict{Symbol, Any}
end

function init_dashboard(ext, history::TrainingHistory, cfg::TrainConfig, y_train, y_val, target_names)
    isnothing(ext) && return nothing

    figures, axes, plots = build_dashboards(history, cfg, y_train, y_val)
    return TrainDashboard(figures, axes, plots)
end

function update_dashboard!(dashboard, ext, history::TrainingHistory, streams, cfg::TrainConfig)
    isnothing(ext) && !cfg.save_training && return
    isnothing(dashboard) && return

    update_step_dashboards!(dashboard, history, cfg)

    if streams !== nothing
        for stream in values(streams)
            recordframe!(stream)
        end
    end
    return nothing
end

function save_dashboard_img!(dashboard, ext, paths::TrainingPaths, cfg::TrainConfig, best_epoch::Int)
    return if !isnothing(ext) && cfg.save_training
        for (name, fig) in pairs(dashboard.figures)
            path = name == :dashboard ? paths.history_img : joinpath(paths.base_dir, "$(name)_history$(paths.suffix).png")
            save_fig(path, fig)
            @info "Dashboard ($name) saved to $(path)"
        end
    else
        nothing
    end
end

function record_or_run(f, ext, dashboard, paths::TrainingPaths, cfg::TrainConfig)
    return if !isnothing(ext) && !isnothing(dashboard) && cfg.save_training
        streams = Dict{Symbol, Any}()
        for (name, fig) in pairs(dashboard.figures)
            if :all in cfg.save_animations || name in cfg.save_animations
                streams[name] = VideoStream(fig; framerate = 24)
            end
        end

        f(streams)

        for (name, stream) in pairs(streams)
            path = name == :dashboard ? paths.history_video : joinpath(paths.base_dir, "$(name)_history$(paths.suffix).mp4")
            save_video(path, stream)
            @info "Animation ($name) saved to $(path)"
        end
    else
        f(nothing)
    end
end
