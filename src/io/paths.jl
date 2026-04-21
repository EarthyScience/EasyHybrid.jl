function resolve_paths(cfg::TrainConfig)
    folder = get_output_path(; folder_to_save = cfg.output_folder)
    suffix = cfg.model_name == "" ? "" : "_$(cfg.model_name)"
    if cfg.save_training
         @info "Training outputs will be saved to: $folder"
    end

    return TrainingPaths(
        joinpath(folder, "trained_model$(suffix).jld2"),
        joinpath(folder, "best_model$(suffix).jld2"),
        joinpath(folder, "config_settings$(suffix).yaml"),
        joinpath(folder, "train_history$(suffix).png"),
        joinpath(folder, "training_history$(suffix).mp4"),
    )
end
