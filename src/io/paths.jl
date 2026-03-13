function resolve_paths(cfg::TrainConfig)
    folder = get_output_path(; folder_to_save = cfg.output_folder)
    suffix = cfg.model_name == "" ? "" : "_$(cfg.model_name)"

    @info "Training outputs will be saved to: $folder"

    return TrainingPaths(
        joinpath(folder, "trained_model_$(suffix).jld2"),
        joinpath(folder, "best_model_$(suffix).jld2"),
        joinpath(folder, "config_settings_$(suffix).yaml"),
        joinpath(folder, "train_history_$(suffix).png"),
        joinpath(folder, "training_history_$(suffix).mp4"),
    )
end
