"""
Paths to all output files produced during a training run.
"""
struct TrainingPaths
    "Main `.jld2` checkpoint file, updated every epoch."
    checkpoint::String

    "Best model `.jld2` file, updated whenever validation loss improves."
    best_model::String

    "YAML snapshot of the training configuration."
    config_yaml::String

    "Final dashboard screenshot saved at the end of training."
    history_img::String

    "Training animation saved as an `.mp4` file."
    history_video::String
end
