struct TrainingPaths
    checkpoint::String      # main .jld2 file, updated every epoch
    best_model::String      # best model .jld2, updated on improvement
    config_yaml::String     # config snapshot
    history_img::String     # final dashboard screenshot
    history_video::String   # training animation .mp4
end
