function build_loader(x_train, forcings_train, y_train, cfg::TrainConfig)
    loader = DataLoader(
        ((x_train, forcings_train), y_train);
        parallel = true,
        batchsize = cfg.batchsize,
        shuffle = true,
    )

    @debug "Loader: $(length(loader)) batches of size $(cfg.batchsize)"

    return loader
end
