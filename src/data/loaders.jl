function build_loader(x_train, forcings_train, y_train, mask, cfg::TrainConfig)
    # An empty forcings NamedTuple reports numobs == 0 (MLCore._check_numobs
    # short-circuits on length(data) == 0), which conflicts with x_train's
    # actual observation count. Keep it out of the DataLoader's numobs check
    # and re-attach it to each batch afterwards instead.
    has_forcings = !isempty(forcings_train)
    inputs = has_forcings ? (x_train, forcings_train) : x_train

    loader = DataLoader(
        (inputs, (y_train, mask));
        parallel = true,
        batchsize = cfg.batchsize,
        shuffle = true,
    )

    @debug "Loader: $(length(loader)) batches of size $(cfg.batchsize)"

    has_forcings && return loader

    return Iterators.map(((x, y),) -> ((x, forcings_train), y), loader)
end
