function load_makie_extension(cfg::TrainConfig)
    ext = Base.get_extension(@__MODULE__, :EasyHybridMakie)

    if isnothing(ext)
        @warn "Makie extension not loaded, no plots will be generated."
        return nothing
    end

    if !cfg.plotting
        @info "Plotting disabled."
        return nothing
    end

    return ext
end

function init_model_state(model, cfg::TrainConfig)
    if isnothing(cfg.train_from)
        ps, st = LuxCore.setup(Random.default_rng(), model)
        ps = ps |> ComponentArray
    else
        ps, st = get_ps_st(cfg.train_from)
    end

    # ps = ComponentArray(ps)
    train_state = Lux.Training.TrainState(model, ps, st, cfg.opt)

    return ps, st, train_state
end

struct EpochSnapshot
    l_train
    l_val
    ŷ_train
    ŷ_val
    epoch
end

function compute_initial_state(model, x_train, forcings_train, y_train, mask_train, x_val, forcings_val, y_val, mask_val, ps, st, cfg::TrainConfig)
    l_train, _, ŷ_train = evaluate_acc(
        model, x_train, forcings_train, y_train, mask_train,
        ps, st, cfg.loss_types, cfg.training_loss, cfg.extra_loss, cfg.agg
    )
    l_val, _, ŷ_val = evaluate_acc(
        model, x_val, forcings_val, y_val, mask_val,
        ps, st, cfg.loss_types, cfg.training_loss, cfg.extra_loss, cfg.agg
    )

    @debug "Initial train loss: $(l_train) | val loss: $(l_val)"

    return EpochSnapshot(l_train, l_val, ŷ_train, ŷ_val, 0.9)
end
