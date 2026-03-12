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
    else
        ps, st = get_ps_st(cfg.train_from)
    end

    ps = ComponentArray(ps)
    opt_state = Lux.Training.TrainState(model, ps, st, cfg.opt)

    return ps, st, opt_state
end

struct EpochSnapshot
    l_train
    l_val
    ŷ_train
    ŷ_val
    y_train
    y_val
    is_no_nan_t
    is_no_nan_v
end


function compute_initial_state(model, x_train, y_train, x_val, y_val, ps, st, cfg::TrainConfig)
    is_no_nan_t = .!isnan.(y_train)
    is_no_nan_v = .!isnan.(y_val)

    l_train, _, ŷ_train = evaluate_acc(
        model, x_train, y_train, is_no_nan_t,
        ps, st, cfg.loss_types, cfg.training_loss, cfg.extra_loss, cfg.agg
    )
    l_val, _, ŷ_val = evaluate_acc(
        model, x_val, y_val, is_no_nan_v,
        ps, st, cfg.loss_types, cfg.training_loss, cfg.extra_loss, cfg.agg
    )

    @info "Initial train loss: $(l_train) | val loss: $(l_val)"

    return EpochSnapshot(l_train, l_val, ŷ_train, ŷ_val, y_train, y_val, is_no_nan_t, is_no_nan_v)
end
