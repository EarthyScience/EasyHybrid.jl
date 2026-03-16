function run_epoch!(loader, model, ps, st, opt_state, cfg::TrainConfig)
    loss_fn = build_loss_fn(model, cfg)

    for (x, y) in loader
        is_no_nan = .!isnan.(y)
        isempty(is_no_nan) && continue

        _, _, _, opt_state = Lux.Training.single_train_step!(
            cfg.autodiff_backend,
            loss_fn,
            (x, (y, is_no_nan)),
            opt_state;
            return_gradients = cfg.return_gradients
        )
    end

    ps = opt_state.parameters
    st = opt_state.states

    return ps, st, opt_state
end

# TODO: move out to losses.jl?
function build_loss_fn(model, cfg::TrainConfig)
    return (model, ps, st, (x, y)) -> compute_loss(
        model, ps, st, (x, y);
        logging = LoggingLoss(
            train_mode = true,
            loss_types = cfg.loss_types,
            training_loss = cfg.training_loss,
            extra_loss = cfg.extra_loss,
            agg = cfg.agg
        )
    )
end

function evaluate_epoch(model, x_train, y_train, x_val, y_val, ps, st, init::EpochSnapshot, cfg::TrainConfig)
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

    return EpochSnapshot(l_train, l_val, ŷ_train, ŷ_val)
end
