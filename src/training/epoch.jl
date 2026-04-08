function collect_dim_data(x, y, cfg)
    x_col = Array(x[1])
    forcing_nt = NamedTuple([k => Array(v) for (k, v) in pairs(x[2])])
    targets_nt = NamedTuple([k => Array(v) for (k, v) in pairs(y[1])])
    masks_nt = NamedTuple([k => Array(v) for (k, v) in pairs(y[2])])
    return ((x_col, forcing_nt), (targets_nt, masks_nt)) |> cfg.gdev
end

function run_epoch!(loader, model, ps, st, train_state, cfg::TrainConfig)
    loss_fn = build_loss_fn(model, cfg)
    for (x, y) in loader
        (x_col, y_col) = collect_dim_data(x, y, cfg)
        if isemptybatch(y_col[2])
            continue
        end
        _, _, _, train_state = Lux.Training.single_train_step!(
            cfg.autodiff_backend,
            loss_fn,
            (x_col, y_col),
            train_state;
            return_gradients = cfg.return_gradients
        )
    end

    ps = train_state.parameters
    st = train_state.states

    return ps, st, train_state
end

function isemptybatch(mask)
    return all(x -> all(x .== 0), mask)
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

function evaluate_epoch(model, x_train, forcings_train, y_train, mask_train, x_val, forcings_val, y_val, mask_val, ps, st, init::EpochSnapshot, cfg::TrainConfig)
    ps_cpu = ps |> cfg.cdev
    l_train, _, ŷ_train = evaluate_acc(
        model, x_train, forcings_train, y_train, mask_train,
        ps_cpu, st, cfg.loss_types, cfg.training_loss, cfg.extra_loss, cfg.agg
    )
    l_val, _, ŷ_val = evaluate_acc(
        model, x_val, forcings_val, y_val, mask_val,
        ps_cpu, st, cfg.loss_types, cfg.training_loss, cfg.extra_loss, cfg.agg
    )

    return EpochSnapshot(l_train, l_val, ŷ_train, ŷ_val)
end
