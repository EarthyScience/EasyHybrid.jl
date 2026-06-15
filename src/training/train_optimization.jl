"""
    _train_optimization(model, data, train_cfg, data_cfg, solve_kwargs)

`Optimization.jl`-based driver dispatched from `_train` whenever `train_cfg.opt`
is not an `Optimisers.AbstractRule` (e.g. `Optim.LBFGS()` / `Optimization.LBFGS()`);
see the
[SciML minibatching tutorial](https://docs.sciml.ai/Optimization/stable/tutorials/minibatch/).

Two modes, selected by `train_cfg.full_batch`:

- `full_batch = true`: pass the full training set as a single tuple to one
  `OptimizationProblem` and a single `solve(...)`. Batch-method idiom (the
  recommended L-BFGS setup): the objective is a single consistent function.
  `solve_kwargs` (e.g. `maxiters`, `g_abstol`, `f_reltol`) are splatted into
  `solve`, and `train_cfg.eval_every` builds a validation `EpochSnapshot` every
  N solver iterations via the callback.
- `full_batch = false`: explicit "repeated minibatch" loop grounded in Le et
  al., 2011 (*On Optimization Methods for Deep Learning*, ICML, §4.2). For each
  of `train_cfg.nepochs` outer passes we iterate a (reshuffled) `DataLoader`
  and, on each *fixed* minibatch, run `train_cfg.inner_maxiters` optimizer
  iterations (`solve(...; maxiters = inner_maxiters)`), warm-starting the next
  minibatch from the current parameters. Holding the minibatch fixed for a few
  iterations keeps the objective (and L-BFGS curvature pairs / line search)
  consistent — naive one-step-per-minibatch L-BFGS does not converge. A
  validation `EpochSnapshot` is built once per outer pass. Optimization.jl's
  own `DataLoader` iteration is **not** used here because it only applies to
  the `Optimisers.jl`-style solvers, not Optim.jl's L-BFGS.

Both modes honour `train_cfg.promote_f64`: promote `ps` to `Float64` before
optimization (workaround for
[Lux.jl#1260](https://github.com/LuxDL/Lux.jl/issues/1260)).
"""
function _train_optimization(model, data, train_cfg::TrainConfig, data_cfg::DataConfig, solve_kwargs::NamedTuple)
    validate_config(train_cfg)
    ext = load_makie_extension(train_cfg)
    seed!(train_cfg.random_seed)

    ((x_train, forcings_train), y_train), ((x_val, forcings_val), y_val) =
        prepare_splits(data, model, data_cfg)
    mask_train, _ = valid_mask(y_train)
    mask_val, _ = valid_mask(y_val)

    ps, st, _ = init_model_state(model, train_cfg)
    ps_ca = ComponentArray(ps)
    if train_cfg.promote_f64
        ps_ca = ps_ca .|> Float64
    end

    init = compute_initial_state(
        model, x_train, forcings_train, y_train, mask_train,
        x_val, forcings_val, y_val, mask_val, ps_ca, st, train_cfg,
    )
    history = TrainingHistory(init)
    stopper = EarlyStopping(init.l_val, ps_ca, st, train_cfg)
    paths = resolve_paths(train_cfg)
    prog = build_progress(train_cfg)
    dashboard = init_dashboard(ext, init, train_cfg, y_train, y_val, model.targets)

    save_initial_state!(paths, model, ps_ca, st, train_cfg)

    loss_fn = _build_optim_loss(model, st, train_cfg)
    opt_func = OptimizationFunction(loss_fn, train_cfg.autodiff_backend)

    final_ps = Ref{Any}(ps_ca)
    record_or_run(ext, paths, train_cfg) do io
        if train_cfg.full_batch
            # Convert once (not per objective/gradient evaluation): the full
            # training set is the fixed `p` for the entire solve.
            data_arg = collect_dim_data(
                (x_train, forcings_train), (y_train, mask_train), train_cfg,
            )
            opt_prob = OptimizationProblem(opt_func, ps_ca, data_arg)
            cb = _optim_callback(
                model, st, init, history, stopper, dashboard, ext, prog, paths,
                x_train, forcings_train, y_train, mask_train,
                x_val, forcings_val, y_val, mask_val,
                io, train_cfg, final_ps,
            )
            res = solve(opt_prob, train_cfg.opt; callback = cb, solve_kwargs...)
            final_ps[] = res.u
        else
            _run_minibatch!(
                final_ps, opt_func, ps_ca, model, st, init, history, stopper,
                dashboard, ext, prog, paths,
                x_train, forcings_train, y_train, mask_train,
                x_val, forcings_val, y_val, mask_val,
                io, train_cfg, solve_kwargs,
            )
        end
    end

    ps_out = final_ps[]
    save_dashboard_img!(dashboard, ext, paths, train_cfg, stopper.best_epoch)
    ps_out, st_out = best_or_final(stopper, ps_out, st, train_cfg)
    save_final!(
        paths, model, ps_out, st_out,
        x_train, forcings_train, y_train,
        x_val, forcings_val, y_val,
        stopper, train_cfg,
    )

    return build_results(
        model, history, stopper, ps_out, st_out,
        x_train, forcings_train, y_train,
        x_val, forcings_val, y_val,
        train_cfg,
    )
end

"""
Build the scalar loss closure consumed by `Optimization.jl`, called as
`loss_fn(p, data)`. `data` is an **already device-placed / `Array`-converted**
batch (shape `((x, forcings), (y, mask))`), produced once per (mini)batch by
`collect_dim_data` in the caller — *not* inside this closure. Keeping the data
prep out of the loss is important: L-BFGS line searches call the objective many
times per iteration, so re-running `collect_dim_data` (NamedTuple rebuilds,
`Array` copies, `gdev` transfers) on every evaluation was a major slowdown,
especially on the minibatch path. It also keeps the closure trivially
Zygote-differentiable (no `pairs(...)`/`∇map` in the AD tape).
"""
function _build_optim_loss(model, st, cfg::TrainConfig)
    logging = LoggingLoss(
        train_mode = true,
        loss_types = cfg.loss_types,
        training_loss = cfg.training_loss,
        extra_loss = cfg.extra_loss,
        agg = cfg.agg,
    )
    return function (p, data)
        loss_val, _, _ = compute_loss(model, p, st, data; logging)
        return loss_val
    end
end

"""
Explicit "repeated minibatch" driver for the `full_batch = false` Optimization
path (Le et al., 2011, ICML, §4.2). For each of `cfg.nepochs` outer passes we
iterate a reshuffled `DataLoader`; on every *fixed* minibatch we run
`cfg.inner_maxiters` optimizer iterations warm-started from the current `ps`,
then resample. A validation `EpochSnapshot` (history / early-stopping /
dashboard / checkpoint) is built once per outer pass, so `patience` is counted
in outer passes — consistent with the `Optimisers.jl` loop.

`maxiters` / `epochs` from `solve_kwargs` are dropped (the per-minibatch budget
is `cfg.inner_maxiters` and the pass count is `cfg.nepochs`); any remaining
`solve_kwargs` (e.g. `g_abstol`, `f_reltol`) are forwarded to each inner solve.
"""
function _run_minibatch!(
        final_ps, opt_func, ps0, model, st, init, history, stopper,
        dashboard, ext, prog, paths,
        x_train, forcings_train, y_train, mask_train,
        x_val, forcings_val, y_val, mask_val,
        io, cfg::TrainConfig, solve_kwargs::NamedTuple,
    )
    loader = build_loader(x_train, forcings_train, y_train, mask_train, cfg)
    inner_kwargs = Base.structdiff(solve_kwargs, (; maxiters = nothing, epochs = nothing))
    ps = ps0

    # Build the problem once and reuse it across minibatches via `remake`, which
    # recreates it cheaply with a new initial guess (`u0`) and minibatch data
    # (`p`) while keeping the same `OptimizationFunction` — see the SciML
    # polyalgorithm tutorial. The placeholder batch is immediately replaced.
    x0b, y0b = first(loader)
    opt_prob = OptimizationProblem(opt_func, ps, collect_dim_data(x0b, y0b, cfg))

    for epoch in 1:cfg.nepochs
        for (x, y) in loader
            isemptybatch(y[2]) && continue
            # Convert the minibatch once here, then hold it fixed as `p` for all
            # `inner_maxiters` solver iterations — avoids re-running
            # `collect_dim_data` on every line-search objective evaluation.
            data = collect_dim_data(x, y, cfg)
            opt_prob = remake(opt_prob; u0 = ps, p = data)
            res = solve(opt_prob, cfg.opt; maxiters = cfg.inner_maxiters, inner_kwargs...)
            ps = res.u
            final_ps[] = ps
        end

        snapshot = evaluate_epoch(
            model, x_train, forcings_train, y_train, mask_train,
            x_val, forcings_val, y_val, mask_val,
            ps, st, init, cfg,
        )
        update!(stopper, history, snapshot, ps, st, epoch, cfg)
        save_epoch!(paths, model, ps, st, snapshot, epoch, cfg)
        update_dashboard!(dashboard, ext, snapshot, epoch, io, cfg)
        log_progress!(prog, init, snapshot, epoch, cfg)

        is_done(stopper) && break
    end

    return final_ps
end

function _optim_callback(
        model, st, init, history, stopper, dashboard, ext, prog, paths,
        x_train, forcings_train, y_train, mask_train,
        x_val, forcings_val, y_val, mask_val,
        io, cfg::TrainConfig, final_ps,
    )
    return function (state, _loss)
        iter = state.iter
        ps_cur = state.u
        final_ps[] = ps_cur

        if iter > 0 && iter % cfg.eval_every == 0
            snapshot = evaluate_epoch(
                model, x_train, forcings_train, y_train, mask_train,
                x_val, forcings_val, y_val, mask_val,
                ps_cur, st, init, cfg,
            )
            update!(stopper, history, snapshot, ps_cur, st, iter, cfg)
            save_epoch!(paths, model, ps_cur, st, snapshot, iter, cfg)
            update_dashboard!(dashboard, ext, snapshot, iter, io, cfg)
            log_progress!(prog, init, snapshot, iter, cfg)
        end

        return is_done(stopper)
    end
end
