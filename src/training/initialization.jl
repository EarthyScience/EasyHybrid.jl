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
    if !cfg.keep_history
        @warn "Plotting enabled but keep_history is false. Plots will not be generated."
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

    train_state = if is_per_branch_opt(cfg.opt)
        # Keep `ps` as a nested `NamedTuple`: `Optimisers.jl` treats a
        # `ComponentArray` as a single leaf (no `Functors.functor` method),
        # which would collapse the per-branch state tree into one big Leaf
        # and silently apply the first rule to everything.
        ps isa ComponentArray && (ps = NamedTuple(ps))
        opt_state = build_opt_state(cfg.opt, ps)
        # `Lux.Training.TrainState`'s public constructor accepts only
        # `Optimisers.AbstractRule`. We use the (internal but stable)
        # positional constructor of `@concrete struct TrainState` to inject
        # a pre-built opt_state. Field order follows Lux 1.x; see
        # `Lux.Training.TrainState` in `Lux/src/helpers/training.jl`.
        Lux.Training.TrainState(
            nothing,    # cache
            nothing,    # objective_function
            nothing,    # allocator_cache
            model, ps, st, cfg.opt, opt_state, 0,
        )
    elseif is_optimisers_rule(cfg.opt)
        ps = ps |> ComponentArray
        Lux.Training.TrainState(model, ps, st, cfg.opt)
    else
        ps = ps |> ComponentArray
        nothing
    end

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
