function load_makie_extension(cfg::TrainConfig)
    ext = Base.get_extension(@__MODULE__, :EasyHybridMakie)

    if isnothing(ext)
        cfg.plotting && @warn "Makie extension not loaded, no plots will be generated."
        return nothing
    end

    if !cfg.plotting
        @debug "Plotting disabled."
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
        cfg.autodiff_backend isa Lux.AutoForwardDiff && (ps = ps |> ComponentArray)
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

function compute_initial_state(model, train_split, val_split, ps, st, cfg::TrainConfig)
    l_train, ŷ_train, _ = evaluate_split(model, train_split, ps, st, cfg)
    l_val, ŷ_val, _ = evaluate_split(model, val_split, ps, st, cfg)

    @debug "Initial train loss: $(l_train) | val loss: $(l_val)"

    return EpochSnapshot(l_train, l_val, ŷ_train, ŷ_val, 1)
end
