# Split-level evaluation used by the training loops. Every function dispatches on
# the split type: `InMemorySplit` (arrays already in memory) or `LazyHybridData`
# (data read block by block from disk).

function _eval_logging(cfg::TrainConfig)
    return LoggingLoss(
        train_mode = false,
        loss_types = cfg.loss_types,
        training_loss = cfg.training_loss,
        extra_loss = cfg.extra_loss,
        agg = cfg.agg,
    )
end

"""
    evaluate_split(model, split, ps, st, cfg::TrainConfig) -> (loss, ŷ, y)

Evaluation-mode losses, predictions and observed targets for a whole training or
validation split. Lazy splits are evaluated block by block; only the targets and
predictions (not the inputs) are held in memory.
"""
function evaluate_split(model, s::InMemorySplit, ps, st, cfg::TrainConfig)
    loss, _, ŷ = evaluate_acc(
        model, s.x, s.forcings, s.y, s.mask,
        ps, st, cfg.loss_types, cfg.training_loss, cfg.extra_loss, cfg.agg
    )
    return loss, ŷ, s.y
end

function evaluate_split(model, d::LazyHybridData, ps, st, cfg::TrainConfig)
    ŷ, _, y, mask = _predict_blocks(model, d, ps, st)
    loss = evaluation_loss(model, ŷ, y, mask, ps, _eval_logging(cfg))
    return loss, ŷ, y
end

"""
    predict_split(model, split, ps, st, cfg::TrainConfig) -> (ŷ, st_out, y)

Test-mode predictions on the CPU for a whole split, together with the returned
model state and the observed targets.
"""
function predict_split(model, s::InMemorySplit, ps, st, cfg::TrainConfig)
    ŷ, st_out = model((cfg.cdev(s.x), cfg.cdev(s.forcings)), cfg.cdev(ps), LuxCore.testmode(cfg.cdev(st)))
    return ŷ, st_out, s.y
end

function predict_split(model, d::LazyHybridData, ps, st, cfg::TrainConfig)
    ŷ, st_out, y, _ = _predict_blocks(model, d, cfg.cdev(ps), cfg.cdev(st))
    return ŷ, st_out, y
end

"""
    observations(split)

Observed targets of a split as a `NamedTuple` of vectors. For lazy data they are
taken from the last full evaluation pass, or read block by block if none ran yet.
"""
observations(s::InMemorySplit) = s.y

function observations(d::LazyHybridData)
    isnothing(d.obs_cache[]) || return d.obs_cache[]
    ys = Any[]
    ns = Int[]
    for (_, (y, _)) in _eval_loader(d)
        push!(ys, y)
        push!(ns, length(first(values(y))))
    end
    isempty(ys) && throw(ArgumentError("lazy split has no observations with complete inputs and at least one observed target"))
    return d.obs_cache[] = _concat_obs(map(identity, ys), ns)
end

"""
    full_batch_data(split, cfg::TrainConfig)

Whole training split as one device-placed batch for the `full_batch = true`
`Optimization.jl` path. Not available for lazy data.
"""
full_batch_data(s::InMemorySplit, cfg::TrainConfig) = collect_dim_data((s.x, s.forcings), (s.y, s.mask), cfg)

full_batch_data(::LazyHybridData, ::TrainConfig) = throw(
    ArgumentError(
        "`full_batch = true` needs the whole training set in memory, which defeats lazy data. Use `full_batch = false` (minibatches), or set `lazy = false`."
    )
)

"""
    compact_history!(history, split, cfg::TrainConfig)

For lazy splits, keep full predictions only in the latest snapshot: the previous one
keeps just the `cfg.monitor_names` outputs (all the monitor plots need), so the
history does not grow by a full set of predictions per epoch. No-op for in-memory splits.
"""
compact_history!(history, split, cfg::TrainConfig) = history

function compact_history!(history::TrainingHistory, ::LazyHybridData, cfg::TrainConfig)
    length(history.snapshots) < 2 && return history
    s = history.snapshots[end - 1]
    history.snapshots[end - 1] = EpochSnapshot(
        s.l_train, s.l_val, _monitored(s.ŷ_train, cfg.monitor_names), _monitored(s.ŷ_val, cfg.monitor_names), s.epoch
    )
    return history
end

_monitored(ŷ::NamedTuple, names) = (; (Symbol(m) => getfield(ŷ, Symbol(m)) for m in names if haskey(ŷ, Symbol(m)))...)
_monitored(ŷ, names) = ŷ

# A cheap stand-in batch (e.g. to build an `OptimizationProblem`). For lazy data this
# avoids `first(loader)`, which would read a whole shuffle group and leave a prefetch
# task reading the store concurrently with the next iteration.
_placeholder_batch(loader) = first(loader)
_placeholder_batch(l::LazyBatchLoader) = MLUtils.getobs(l.data, 1:min(l.batchsize, MLUtils.numobs(l.data)))

# Sequential (unshuffled) pass over a lazy split, one storage-aligned block per batch.
_eval_loader(d::LazyHybridData) = LazyBatchLoader(d; batchsize = d.block_span, shuffle = false)

function _predict_blocks(model, d::LazyHybridData, ps, st)
    st_test = LuxCore.testmode(st)
    ŷs, ys, masks, ns = Any[], Any[], Any[], Int[]
    st_out = st_test
    for ((x, forcings), (y, mask)) in _eval_loader(d)
        ŷ, st_out = model((x, forcings), ps, st_test)
        push!(ŷs, ŷ)
        push!(ys, y)
        push!(masks, mask)
        push!(ns, length(first(values(y))))
    end
    isempty(ŷs) && throw(ArgumentError("lazy split has no observations with complete inputs and at least one observed target"))
    # `map(identity, …)` narrows the `Any` accumulators (function barrier for `_concat_obs`).
    y = _concat_obs(map(identity, ys), ns)
    d.obs_cache[] = y
    return _concat_obs(map(identity, ŷs), ns), st_out, y, _concat_obs(map(identity, masks), ns)
end

# Concatenate per-block values along the observation (last) dimension. Arrays whose
# last dimension matches the block size in every block are per-observation outputs;
# anything else (e.g. global parameters) is taken from the first block.
function _concat_obs(vs::AbstractVector, ns::AbstractVector{Int})
    v1 = first(vs)
    if v1 isa NamedTuple
        ks = keys(v1)
        return NamedTuple{ks}(map(k -> _concat_obs([v[k] for v in vs], ns), ks))
    elseif v1 isa AbstractArray && all(i -> vs[i] isa AbstractArray && size(vs[i], ndims(vs[i])) == ns[i], eachindex(vs))
        length(vs) == 1 && return copy(v1)
        nd = ndims(v1)
        nd == 1 && return reduce(vcat, vs)
        nd == 2 && return reduce(hcat, vs)
        return cat(vs...; dims = nd)
    else
        return v1
    end
end
