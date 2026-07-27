export cv_test, CVTestResults, cv_fold_losses, pooled_obs_pred, cv_performance_table

# =============================================================================
# Result type
# =============================================================================

"""
    CVTestResults

Output of [`cv_test`](@ref). `mode` is one of `:cv` (cross-validation only),
`:test` (single held-out test fold) or `:nested` (`test_fold=:all`: every fold
is held out as test once, with its own inner search/CV).

Fields are `nothing` when they do not apply to the current `mode`. Scores are in
the *natural* orientation of the first `loss_types` metric (higher is better for
`:nse`/`:kge`/`:r2`/`:pearson`, lower otherwise) and aggregated across targets
with `agg`.

For `:nested`, `pooled_test_loss` is the metric recomputed **once** on the
concatenation of every fold's held-out test predictions (each observation
predicted exactly once by a model that never saw it) — the honest out-of-sample
score. `pooled_val_loss` is the same idea over the held-out folds' validation sets.
"""
struct CVTestResults
    mode::Symbol
    agg::Function
    loss_types::Vector{Symbol}
    folds::Vector{Int}
    best_hyperparams::NamedTuple
    ho::Union{Hyperoptimizer, Nothing}
    cv_fold_results::Union{Vector{Union{TrainResults, Nothing}}, Nothing}
    mean_cv_loss::Float64
    test_fold::Union{Int, Nothing}
    test_loss::Union{Float64, Nothing}
    test_obs_pred::Union{DataFrame, Nothing}
    final_train::Union{TrainResults, Nothing}
    mean_test_loss::Union{Float64, Nothing}
    pooled_val_loss::Union{Float64, Nothing}
    pooled_test_loss::Union{Float64, Nothing}
    """One `CVTestResults` per held-out test fold when `mode=:nested`."""
    held_outs::Union{Vector{CVTestResults}, Nothing}
end

"""Best validation score per CV fold (derived from `cv_fold_results`)."""
cv_fold_losses(r::CVTestResults) =
    r.cv_fold_results === nothing ? Float64[] :
    Float64[Float64(out.best_loss) for out in r.cv_fold_results if out !== nothing]

# =============================================================================
# Loss orientation helpers
# =============================================================================

_metric(loss_types) = isempty(loss_types) ? :mse : first(loss_types)

"""Index of the best score under the direction of `loss_type` (skips non-finite)."""
function _best_index(scores::AbstractVector{<:Real}, loss_type::Symbol)
    best = findfirst(isfinite, scores)
    best === nothing && return 1
    for i in (best + 1):length(scores)
        isfinite(scores[i]) && isbetter(scores[i], scores[best], loss_type) && (best = i)
    end
    return best
end

# --- Pooled (concatenated) obs-vs-pred metrics -------------------------------

"""Build an obs/pred `DataFrame` (`t`, `t_pred` per target) from raw arrays."""
function _obs_pred_df(ŷ, y, targets)
    cols = Pair{Symbol, Any}[]
    for t in targets
        y_t = _get_target_y(y, t)
        ŷ_t = _get_target_ŷ(ŷ, y_t, t)
        push!(cols, Symbol(t) => vec(collect(y_t)))
        push!(cols, Symbol(string(t), "_pred") => vec(collect(ŷ_t)))
    end
    return DataFrame(cols...)
end

"""Target columns of an obs/pred frame (those `c` for which `c_pred` also exists)."""
function _target_cols(df::DataFrame)
    cols = propertynames(df)
    return [c for c in cols if Symbol(string(c), "_pred") in cols]
end

"""
Concatenate obs/pred frames and compute `metric` once over the pooled samples,
aggregating across targets with `agg`. Returns `nothing` if nothing is poolable.
"""
function _pooled_metric(dfs, metric::Symbol, agg)
    frames = [df for df in dfs if df !== nothing && size(df, 1) > 0]
    isempty(frames) && return nothing
    df = reduce((a, b) -> vcat(a, b; cols = :intersect), frames)
    scores = Float64[]
    for t in _target_cols(df)
        obs = Float64.(df[!, t])
        pred = Float64.(df[!, Symbol(string(t), "_pred")])
        m = .!isnan.(obs) .& .!isnan.(pred)
        any(m) && push!(scores, Float64(loss_fn(pred, obs, m, Val(metric))))
    end
    return isempty(scores) ? nothing : Float64(agg(scores))
end

"""Aggregated train and val scores taken from the same (best-val) snapshot."""
function _train_val_at_best(tr::TrainResults, agg)
    vh, th = tr.val_history, tr.train_history
    isempty(vh) && return (nothing, Float64(tr.best_loss))
    valscores = Float64[Float64(extract_agg_loss(v, agg)) for v in vh]
    # Align to the snapshot whose val score matches the early-stopping best.
    idx = argmin(abs.(valscores .- Float64(tr.best_loss)))
    train = idx <= length(th) ? Float64(extract_agg_loss(th[idx], agg)) : nothing
    return (train, valscores[idx])
end

# =============================================================================
# Pooled held-out predictions & per-datastream performance (public helpers)
# =============================================================================

"""
    pooled_obs_pred(r::CVTestResults; which=:test)

Concatenate the held-out obs/pred frames of a cross-validation result, tagging each
row with the `fold` that produced it. For `mode=:nested` this pools every fold's
held-out predictions (each observation predicted exactly once by a model that never
saw it); for `mode=:test` it returns the single held-out fold.

- `which=:test`: held-out **test** predictions.
- `which=:val`: the held-out folds' **validation** predictions (`final_train.val_obs_pred`).

Returns a `DataFrame` with the per-target `t`/`t_pred` columns plus a `:fold` column,
or `nothing` when no matching frames are available (e.g. `mode=:cv`).
"""
function pooled_obs_pred(r::CVTestResults; which::Symbol = :test)
    which in (:test, :val) ||
        throw(ArgumentError("`which` must be :test or :val; got :$which."))
    _frame(x) = which === :test ? x.test_obs_pred :
        (x.final_train === nothing ? nothing : x.final_train.val_obs_pred)

    results = r.mode === :nested ? r.held_outs : [r]
    frames = DataFrame[]
    for x in results
        df = _frame(x)
        df === nothing && continue
        d = copy(df)
        d[!, :fold] .= x.test_fold === nothing ? 1 : x.test_fold
        push!(frames, d)
    end
    isempty(frames) && return nothing
    return reduce((a, b) -> vcat(a, b; cols = :intersect), frames)
end

"""
    cv_performance_table(r::CVTestResults; which=:test, metric=first(loss_types))

Per-datastream performance from pooled held-out predictions (see [`pooled_obs_pred`](@ref)).
Returns a `DataFrame` with one row per `fold` plus a final `:pooled` row, and one
`metric` column per target named `<metric>_<target>`. Returns `nothing` when no
pooled predictions are available.
"""
function cv_performance_table(r::CVTestResults; which::Symbol = :test,
        metric::Symbol = _metric(r.loss_types))
    pooled = pooled_obs_pred(r; which)
    pooled === nothing && return nothing

    targets = _target_cols(pooled)
    colnames = [Symbol(metric, :_, t) for t in targets]
    score(df, t) = begin
        obs  = Float64.(df[!, t])
        pred = Float64.(df[!, Symbol(string(t), "_pred")])
        m = .!isnan.(obs) .& .!isnan.(pred)
        any(m) ? Float64(loss_fn(pred, obs, m, Val(metric))) : NaN
    end

    rows = NamedTuple[]
    for f in sort(unique(pooled.fold))
        sub = filter(:fold => ==(f), pooled)
        push!(rows, (; fold = f,
            (colnames[i] => score(sub, targets[i]) for i in eachindex(targets))...))
    end
    push!(rows, (; fold = :pooled,
        (colnames[i] => score(pooled, targets[i]) for i in eachindex(targets))...))
    return DataFrame(rows)
end

# =============================================================================
# Pretty-printing (compact)
# =============================================================================

_fmt(::Nothing) = "—"
_fmt(x::Real) = isfinite(x) ? @sprintf("%.5g", x) : string(x)

function _fmt_hyper(hp::NamedTuple)
    isempty(hp) && return "(none)"
    return "(" * join(("$k=$(sprint(show, v; context = :compact => true))" for (k, v) in pairs(hp)), ", ") * ")"
end

function Base.show(io::IO, ::MIME"text/plain", r::CVTestResults)
    printstyled(io, "CVTestResults"; bold = true, color = :cyan)
    println(io, "  (mode=:$(r.mode), metric=:$(_metric(r.loss_types)), agg=$(nameof(r.agg)))")

    if r.mode === :nested
        printstyled(io, "  per held-out test fold (each selects its own model):\n"; color = :light_black)
        @printf(io, "  %4s  %10s  %10s  %10s\n", "fold", "train", "val", "test")
        for ho in r.held_outs
            tr, vl = ho.final_train === nothing ? (nothing, nothing) : _train_val_at_best(ho.final_train, r.agg)
            @printf(io, "  %4s  %10s  %10s  %10s\n",
                string(ho.test_fold), _fmt(tr), _fmt(vl), _fmt(ho.test_loss))
        end
        printstyled(io, "  pooled val:  "; color = :yellow); println(io, _fmt(r.pooled_val_loss))
        printstyled(io, "  pooled test: "; color = :green); println(io, _fmt(r.pooled_test_loss))
        return
    end

    !isempty(r.best_hyperparams) &&
        (printstyled(io, "  best hyperparams: "; color = :yellow); println(io, _fmt_hyper(r.best_hyperparams)))

    if r.final_train !== nothing
        tr, vl = _train_val_at_best(r.final_train, r.agg)
        printstyled(io, "  best train: "; color = :yellow); println(io, _fmt(tr))
        printstyled(io, "  best val:   "; color = :yellow); println(io, _fmt(vl))
    elseif r.cv_fold_results !== nothing
        tr, vl = _mean_train_val(r.cv_fold_results, r.agg)
        printstyled(io, "  best train: "; color = :yellow); println(io, _fmt(tr))
        printstyled(io, "  best val:   "; color = :yellow); println(io, _fmt(vl))
    end
    r.test_loss === nothing ||
        (printstyled(io, "  test:         "; color = :yellow); println(io, _fmt(r.test_loss)))
    return
end

function Base.show(io::IO, r::CVTestResults)
    if r.mode === :nested
        print(io, "CVTestResults(:nested, pooled_test=", _fmt(r.pooled_test_loss), ", pooled_val=", _fmt(r.pooled_val_loss), ")")
    elseif r.mode === :test
        print(io, "CVTestResults(:test, test_fold=", r.test_fold, ", mean_cv=", _fmt(r.mean_cv_loss), ", test=", _fmt(r.test_loss), ")")
    else
        print(io, "CVTestResults(:cv, mean_cv=", _fmt(r.mean_cv_loss), ")")
    end
end

# =============================================================================
# Fold / option resolution
# =============================================================================

function _resolve_folds(data, folds; k::Int, shuffle::Bool)
    folds === nothing || return folds isa Symbol ?
        (data isa DataFrame ? Vector{Int}(data[!, folds]) :
         throw(ArgumentError("`folds` as a Symbol requires a DataFrame."))) :
        Vector{Int}(collect(folds))
    data isa DataFrame || throw(ArgumentError("Provide `folds` when `data` is not a DataFrame."))
    return make_folds(data; k = k, shuffle = shuffle)
end

function _validate_folds(folds, k::Int)
    sort(unique(folds)) == collect(1:k) || throw(ArgumentError(
        "Fold labels must be exactly 1:$k (got $(sort(unique(folds))))."
    ))
end

_resolve_test_fold(::Nothing, ::Int) = nothing
_resolve_test_fold(t::Integer, k::Int) = (1 ≤ t ≤ k) ? Int(t) :
    throw(ArgumentError("test_fold=$t is out of range 1:$k."))
_resolve_test_fold(::Val{:random}, k::Int) = rand(1:k)
_resolve_test_fold(::Val{:all}, ::Int) = :all
_resolve_test_fold(t::Symbol, k::Int) = _resolve_test_fold(Val(t), k)
_resolve_test_fold(t, ::Int) =
    throw(ArgumentError("test_fold must be nothing, an Int in 1:k, :random, or :all; got $t."))

function _resolve_parallel(parallel::Symbol)
    parallel in (:none, :hyper, :folds, :auto) ||
        throw(ArgumentError("`parallel` must be :auto, :none, :hyper, or :folds; got :$parallel."))
    return parallel
end

_parallel_flags(p::Symbol) = (p === :hyper, p === :folds)
_fold_job_count(resolved, k::Int) = resolved isa Integer ? max(k - 1, 1) : k

"""Pick `:hyper`/`:folds`/`:none` by which loop is larger; print the choice."""
function _auto_parallel(fold_jobs::Int, hyper_jobs::Int)
    choice, why = if Threads.nthreads() == 1
        :none, "1 Julia thread (start with `-t auto`)"
    elseif max(fold_jobs, hyper_jobs) <= 1
        :none, "nothing to parallelize (fold_jobs=$fold_jobs, nhyper=$hyper_jobs)"
    elseif hyper_jobs >= fold_jobs
        :hyper, "nhyper=$hyper_jobs ≥ fold_jobs=$fold_jobs"
    else
        :folds, "fold_jobs=$fold_jobs > nhyper=$hyper_jobs"
    end
    @info "parallel=:auto → :$choice ($why)"
    return choice
end

# =============================================================================
# Small utilities
# =============================================================================

_subset_obs(data::DataFrame, idx) = data[idx, :]

"""Drop `test_fold`, keep remaining obs, and relabel remaining folds to 1:k-1."""
function _partition_cv_test(data, folds, test_fold::Int)
    cv_idx = findall(!=(test_fold), folds)
    raw = folds[cv_idx]
    remap = Dict(u => i for (i, u) in enumerate(sort(unique(raw))))
    return _subset_obs(data, cv_idx), [remap[f] for f in raw], length(remap)
end

_cv_name(kwargs, parts...) = string(get(Dict{Symbol, Any}(kwargs), :model_name, "cv_test"), parts...)

"""
Resolve the kwargs forwarded to `train`/`tune`. `cv_test` prescribes defaults that
differ from `train` (all overridable except `shuffleobs`, which is always forced
because the split is driven by `folds`):

- `shuffleobs = false` (forced),
- `show_progress = false`, `plotting = false`, `save_training = false`,
- `agg = mean`.
"""
function _train_kwargs(kwargs)
    defaults = (; show_progress = false, plotting = false, save_training = false, agg = mean)
    forced = (; shuffleobs = false)
    return merge(defaults, (; kwargs...), forced)
end

"""Suppress Info/Warn from nested `train`/`split_data` (task-local; thread-safe)."""
_with_quiet(f, quiet::Bool) = quiet ?
    with_logger(() -> f(), ConsoleLogger(stderr, Logging.Error)) : f()

# =============================================================================
# Progress tracker (per-item score of the just-finished job, plus running best)
# =============================================================================

mutable struct _BestTracker
    prog::Union{Progress, Nothing}
    n::Int
    done::Int
    lock::ReentrantLock
    prefix::String
    metric::Symbol
    best_val::Union{Float64, Nothing}
    best_train::Union{Float64, Nothing}
    best_hp::NamedTuple
    cur_item::Union{Int, Nothing}
    cur_val::Union{Float64, Nothing}
    cur_train::Union{Float64, Nothing}
    cur_hp::NamedTuple
end

function _BestTracker(n::Int; desc::String, metric::Symbol, enabled::Bool)
    prog = enabled ? Progress(n; desc = desc, showspeed = true) : nothing
    return _BestTracker(prog, n, 0, ReentrantLock(), desc, metric,
        nothing, nothing, NamedTuple(), nothing, nothing, nothing, NamedTuple())
end

function _best_status(t::_BestTracker)
    id = t.cur_item === nothing ? "" : " fold=$(t.cur_item)"
    return "$(t.prefix) [$(t.done)/$(t.n)]$id " *
           "train=$(_fmt(t.cur_train)) val=$(_fmt(t.cur_val)) " *
           "(best_val=$(_fmt(t.best_val))) hp=$(_fmt_hyper(t.cur_hp))"
end

"""Record a finished trial/fold; display its own score and keep the running best."""
function _done!(t::Union{_BestTracker, Nothing}, val::Real, train, hp::NamedTuple = NamedTuple();
        item::Union{Int, Nothing} = nothing)
    t === nothing && return nothing
    lock(t.lock) do
        t.done += 1
        v = Float64(val)
        t.cur_item = item
        t.cur_val = isfinite(v) ? v : nothing
        t.cur_train = train === nothing ? nothing : Float64(train)
        t.cur_hp = hp
        if isfinite(v) && (t.best_val === nothing || isbetter(v, t.best_val, t.metric))
            t.best_val = v
            t.best_train = train === nothing ? nothing : Float64(train)
            t.best_hp = hp
        end
        status = _best_status(t)
        t.prog === nothing ? (@info status) : (next!(t.prog); t.prog.desc = status)
    end
    return nothing
end

"""Mean train / val across successful fold `TrainResults` (same snapshot)."""
function _mean_train_val(fold_results, agg)
    trains = Float64[]
    vals = Float64[]
    for out in fold_results
        out === nothing && continue
        tr, vl = _train_val_at_best(out, agg)
        tr !== nothing && isfinite(tr) && push!(trains, Float64(tr))
        isfinite(vl) && push!(vals, Float64(vl))
    end
    return (isempty(trains) ? nothing : mean(trains), isempty(vals) ? nothing : mean(vals))
end

# =============================================================================
# Core CV / hyperopt
# =============================================================================

"""Train one fold and return its `TrainResults` (or `nothing` on failure)."""
function _train_fold(model, data, mspec, hp::NamedTuple; folds, val_fold::Int, label, quiet, kwargs...)
    return _with_quiet(quiet) do
        sdata = split_data(data, model; folds = folds, val_fold = val_fold)
        tune(model, sdata, mspec; hp..., model_name = _cv_name(kwargs, label, "_fold$val_fold"), kwargs...)
    end
end

"""
Run k-fold CV for fixed hyperparameters. Returns `(fold_results, mean_score)`
where `mean_score` uses the natural orientation of the first metric and is
optionally weighted by validation-fold size.
"""
function _run_cv(model, data, mspec, hp::NamedTuple;
        folds, k, parallel_folds::Bool, weighted::Bool,
        label = "", tracker = nothing, quiet = true, kwargs...)
    results = Vector{Union{TrainResults, Nothing}}(undef, k)
    agg = get(Dict{Symbol, Any}(kwargs), :agg, mean)
    do_fold(v) = begin
        out = _train_fold(model, data, mspec, hp; folds, val_fold = v, label, quiet, kwargs...)
        results[v] = out
        if out === nothing
            @warn "CV fold $v (label=$label) returned no result"
        else
            tr, _ = _train_val_at_best(out, agg)
            _done!(tracker, out.best_loss, tr, hp; item = v)
        end
    end
    if parallel_folds
        Threads.@threads for v in 1:k
            do_fold(v)
        end
    else
        for v in 1:k
            do_fold(v)
        end
    end
    scores = Float64[]
    weights = Float64[]
    for v in 1:k
        out = results[v]
        (out === nothing || !isfinite(out.best_loss)) && continue
        push!(scores, Float64(out.best_loss))
        push!(weights, weighted ? Float64(count(==(v), folds)) : 1.0)
    end
    mean_score = isempty(scores) ? NaN : sum(scores .* weights) / sum(weights)
    return results, mean_score
end

_check_sampler(s) = s isa Hyperband && throw(ArgumentError(
    "Hyperband/BOHB need a resource/state API and are not supported in cv_test; " *
    "use RandomSampler(), LHSampler(), or CLHSampler(...)."))

"""Draw all hyperparameter samples up front (mirrors `@thyperopt`)."""
function _draw_samples(nhyper::Int, hyper::NamedTuple, sampler)
    isempty(hyper) && throw(ArgumentError("`hyper` must be a non-empty NamedTuple of candidate vectors."))
    nhyper > 0 || throw(ArgumentError("`nhyper` must be positive; got $nhyper."))
    _check_sampler(sampler)
    ho = Hyperoptimizer(nhyper, sampler; pairs(hyper)...)
    ho.sampler isa Union{LHSampler, CLHSampler} && Hyperopt.init!(ho.sampler, ho)
    hps = Vector{NamedTuple}(undef, nhyper)
    drawn = Vector{Any}(undef, nhyper)
    for i in 1:nhyper
        nt, _ = iterate(ho, i; update_history = false)
        hps[i] = NamedTuple{keys(hyper)}(ntuple(j -> getproperty(nt, keys(hyper)[j]), length(hyper)))
        drawn[i] = [getproperty(nt, p) for p in ho.params]
    end
    return ho, hps, drawn
end

"""
Select hyperparameters (if `hyper !== nothing`) and return the winning fold
results without re-running CV. Direction-aware: works for maximize metrics too.

Returns `(best_hp, ho, fold_results, mean_score)`.
"""
function _select(model, data, mspec; hyper, nhyper, sampler, metric,
        parallel_hyper, parallel_folds, weighted, k, folds, label,
        show_cv_progress, quiet, kwargs...)
    agg = get(Dict{Symbol, Any}(kwargs), :agg, mean)
    if hyper === nothing
        tracker = _BestTracker(k; desc = "CV folds$label", metric, enabled = show_cv_progress)
        fr, sc = _run_cv(model, data, mspec, NamedTuple();
            folds, k, parallel_folds, weighted, label, tracker, quiet, kwargs...)
        return NamedTuple(), nothing, fr, sc
    end

    ho, hps, drawn = _draw_samples(nhyper, hyper, sampler)
    scores = Vector{Float64}(undef, nhyper)
    fold_res = Vector{Any}(undef, nhyper)
    tracker = _BestTracker(nhyper; desc = "Hyperopt$label", metric, enabled = show_cv_progress)
    eval_trial(i) = begin
        fr, sc = _run_cv(model, data, mspec, hps[i];
            folds, k, parallel_folds, weighted, label, tracker = nothing, quiet, kwargs...)
        scores[i] = sc
        fold_res[i] = fr
        mean_train, _ = _mean_train_val(fr, agg)
        _done!(tracker, sc, mean_train, hps[i]; item = i)
    end
    if parallel_hyper
        Threads.@threads for i in 1:nhyper
            eval_trial(i)
        end
    else
        for i in 1:nhyper
            eval_trial(i)
        end
    end

    empty!(ho.history); append!(ho.history, drawn)
    empty!(ho.results); append!(ho.results, scores)
    best = _best_index(scores, metric)
    return hps[best], ho, fold_res[best], scores[best]
end

"""One held-out test evaluation: search + CV on the remaining folds, then test."""
function _cv_test_once(model, data, mspec; folds, k, test_fold::Int, metric, weighted,
        hyper, nhyper, sampler, parallel_hyper, parallel_folds, label = "",
        show_cv_progress, quiet = true, kwargs...)
    k ≥ 3 || throw(ArgumentError("cv_test with a held-out test fold needs k ≥ 3; got k=$k."))
    cv_data, cv_folds, k_cv = _partition_cv_test(data, folds, test_fold)
    # Nested mode already tags the held-out fold via `label` (`_hold<i>`); only add
    # a `_test<i>` tag for the single held-out case where there is no outer label.
    lbl = isempty(label) ? "_test$test_fold" : label

    best_hp, ho, fold_results, mean_cv = _select(model, cv_data, mspec;
        hyper, nhyper, sampler, metric, parallel_hyper, parallel_folds, weighted,
        k = k_cv, folds = cv_folds, label = lbl, show_cv_progress, quiet, kwargs...)

    final = _with_quiet(quiet) do
        tune(model, cv_data, mspec; best_hp..., model_name = _cv_name(kwargs, lbl, "_final"), kwargs...)
    end
    test_loss, test_obs_pred = final === nothing ? (NaN, nothing) :
        _evaluate_on_test(model, data, mspec, best_hp, final; folds, test_fold, kwargs...)

    return CVTestResults(:test, get(Dict{Symbol,Any}(kwargs), :agg, mean), _loss_types(kwargs), folds,
        best_hp, ho, fold_results, mean_cv, test_fold, test_loss, test_obs_pred, final,
        nothing, nothing, nothing, nothing)
end

"""Predict the held-out `test_fold` with `final`; return `(scalar_loss, obs_pred_df)`."""
function _evaluate_on_test(model, data, mspec, hp::NamedTuple, final::TrainResults; folds, test_fold::Int, kwargs...)
    kwargs_model = merge(
        Base.structdiff(to_namedtuple(model), NamedTuple{(:config,)}),
        model.config, (; kwargs...), mspec.hyper_model, hp,
    )
    hm = constructHybridModel(; kwargs_model...)
    train_cfg, _ = EasyHybrid.kwargs_to_configs((), merge((; kwargs...), mspec.hyper_train, hp, (; target_names = hm.targets)))
    (_, _), ((x, forcings), y) = split_data(data, hm; folds, val_fold = test_fold)
    mask, empty_mask = valid_mask(y)
    empty_mask && (@warn "Test fold $test_fold has no valid targets"; return (NaN, nothing))
    l, _, ŷ = evaluate_acc(hm, x, forcings, y, mask, final.ps, final.st,
        train_cfg.loss_types, train_cfg.training_loss, train_cfg.extra_loss, train_cfg.agg)
    return Float64(extract_agg_loss(l, train_cfg.agg)), _obs_pred_df(ŷ, y, hm.targets)
end

_loss_types(kwargs) = Vector{Symbol}(get(Dict{Symbol, Any}(kwargs), :loss_types, [:mse, :r2]))

# =============================================================================
# Public API
# =============================================================================

"""
    cv_test(model, data; k=5, test_fold=nothing, hyper=nothing, nhyper=10,
            sampler=RandomSampler(), parallel=:auto, weighted_cv=false,
            show_cv_progress=true, quiet=true, kwargs...)

Cross-validated training with optional hyperparameter search and optional
held-out testing. Extra `kwargs...` are forwarded to [`train`](@ref)/`tune`.

# Folds
- `k`: number of folds; labels must be exactly `1:k`.
- `folds`: a vector of labels or a DataFrame column name (`Symbol`); if omitted,
  random folds are built with [`make_folds`](@ref).
- `test_fold=nothing`: CV only (`mode=:cv`).
- `test_fold=:random` / `Int`: hold out one fold as test (`mode=:test`, needs `k ≥ 3`).
- `test_fold=:all`: nested CV — every fold is held out as test once (`mode=:nested`).
  Reports `pooled_test_loss`: the metric recomputed once over all held-out test
  predictions (each observation predicted exactly once), plus `pooled_val_loss`.

# Hyperparameter search
Pass candidate vectors in `hyper` (e.g. `hyper=(; opt=[...], input_batchnorm=[true,false])`).
Search minimises/maximises the first `loss_types` metric correctly (e.g. maximises
`:nse`). The winning trial's fold models are reused (no redundant retrain).
Supported samplers: `RandomSampler()` (default), `LHSampler()`, `CLHSampler(dims=...)`.

# Scoring
Scores are the first-metric value aggregated across targets by `agg` (defaults to
`mean` in `cv_test`), in natural orientation. `weighted_cv=true` weights the fold
mean by validation-fold size.

# Prescribed `train` defaults (all overridable except `shuffleobs`)
`cv_test` forwards `kwargs...` to `train`/`tune` but overrides some defaults:
`shuffleobs=false` (forced; the split is driven by `folds`), and, unless you pass
your own, `show_progress=false`, `plotting=false`, `save_training=false`, `agg=mean`.

# Parallelism (one dimension only; nested threading oversubscribes)
- `:auto` (default): choose `:hyper` or `:folds` by the larger job count.
- `:none`, `:hyper` (thread trials), `:folds` (thread folds / held-out runs).
"""
function cv_test(model, data;
    mspec::ModelSpec = ModelSpec(),
    k::Int = 5,
    folds = nothing,
    shuffle::Bool = true,
    test_fold = nothing,
    hyper = nothing,
    nhyper::Int = 10,
    sampler = RandomSampler(),
    parallel::Symbol = :auto,
    weighted_cv::Bool = false,
    show_cv_progress::Bool = true,
    quiet::Bool = true,
    kwargs...,
)
    tkwargs = _train_kwargs(kwargs)
    agg = tkwargs.agg
    loss_types = _loss_types(kwargs)
    metric = _metric(loss_types)

    folds = _resolve_folds(data, folds; k, shuffle)
    _validate_folds(folds, k)
    resolved = _resolve_test_fold(test_fold, k)

    parallel = _resolve_parallel(parallel)
    if parallel === :auto
        parallel = _auto_parallel(_fold_job_count(resolved, k), hyper === nothing ? 0 : nhyper)
    end
    parallel_hyper, parallel_folds = _parallel_flags(parallel)
    parallel_hyper && hyper === nothing && @warn "`parallel=:hyper` has no effect without `hyper`"
    @info "cv_test: k=$k, $(hyper === nothing ? "no search" : "nhyper=$nhyper"), metric=:$metric, parallel=:$parallel"

    once(test_fold_i, label) = _cv_test_once(model, data, mspec;
        folds, k, test_fold = test_fold_i, metric, weighted = weighted_cv,
        hyper, nhyper, sampler, parallel_hyper, parallel_folds,
        label, show_cv_progress, quiet, tkwargs...)

    if resolved === :all
        nest_tracker = _BestTracker(k; desc = "Nested CV", metric, enabled = show_cv_progress)
        held_outs = Vector{CVTestResults}(undef, k)
        do_hold(i) = begin
            held_outs[i] = once(i, "_hold$i")
            r = held_outs[i]
            tr, _ = r.final_train === nothing ? (nothing, nothing) : _train_val_at_best(r.final_train, agg)
            score = r.test_loss === nothing ? r.mean_cv_loss : r.test_loss
            _done!(nest_tracker, score, tr, r.best_hyperparams; item = i)
        end
        if parallel_folds
            Threads.@threads for i in 1:k
                do_hold(i)
            end
        else
            for i in 1:k
                do_hold(i)
            end
        end
        cvs = [r.mean_cv_loss for r in held_outs if isfinite(r.mean_cv_loss)]
        tls = [r.test_loss for r in held_outs if r.test_loss !== nothing && isfinite(r.test_loss)]
        pooled_val = _pooled_metric((r.final_train === nothing ? nothing : r.final_train.val_obs_pred for r in held_outs), metric, agg)
        pooled_test = _pooled_metric((r.test_obs_pred for r in held_outs), metric, agg)
        return CVTestResults(:nested, agg, loss_types, folds, NamedTuple(), nothing, nothing,
            isempty(cvs) ? NaN : mean(cvs), nothing, nothing, nothing, nothing,
            isempty(tls) ? NaN : mean(tls), pooled_val, pooled_test, held_outs)

    elseif resolved isa Integer
        return once(resolved, "")

    else
        best_hp, ho, fold_results, mean_cv = _select(model, data, mspec;
            hyper, nhyper, sampler, metric, parallel_hyper, parallel_folds,
            weighted = weighted_cv, k, folds, label = "", show_cv_progress, quiet, tkwargs...)
        return CVTestResults(:cv, agg, loss_types, folds, best_hp, ho, fold_results,
            mean_cv, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing)
    end
end
