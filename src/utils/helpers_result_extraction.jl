export extract_histories, extract_parameters, stack_extracts, wide_params, wide_histories, long_histories

"""
    extract_histories(obj; metric::Union{Symbol,AbstractVector{Symbol}},
                          target::Union{Symbol,AbstractVector{Symbol}})

Return both train and validation histories for the given `metric`(s) and `target`(s).

`obj` is assumed to have fields `train_history` and `val_history`.
Returns a `NamedTuple`: `(; train, val)`, where each entry is a `DataFrame`.

- If `metric` is a single `Symbol`, columns are named by `target`s.
- If `metric` is a vector, columns are named `:<metric>_<target>`.
"""
function extract_histories(obj;
    metric::Union{Symbol,AbstractVector{Symbol}},
    target::Union{Symbol,AbstractVector{Symbol}},
)
    metrics = _asvec(metric)
    targets = _asvec(target)

    train = _history_df(obj.train_history, metrics, targets)
    val   = _history_df(obj.val_history,   metrics, targets)
    (; train, val)
end

# history: whatever your history container is
# metrics, targets: AbstractVector{Symbol}
function _history_df(history, metrics::AbstractVector{Symbol}, targets::AbstractVector{Symbol})
    cols  = Any[]
    names = Symbol[]

    # Add epoch column as the first column
    n_epochs = length(history)
    push!(cols, 1:n_epochs)
    push!(names, :epoch)

    single_metric = (length(metrics) == 1)

    for m in metrics, t in targets
        push!(cols, metric_target_history(history, m, t))
        # if there is only one metric, keep the old behaviour:
        # column names are just the targets
        name = single_metric ? t : Symbol(string(m), "_", string(t))
        push!(names, name)
    end

    DataFrame(cols, names)
end


"""
    extract_parameters(nt::NamedTuple)

Convert a `NamedTuple` of vectors and scalars to a `DataFrame`.

- Vector fields with length `N` become columns of length `N`.
- Scalars and length-1 vectors are repeated to length `N`.
- Throws an error if vector lengths are incompatible.
"""
function extract_parameters(nt::NamedTuple)
    vals = values(nt)

    # Determine number of rows: maximum length of all vector-like entries
    lengths = map(vals) do v
        v isa AbstractVector ? length(v) : 1
    end
    nrows = maximum(lengths)

    cols = map(vals) do v
        if v isa AbstractVector
            if length(v) == nrows
                v
            elseif length(v) == 1
                fill(v[1], nrows)
            else
                throw(ArgumentError("Incompatible vector length $(length(v)); expected 1 or $nrows"))
            end
        else
            fill(v, nrows)
        end
    end

    # Build a DataFrame with the same field names as the NamedTuple
    return DataFrame((; zip(keys(nt), cols)...))
end

function extract_parameters(tr::TrainResults)
    train = extract_parameters(tr.train_diffs.parameters)
    val = extract_parameters(tr.val_diffs.parameters)
    (; train, val)
end

"""
    stack_extracts(h; col = :set)

Given a NamedTuple `h` with fields `:train` and `:val` (both `DataFrame`s),
return a single `DataFrame` with an extra column `col` indicating `"train"` vs `"val"`.
"""
function stack_extracts(h::NamedTuple{(:train, :val)}; col::Symbol = :set)
    train = copy(h.train)
    val   = copy(h.val)

    train[!, col] = fill("train", nrow(train))
    val[!, col]   = fill("val",   nrow(val))

    vcat(train, val)
end

function wide_params(tr::TrainResults)
    params = extract_parameters(tr)
    stack_extracts(params)
end

function wide_histories(tr::TrainResults;
    metrics::Union{Symbol,AbstractVector{Symbol}} = :all,
    targets::Union{Symbol,AbstractVector{Symbol}} = :all,
)
    ms = metrics
    ts = targets

    if ms == :all
        ms = collect(keys(tr.train_history[1]))
    end
    if ts == :all
        ts = collect(keys(tr.train_history[1][1]))
    end

    histories = extract_histories(tr, metric = ms, target = ts)
    stack_extracts(histories)
end

"""
    long_histories(tr::TrainResults; metrics=:all, targets=:all)

Return a long-format DataFrame of the training/validation history.
Columns: `epoch`, `set`, `metric`, `target`, `value`.
"""
function long_histories(tr::TrainResults;
    metrics::Union{Symbol,AbstractVector{Symbol}} = :all,
    targets::Union{Symbol,AbstractVector{Symbol}} = :all,
)
    ms = metrics
    ts = targets

    if ms == :all
        ms = collect(keys(tr.train_history[1]))
    end
    if ts == :all
        ts = collect(keys(tr.train_history[1][1]))
    end

    ms = _asvec(ms)
    ts = _asvec(ts)

    function _process(history, set_val)
        dfs = DataFrame[]
        for m in ms, t in ts
            vals = metric_target_history(history, m, t)
            push!(dfs, DataFrame(
                :epoch  => 1:length(vals),
                :set    => set_val,
                :metric => m,
                :target => t,
                :value  => vals
            ))
        end
        vcat(dfs...)
    end

    df_train = _process(tr.train_history, "train")
    df_val   = _process(tr.val_history,   "val")

    vcat(df_train, df_val)
end

_asvec(x::AbstractVector) = x
_asvec(x::Symbol)         = [x]

"""
    metric_target_history(history, metric, target) -> Vector

Return the time series (over epochs) for a given `metric` and `target`.

Assumes `history` is an indexable collection where, for each entry `h`,
`h[metric][target]` is a scalar.
"""
function metric_target_history(history, metric::Symbol, target::Symbol)
    [h[metric][target] for h in history]
end