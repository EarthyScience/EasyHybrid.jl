export LazyHybridData, LazyBatchLoader, is_lazy_data

"""
    is_lazy_data(data) -> Bool

Return `true` when `data` is a dimensional array whose values live on disk (or in
remote storage) and are only read on demand, e.g. a `YAXArray` opened from a Zarr
or NetCDF store. Such inputs are trained on with [`LazyHybridData`](@ref): data are
read block by block instead of being loaded into memory up front.

Detection relies on the `DiskArrays.jl` package extension (loaded automatically
together with `YAXArrays`, `Zarr` or `NCDatasets`-backed `DiskArrays`). Force or
disable the lazy path with `DataConfig(lazy = true/false)`.
"""
is_lazy_data(data) = false
is_lazy_data(data::AbstractDimArray) = _is_disk_backed(parent(data))

# Overloaded by the DiskArrays extension (`ext/EasyHybridDiskArraysExt.jl`).
_is_disk_backed(x) = false
# Storage chunk length along dimension `d`, or `nothing` when unknown.
_storage_chunk_length(x, d::Int) = nothing
# Start of the first storage chunk relative to index 1 (non-zero e.g. for views that
# begin mid-chunk), so blocks can be aligned with the chunk grid.
_storage_chunk_offset(x, d::Int) = 0

const _DEFAULT_LAZY_BLOCK = 8192

"""
    LazyHybridData(data::AbstractDimArray, model; idx = nothing, block_size = nothing)

Lazy view of a 2-D `(variable × observation)` array (for example a disk-backed
`YAXArray`) restricted to the predictors, forcings and targets of `model`.
Nothing is read at construction time; observations are read in *blocks* that are
aligned with the storage chunks along the observation dimension.

`LazyHybridData` implements `numobs` and `getobs` from `MLUtils.jl`. Each read
returns plain `Float32` arrays in the same layout used for in-memory training:

```julia
((x, forcings), (targets, masks))
```

Rows with a missing (`NaN`/`missing`) predictor or forcing, or without any
observed target, are dropped from every block, matching the behaviour of the
`DataFrame` input path. Note that this makes `getobs(d, i)` return *at most*
`length(i)` observations (an `Integer` index gives a batch of 0 or 1), so generic
`MLUtils.DataLoader`s yield batches of varying size; use [`LazyBatchLoader`](@ref).

# Arguments
- `data`: 2-D `AbstractDimArray` with a `:variable` dimension. Variable names
  are matched against the model names by their string form, so both `Symbol`
  and `String` lookups (as stored in Zarr) work.
- `model`: hybrid model providing `predictors`, `forcing` and `targets`.

# Keywords
- `idx`: observation indices (along the non-variable dimension) included in this
  view. Default: all observations.
- `block_size`: number of observations read per block. It is rounded up to a
  multiple of the storage chunk length. Default: one chunk, or $(_DEFAULT_LAZY_BLOCK)
  observations when the chunking is unknown.
- `prefetch`: read the next block on another thread while the current one is
  being used. Default: `Threads.nthreads() > 1`.
- `shuffle_buffer`: number of observations (from randomly chosen blocks) mixed
  together when building training minibatches. Default: `65536`.

# Example
```julia
using YAXArrays, Zarr
cube = Cube(open_dataset("fluxes.zarr"))      # dims (variable, batch_size), lazy
d = LazyHybridData(cube, hybrid_model)
numobs(d)
((x, forcings), (y, mask)) = getobs(d, 1:64)  # reads only the chunk(s) needed
```
"""
struct LazyHybridData{A <: AbstractArray, P, F <: NamedTuple, T <: NamedTuple, C}
    "Raw 2-D array (unwrapped from its dimensions), possibly disk-backed."
    data::A
    "Dimension of `data` holding the variables (1 or 2)."
    vardim::Int
    "Variable positions in `data` read for every block (sorted, unique)."
    rows::Vector{Int}
    "Predictor rows within a block: `Vector{Int}` or `NamedTuple` of them."
    predictors::P
    "Forcing name => row within a block."
    forcings::F
    "Target name => row within a block."
    targets::T
    "Sorted observation indices covered by this view."
    idx::Vector{Int}
    "Observations per read block (multiple of the storage chunk length)."
    block_span::Int
    "Offset of the storage chunk grid; blocks cover `(i - 1 + block_offset) ÷ block_span`."
    block_offset::Int
    "Read the next block on another thread while the current one is used."
    prefetch::Bool
    "Observations mixed together when shuffling minibatches (see `LazyBatchLoader`)."
    shuffle_buffer::Int
    "Targets of this view, filled by the first full pass (see `observations`)."
    obs_cache::Base.RefValue{C}
end

function LazyHybridData(
        data::AbstractDimArray, model;
        idx = nothing, block_size = nothing, prefetch::Bool = Threads.nthreads() > 1,
        shuffle_buffer::Integer = 65536,
    )
    ndims(data) == 2 || throw(
        ArgumentError(
            "lazy training expects a 2-D (variable × observation) array, got $(ndims(data)) dimensions $(DimensionalData.name(dims(data)))."
        )
    )
    DimensionalData.hasdim(data, :variable) || throw(
        ArgumentError("lazy training expects a `:variable` dimension, got $(DimensionalData.name(dims(data))).")
    )
    vardim = DimensionalData.dimnum(data, :variable)
    obsdim = 3 - vardim
    names = string.(collect(lookup(data, :variable)))

    predictors, forcings, targets = get_prediction_target_names(model)
    function position(name)
        i = findfirst(==(string(name)), names)
        isnothing(i) && throw(ArgumentError("variable `$(name)` not found in the data; available variables: $(names)"))
        return i
    end
    all_pred = predictors isa NamedTuple ? reduce(vcat, collect.(values(predictors))) : collect(predictors)
    rows = sort!(unique!(Int[position(n) for n in [all_pred; forcings; targets]]))
    row_in_block(name) = findfirst(==(position(name)), rows)

    pred_rows = predictors isa NamedTuple ?
        map(p -> Int[row_in_block(n) for n in p], predictors) :
        Int[row_in_block(n) for n in predictors]
    forcing_rows = NamedTuple{Tuple(Symbol.(forcings))}(Tuple(row_in_block.(forcings)))
    target_rows = NamedTuple{Tuple(Symbol.(targets))}(Tuple(row_in_block.(targets)))

    raw = parent(data)
    n = size(raw, obsdim)
    obs_idx = isnothing(idx) ? collect(1:n) : sort!(collect(Int, idx))
    span = _block_span(block_size, _storage_chunk_length(raw, obsdim))
    offset = mod(_storage_chunk_offset(raw, obsdim), span)

    return LazyHybridData(
        raw, vardim, rows, pred_rows, forcing_rows, target_rows,
        obs_idx, span, offset, prefetch, Int(shuffle_buffer), _obs_cache(target_rows)
    )
end

"""
    _lazy_subset(d::LazyHybridData, idx)

Return a new [`LazyHybridData`](@ref) over the same storage restricted to the
observation indices `idx` (positions along the observation dimension of the
underlying array).
"""
function _lazy_subset(d::LazyHybridData, idx)
    return LazyHybridData(
        d.data, d.vardim, d.rows, d.predictors, d.forcings, d.targets,
        sort!(collect(Int, idx)), d.block_span, d.block_offset, d.prefetch, d.shuffle_buffer,
        _obs_cache(d.targets)
    )
end

# Typed cache for the observed targets (`NamedTuple` of `Vector{Float32}`).
_obs_cache(targets::NamedTuple{K}) where {K} =
    Ref{Union{Nothing, NamedTuple{K, NTuple{length(K), Vector{Float32}}}}}(nothing)

function _block_span(block_size, chunk)
    isnothing(block_size) || block_size > 0 ||
        throw(ArgumentError("lazy_block_size must be positive, got $block_size"))
    if isnothing(chunk)
        return isnothing(block_size) ? _DEFAULT_LAZY_BLOCK : block_size
    end
    isnothing(block_size) && return chunk
    return chunk * cld(block_size, chunk)
end

MLUtils.numobs(d::LazyHybridData) = length(d.idx)

function MLUtils.getobs(d::LazyHybridData, i)
    obs = d.idx[i]
    obs isa Integer && (obs = [obs])
    p = sortperm(obs)
    B = _read_filtered(d, obs[p]; reorder = invperm(p))
    return _unpack(d, B)
end

Base.length(d::LazyHybridData) = numobs(d)

function Base.show(io::IO, d::LazyHybridData)
    return print(
        io, "LazyHybridData(", numobs(d), " observations, ", length(d.rows),
        " variables, blocks of ", d.block_span, ", ", typeof(d.data).name.name, ")"
    )
end

# ------------------------------------------------------------------------------
# Reading
# ------------------------------------------------------------------------------

_as_range(v::AbstractUnitRange) = v
function _as_range(v::AbstractVector{<:Integer})
    isempty(v) && return v
    return (last(v) - first(v) + 1 == length(v) && issorted(v) && allunique(v)) ? (first(v):last(v)) : v
end

_to_float32(x::Float32) = x
_to_float32(::Missing) = NaN32
_to_float32(x) = Float32(x)

# Read the requested (sorted) observations as a `rows × n` Float32 matrix.
function _read_block(d::LazyHybridData, obs::AbstractVector{<:Integer})
    r = _as_range(d.rows)
    o = _as_range(obs)
    B = d.vardim == 1 ? d.data[r, o] : permutedims(d.data[o, r])
    return B isa Matrix{Float32} ? B : Matrix{Float32}(_to_float32.(B))
end

_input_rows(p::AbstractVector{Int}, forcings) = unique!([p; collect(Int, values(forcings))])
_input_rows(p::NamedTuple, forcings) = unique!([reduce(vcat, collect(values(p))); collect(Int, values(forcings))])

# Drop observations with a missing input or without any observed target.
function _keep_columns(d::LazyHybridData, B::AbstractMatrix)
    inputs = _input_rows(d.predictors, d.forcings)
    targets = collect(Int, values(d.targets))
    return [
        !any(isnan, view(B, inputs, j)) && any(!isnan, view(B, targets, j))
            for j in axes(B, 2)
    ]
end

function _read_filtered(d::LazyHybridData, obs; reorder = nothing)
    B = _read_block(d, obs)
    isnothing(reorder) || (B = B[:, reorder])
    keep = _keep_columns(d, B)
    return all(keep) ? B : B[:, keep]
end

function _unpack(d::LazyHybridData, B::AbstractMatrix{Float32})
    x = d.predictors isa NamedTuple ?
        map(p -> B[p, :], d.predictors) :
        B[d.predictors, :]
    forcings = map(i -> B[i, :], d.forcings)
    y = map(i -> B[i, :], d.targets)
    mask = map(v -> .!isnan.(v), y)
    return ((x, forcings), (y, mask))
end

# Partition the (sorted) observation indices into blocks that never straddle a
# `block_span`-aligned window of the (regular) chunk grid, so every block touches as
# few storage chunks as possible. Irregular chunk grids are only approximately aligned.
function _blocks(d::LazyHybridData)
    blocks = Vector{Vector{Int}}()
    isempty(d.idx) && return blocks
    span, offset = d.block_span, d.block_offset
    start = 1
    g = (d.idx[1] - 1 + offset) ÷ span
    for k in 2:length(d.idx)
        gk = (d.idx[k] - 1 + offset) ÷ span
        if gk != g
            push!(blocks, d.idx[start:(k - 1)])
            start = k
            g = gk
        end
    end
    push!(blocks, d.idx[start:end])
    return blocks
end

# ------------------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------------------

"""
    LazyBatchLoader(data::LazyHybridData; batchsize, shuffle = true, shuffle_buffer = 65536, rng = Random.default_rng())

Chunk-aware minibatch iterator over a [`LazyHybridData`](@ref). Every epoch it

1. shuffles the order of the storage-aligned blocks (when `shuffle = true`),
2. reads them in groups of `cld(shuffle_buffer, data.block_span)` blocks
   (prefetching the next group on another thread when `data.prefetch` is set),
3. shuffles the observations of the group and yields minibatches of
   `batchsize` observations as `((x, forcings), (targets, masks))`.

Mixing several randomly chosen blocks matters for time series: a single
time-contiguous chunk holds strongly autocorrelated observations, and minibatches
drawn from it alone make training unstable. Larger `shuffle_buffer` values get
closer to a global shuffle at the cost of memory (at most two groups are held at
any time). Leftover observations are carried over to the next group, so all
batches but the last one are full.
"""
struct LazyBatchLoader{D <: LazyHybridData, R <: AbstractRNG}
    data::D
    batchsize::Int
    shuffle::Bool
    "Number of blocks read and shuffled together."
    group::Int
    rng::R
end

function LazyBatchLoader(
        data::LazyHybridData; batchsize::Integer, shuffle::Bool = true,
        shuffle_buffer::Integer = 65536, rng::AbstractRNG = Random.default_rng(),
    )
    batchsize > 0 || throw(ArgumentError("batchsize must be positive, got $batchsize"))
    shuffle_buffer > 0 || throw(ArgumentError("shuffle_buffer must be positive, got $shuffle_buffer"))
    group = shuffle ? max(1, cld(shuffle_buffer, data.block_span)) : 1
    return LazyBatchLoader(data, Int(batchsize), shuffle, group, rng)
end

Base.IteratorSize(::Type{<:LazyBatchLoader}) = Base.SizeUnknown()

# Start reading the group of blocks beginning at block `k`.
function _start_read(l::LazyBatchLoader, blocks, k)
    k > length(blocks) && return nothing
    group = view(blocks, k:min(k + l.group - 1, length(blocks)))
    return l.data.prefetch ? Threads.@spawn(_read_group(l.data, group)) : group
end

_read_group(d::LazyHybridData, group) = reduce(hcat, [_read_filtered(d, obs) for obs in group])

_fetch_block(l::LazyBatchLoader, t::Task) = fetch(t)
_fetch_block(l::LazyBatchLoader, group) = _read_group(l.data, group)

function Base.iterate(l::LazyBatchLoader)
    blocks = _blocks(l.data)
    l.shuffle && Random.shuffle!(l.rng, blocks)
    buffer = Matrix{Float32}(undef, length(l.data.rows), 0)
    return iterate(l, (blocks, 1, buffer, _start_read(l, blocks, 1)))
end

function Base.iterate(l::LazyBatchLoader, (blocks, k, buffer, pending))
    while size(buffer, 2) < l.batchsize && !isnothing(pending)
        B = _fetch_block(l, pending)
        k += l.group
        pending = _start_read(l, blocks, k)
        l.shuffle && (B = B[:, randperm(l.rng, size(B, 2))])
        buffer = isempty(buffer) ? B : hcat(buffer, B)
    end
    nb = min(l.batchsize, size(buffer, 2))
    nb == 0 && return nothing
    batch = buffer[:, 1:nb]
    buffer = buffer[:, (nb + 1):end]
    return _unpack(l.data, batch), (blocks, k, buffer, pending)
end

"""
    lazy_splits(data, model, cfg::DataConfig) -> (train, val)

Split a lazy `(variable × observation)` array into training and validation
[`LazyHybridData`](@ref) views. Like the in-memory path, the split proportions apply
to the *usable* observations: one streaming pass (block by block) drops observations
with a missing input or no observed target before splitting. `split_by_id` and
`folds` given as a variable name read only that variable.
"""
function lazy_splits(data::AbstractDimArray, model, cfg::DataConfig)
    isnothing(cfg.sequence_length) || throw(
        ArgumentError("sequence training (`sequence_length`) is not supported with lazy data yet; load the data into memory or set `lazy = false`.")
    )
    full = LazyHybridData(
        data, model; block_size = cfg.lazy_block_size, shuffle_buffer = cfg.lazy_shuffle_buffer,
        prefetch = cfg.lazy_prefetch && Threads.nthreads() > 1,
    )
    # Split the usable observations only (as the in-memory path does after dropping
    # rows): one streaming pass over the needed variables, keeping a Bool per observation.
    usable = reduce(vcat, [_keep_columns(full, _read_block(full, b)) for b in _blocks(full)]; init = Bool[])
    full = _lazy_subset(full, full.idx[usable])
    ids = _usable_values(data, full, cfg.split_by_id, usable)
    folds = _usable_values(data, full, cfg.folds, usable)
    train_idx, val_idx = split_indices(
        numobs(full); split_by_id = ids, id_name = cfg.split_by_id, folds = folds, val_fold = cfg.val_fold,
        shuffleobs = cfg.shuffleobs, split_data_at = cfg.split_data_at,
    )
    train, val = _lazy_subset(full, full.idx[train_idx]), _lazy_subset(full, full.idx[val_idx])
    @info "Lazy data: $(numobs(train)) training / $(numobs(val)) validation observations, read in blocks of $(full.block_span)"
    return train, val
end

lazy_splits(data, model, cfg::DataConfig) = throw(
    ArgumentError(
        "lazy training expects a 2-D `AbstractDimArray` (e.g. a `YAXArray`) with a `:variable` dimension, got $(typeof(data)). Set `lazy = false` for in-memory data."
    )
)

# Per-observation split labels (a variable name or a vector) restricted to usable observations.
_usable_values(data, d, v::Symbol, usable) = _read_variable(data, d, v)[usable]
_usable_values(data, d, v::AbstractVector, usable) = v[usable]
_usable_values(data, d, v, usable) = v

# Read one whole variable (e.g. site ids or fold assignments) along the observation dim.
function _read_variable(data::AbstractDimArray, d::LazyHybridData, name::Symbol)
    names = string.(collect(lookup(data, :variable)))
    i = findfirst(==(string(name)), names)
    isnothing(i) && throw(ArgumentError("variable `$(name)` not found in the data; available variables: $(names)"))
    v = d.vardim == 1 ? d.data[i, :] : d.data[:, i]
    return collect(v)
end
