export InMemorySplit, prepare_training_splits

"""
    InMemorySplit(x, forcings, y)

Training or validation split whose arrays are fully loaded in memory, as returned by
[`prepare_splits`](@ref): predictors `x`, a `NamedTuple` of `forcings`, a `NamedTuple`
of targets `y` and the corresponding non-`NaN` `mask`.

The lazy counterpart is [`LazyHybridData`](@ref). Both are consumed by the training
loop through `build_loader`, `evaluate_split`, `predict_split` and `observations`.
"""
struct InMemorySplit{X, F, Y, M}
    x::X
    forcings::F
    y::Y
    mask::M
end

InMemorySplit(x, forcings, y) = InMemorySplit(x, forcings, y, first(valid_mask(y)))

"""
    prepare_training_splits(data, model, cfg::DataConfig) -> (train, val)

Prepare the training and validation splits consumed by `train`.

- Lazy (disk-backed) data, see [`is_lazy_data`](@ref) and `DataConfig.lazy`, gives two
  [`LazyHybridData`](@ref) views: nothing is loaded into memory.
- Everything else goes through [`prepare_splits`](@ref) and gives two [`InMemorySplit`](@ref)s.
"""
function prepare_training_splits(data, model, cfg::DataConfig)
    if _use_lazy(data, cfg)
        return lazy_splits(data, model, cfg)
    end
    ((x_train, forcings_train), y_train), ((x_val, forcings_val), y_val) = prepare_splits(data, model, cfg)
    return InMemorySplit(x_train, forcings_train, y_train), InMemorySplit(x_val, forcings_val, y_val)
end

_use_lazy(data, cfg::DataConfig) = isnothing(cfg.lazy) ? _auto_lazy(data, cfg) : cfg.lazy

# Auto mode: only disk-backed data the lazy path supports; anything else (other
# layouts, sequence models) keeps loading into memory as before.
function _auto_lazy(data, cfg::DataConfig)
    is_lazy_data(data) || return false
    supported = ndims(data) == 2 && DimensionalData.hasdim(data, :variable) && isnothing(cfg.sequence_length)
    supported || @info "Disk-backed data with an unsupported layout or `sequence_length` for lazy training; loading it into memory."
    return supported
end

"""
    valid_mask(y) -> (mask, isempty)

Return a `NamedTuple` of `!isnan` masks for every target in `y`, and whether all
targets are missing.
"""
function valid_mask(y)
    nt = (;)
    isempty = true
    for (k, v) in pairs(y)
        k_mask = .!isnan.(v)
        if !all(k_mask .== false)
            isempty = false
        end
        nt = merge(nt, NamedTuple([k => .!isnan.(v)]))
    end
    return nt, isempty
end
