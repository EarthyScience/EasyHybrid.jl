export split_data

function split_data(data::Tuple{Tuple, Tuple}, hybridModel; kwargs...)
    @warn "data was prepared already, none of the keyword arguments for split_data will be used"
    return data
end

function split_data(
        data::Union{DataFrame, KeyedArray, Tuple, AbstractDimArray},
        hybridModel;
        split_by_id::Union{Nothing, Symbol, AbstractVector} = nothing,
        folds::Union{Nothing, AbstractVector, Symbol} = nothing,
        val_fold::Union{Nothing, Int} = nothing,
        shuffleobs::Bool = false,
        split_data_at::Real = 0.8,
        sequence_kwargs::Union{Nothing, NamedTuple} = nothing,
        array_type::Symbol = :KeyedArray,
        cfg = DataConfig(),
        kwargs...
    )
    data_ = prepare_data(
        hybridModel, data; array_type = array_type,
        drop_missing_rows = (sequence_kwargs === nothing)
    )

    if sequence_kwargs !== nothing
        (x_all, forcings_all), y_all = data_
        sis_default = (; input_window = 10, output_window = 1, output_shift = 1, lead_time = 1)
        sis = merge(sis_default, sequence_kwargs)
        @info "Using split_into_sequences: $sis"
        x_all, y_all = split_into_sequences(x_all, y_all; sis.input_window, sis.output_window, sis.output_shift, sis.lead_time)
        x_all, y_all = filter_sequences(x_all, y_all)
    else
        (x_all, forcings_all), y_all = data_
    end

    if split_by_id !== nothing && folds !== nothing

        throw(ArgumentError("split_by_id and folds are not supported together; do the split when constructing folds"))

    elseif split_by_id !== nothing
        # --- Option A: split by ID ---
        ids = isa(split_by_id, Symbol) ? getbyname(data, split_by_id) : split_by_id
        unique_ids = unique(ids)
        train_ids, val_ids = splitobs(unique_ids; at = split_data_at, shuffle = shuffleobs)
        train_idx = findall(in(train_ids), ids)
        val_idx = findall(in(val_ids), ids)

        @info "Splitting data by $(split_by_id)"
        @info "Number of unique $(split_by_id): $(length(unique_ids))"
        @info "Train IDs: $(length(train_ids)) | Val IDs: $(length(val_ids))"

        x_train, forcings_train, y_train = collect_end_dim(x_all, train_idx), collect_end_dim(forcings_all, train_idx), collect_end_dim(y_all, train_idx)
        x_val, forcings_val, y_val = collect_end_dim(x_all, val_idx), collect_end_dim(forcings_all, val_idx), collect_end_dim(y_all, val_idx)
        return ((x_train, forcings_train), y_train), ((x_val, forcings_val), y_val)

    elseif folds !== nothing || val_fold !== nothing
        # --- Option B: external K-fold assignment ---
        @assert val_fold !== nothing "Provide val_fold when using folds."
        @assert folds !== nothing "Provide folds when using val_fold."
        @warn "shuffleobs is not supported when using folds and val_fold, this will be ignored and should be done during fold constructions"
        f = isa(folds, Symbol) ? getbyname(data, folds) : folds
        n = size(x_all, 2)
        @assert length(f) == n "length(folds) ($(length(f))) must equal number of samples/columns ($n)."
        @assert 1 ≤ val_fold ≤ maximum(f) "val_fold=$val_fold is out of range 1:$(maximum(f))."

        val_idx = findall(==(val_fold), f)
        @assert !isempty(val_idx) "No samples assigned to validation fold $val_fold."
        train_idx = setdiff(1:n, val_idx)

        @info "K-fold via external assignments: val_fold=$val_fold → train=$(length(train_idx)) val=$(length(val_idx))"

        x_train, forcings_train, y_train = collect_end_dim(x_all, train_idx), collect_end_dim(forcings_all, train_idx), collect_end_dim(y_all, train_idx)
        x_val, forcings_val, y_val = collect_end_dim(x_all, val_idx), collect_end_dim(forcings_all, val_idx), collect_end_dim(y_all, val_idx)
        return ((x_train, forcings_train), y_train), ((x_val, forcings_val), y_val)

    else
        # --- Fallback: simple random/chronological split of prepared data ---
        (x_train, forcings_train, y_train), (x_val, forcings_val, y_val) = splitobs((x_all, forcings_all, y_all); at = split_data_at, shuffle = shuffleobs)
        return ((x_train, forcings_train), y_train), ((x_val, forcings_val), y_val)
    end
end


"""
    split_data(data, hybridModel; split_by_id=nothing, shuffleobs=false, split_data_at=0.8, kwargs...)
    split_data(data::Union{DataFrame, KeyedArray}, hybridModel; split_by_id=nothing, shuffleobs=false, split_data_at=0.8, folds=nothing, val_fold=nothing, kwargs...)
    split_data(data::AbstractDimArray, hybridModel; split_by_id=nothing, shuffleobs=false, split_data_at=0.8, kwargs...)
    split_data(data::Tuple, hybridModel; split_by_id=nothing, shuffleobs=false, split_data_at=0.8, kwargs...)
    split_data(data::Tuple{Tuple, Tuple}, hybridModel; kwargs...)

Split data into training and validation sets, either randomly, by grouping by ID, or using external fold assignments.

# Arguments:
- `data`: The data to split, which can be a DataFrame, KeyedArray, AbstractDimArray, or Tuple
- `hybridModel`: The hybrid model object used for data preparation
- `split_by_id=nothing`: Either `nothing` for random splitting, a `Symbol` for column-based splitting, or an `AbstractVector` for custom ID-based splitting
- `shuffleobs=false`: Whether to shuffle observations during splitting
- `split_data_at=0.8`: Ratio of data to use for training
- `folds`: Vector or column name of fold assignments (1..k), one per sample/column for k-fold cross-validation
- `val_fold`: The validation fold to use when `folds` is provided
- `sequence_kwargs=nothing`: NamedTuple of keyword arguments forwarded to `split_into_sequences` (e.g. `(; input_window=10, output_window=1, output_shift=1, lead_time=2)`). When set, data is windowed into 3D sequences before splitting.

# Behavior:
- For DataFrame/KeyedArray: Supports random splitting, ID-based splitting, and external fold assignments
- For AbstractDimArray/Tuple: Random splitting only after data preparation
- For pre-split Tuple{Tuple, Tuple}: Returns input unchanged

# Returns:
- `((x_train, y_train), (x_val, y_val))`: Tuple containing training and validation data pairs
"""
function split_data end

function getbyname(df::DataFrame, name::Symbol)
    return df[!, name]
end

function getbyname(ka::KeyedArray, name::Symbol)
    return ka(variable = name)
end

function getbyname(ka::AbstractDimArray, name::Symbol)
    return ka[variable = At(name)]
end

function view_end_dim(x_all::AbstractMatrix{T}, idx) where {T}
    return view(x_all, :, idx)
end

function view_end_dim(x_all::AbstractVector{T}, idx) where {T}
    return view(x_all, idx)
end

function view_end_dim(x_all::NamedTuple, idx)
    nt = (;)
    for (k, v) in pairs(x_all)
        nt = merge(nt, NamedTuple([k => view_end_dim(v, idx)]))
    end
    return nt
end

function view_end_dim(x_all::Union{KeyedArray{Float32, 2}, AbstractDimArray{Float32, 2}}, idx)
    return view(x_all, :, idx)
end

function view_end_dim(x_all::Union{KeyedArray{Float32, 3}, AbstractDimArray{Float32, 3}}, idx)
    return view(x_all, :, :, idx)
end

function collect_end_dim(x_all::AbstractMatrix{T}, idx) where {T}
    return collect(getindex(x_all, :, idx))
end

function collect_end_dim(x_all::AbstractVector{T}, idx) where {T}
    return collect(getindex(x_all, idx))
end

function collect_end_dim(x_all::NamedTuple, idx)
    nt = (;)
    for (k, v) in pairs(x_all)
        nt = merge(nt, NamedTuple([k => collect_end_dim(v, idx)]))
    end
    return nt
end

function collect_end_dim(x_all::Union{KeyedArray{Float32, 2}, AbstractDimArray{Float32, 2}}, idx)
    return collect(getindex(x_all, :, idx))
end

function collect_end_dim(x_all::Union{KeyedArray{Float32, 3}, AbstractDimArray{Float32, 3}}, idx)
    return collect(getindex(x_all, :, :, idx))
end
