"""
Configuration for data preparation and loading.

Controls array types, observation shuffling, data splitting,
cross-validation, and sequence construction for time-series training.

$(TYPEDFIELDS)
"""
@kwdef struct DataConfig
    """
    Output array type for data conversion from DataFrame: `:KeyedArray` (default) or `:DimArray`.
    """
    array_type::Symbol = :KeyedArray

    "Whether to shuffle the training observations. Default: `false`."
    shuffleobs::Bool = false

    """
    Column name or function to split data by ID. Default: `nothing` (no ID-based splitting).
    """
    split_by_id = nothing

    "Fraction of data to use for training when splitting. Default: 0.8."
    split_data_at::Float64 = 0.8

    """
    Vector or column name of fold assignments (1..k), one per sample/column,
    for k-fold cross-validation. Default: `nothing`.
    """
    folds = nothing

    "The validation fold to use when `folds` is provided. Default: `nothing`."
    val_fold = nothing

    """
    Number of input time steps per sequence sample. If `nothing`, no sequencing
    is applied and data is passed as-is. Default: `nothing`.
    """
    sequence_length::Union{Int, Nothing} = nothing

    """
    Number of target time steps per sequence sample (output window size).
    See `split_into_sequences`. Default: 1.
    """
    sequence_output_window::Int = 1

    """
    Stride between consecutive sequence samples (output shift).
    See `split_into_sequences`. Default: 1.
    """
    sequence_output_shift::Int = 1

    """
    Gap between the end of the input window and the end of the output window (lead time).
    See `split_into_sequences`. Default: 1.
    """
    sequence_lead_time::Int = 1

    "Whether to apply batch normalization to the model inputs. Default: `false`."
    input_batchnorm::Bool = false

    """
    Train on lazily loaded (disk-backed) data, reading it block by block instead of
    loading it into memory; see [`LazyHybridData`](@ref). `nothing` (default) detects
    it automatically (`true` for disk-backed arrays such as a `YAXArray` opened from Zarr).
    """
    lazy::Union{Nothing, Bool} = nothing

    """
    Lazy data only: number of observations read per block. Rounded up to a multiple of
    the storage chunk length. `nothing` (default) reads one storage chunk per block.
    """
    lazy_block_size::Union{Nothing, Int} = nothing

    """
    Lazy data only: number of observations, taken from randomly chosen blocks, that are
    shuffled together to build training minibatches. Time-contiguous chunks hold strongly
    autocorrelated observations, so mixing several of them is needed for stable training.
    Larger values approach a global shuffle at the cost of memory. Default: `65536`.
    """
    lazy_shuffle_buffer::Int = 65536

    """
    Lazy data only: read the next block on another thread while the current one is used
    (effective when Julia runs with more than one thread). Default: `true`.
    """
    lazy_prefetch::Bool = true
end
