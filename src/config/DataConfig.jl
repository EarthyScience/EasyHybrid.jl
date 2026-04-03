"""
Configuration for data preparation and loading.

Controls array types, observation shuffling, data splitting,
cross-validation, and sequence construction for time-series training.
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

    "Select a gpu_device or default to cpu if none available"
    gdev = gpu_device()

    "Set the `cpu_device`, useful for sending back to the cpu model parameters"
    cdev = cpu_device()
end
