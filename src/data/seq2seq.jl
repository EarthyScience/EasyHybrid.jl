export split_seq2seq

"""
    split_seq2seq(x, forcings, y; enc_window, dec_window, lead_time=1, stride=1)
    split_seq2seq(x, y; enc_window, dec_window, lead_time=1, stride=1)

Slides a window over N-dimensional arrays (where time is the last dimension) to generate 
(Encoder Input, Decoder Input, Target) tuples for Sequence-to-Sequence Modeling.

# Arguments:
- `x`: Historical covariates array `(..., Features, Time)`.
- `forcings`: Known future covariates array `(..., Features, Time)`. Can be omitted if not used.
- `y`: Target array `(..., Features, Time)`.
- `enc_window`: Length of the historical past to feed the Encoder.
- `dec_window`: Length of the future horizon to feed the Decoder and predict.
- `lead_time`: Gap between the end of the encoder window and start of the decoder window. 
               Use `lead_time=1` for Forecasting (predicting next steps).
               Use `lead_time=0` for Regression (predicting concurrently with the last encoder step).
- `stride`: Step size between consecutive sliding windows.

# Returns:
- `enc_x`: Array of shape `(..., Features, enc_window, Batch)`
- `dec_x`: Array of shape `(..., Features, dec_window, Batch)` (or `nothing` if `forcings` is omitted)
- `y_target`: Array of shape `(..., Features, dec_window, Batch)`
"""
function split_seq2seq(
        x::AbstractArray, forcings::AbstractArray, y::AbstractArray;
        enc_window::Int, dec_window::Int, lead_time::Int = 1, stride::Int = 1
    )
    # 1. Verify time dimensions match
    L = size(x)[end]
    @assert size(forcings)[end] == L "Time dimension must match between x and forcings"
    @assert size(y)[end] == L "Time dimension must match between x and y"

    # 2. Calculate valid sliding windows
    start_idxs = 1:stride:(L - enc_window - dec_window - lead_time + 2)
    num_samples = length(start_idxs)
    num_samples ≥ 1 || throw(ArgumentError("no samples with given enc_window/dec_window/stride/lead_time for sequence length $L"))

    # 3. Pre-allocate outputs by injecting `enc_window`/`dec_window` and a `batch` dimension
    enc_x_size = (size(x)[1:(end - 1)]..., enc_window, num_samples)
    dec_x_size = (size(forcings)[1:(end - 1)]..., dec_window, num_samples)
    y_size = (size(y)[1:(end - 1)]..., dec_window, num_samples)

    enc_x = similar(x, Float32, enc_x_size)
    dec_x = similar(forcings, Float32, dec_x_size)
    y_target = similar(y, Float32, y_size)

    # 4. Fill sliding windows dynamically using `selectdim` for N-dimensional safety
    for (b, sx) in enumerate(start_idxs)
        ex = sx + enc_window - 1
        sy = ex + lead_time
        ey = sy + dec_window - 1

        # enc_x[..., :, b] = x[..., sx:ex]
        enc_slice = selectdim(enc_x, ndims(enc_x), b)
        x_slice = selectdim(x, ndims(x), sx:ex)
        copyto!(enc_slice, x_slice)

        # dec_x[..., :, b] = forcings[..., sy:ey]
        dec_slice = selectdim(dec_x, ndims(dec_x), b)
        forcings_slice = selectdim(forcings, ndims(forcings), sy:ey)
        copyto!(dec_slice, forcings_slice)

        # y_target[..., :, b] = y[..., sy:ey]
        y_slice = selectdim(y_target, ndims(y_target), b)
        y_src_slice = selectdim(y, ndims(y), sy:ey)
        copyto!(y_slice, y_src_slice)
    end

    return enc_x, dec_x, y_target
end

function split_seq2seq(
        x::AbstractArray, y::AbstractArray;
        enc_window::Int, dec_window::Int, lead_time::Int = 1, stride::Int = 1
    )
    L = size(x)[end]
    @assert size(y)[end] == L "Time dimension must match between x and y"

    start_idxs = 1:stride:(L - enc_window - dec_window - lead_time + 2)
    num_samples = length(start_idxs)
    num_samples ≥ 1 || throw(ArgumentError("no samples with given enc_window/dec_window/stride/lead_time for sequence length $L"))

    enc_x_size = (size(x)[1:(end - 1)]..., enc_window, num_samples)
    y_size = (size(y)[1:(end - 1)]..., dec_window, num_samples)

    enc_x = similar(x, Float32, enc_x_size)
    y_target = similar(y, Float32, y_size)

    for (b, sx) in enumerate(start_idxs)
        ex = sx + enc_window - 1
        sy = ex + lead_time
        ey = sy + dec_window - 1

        enc_slice = selectdim(enc_x, ndims(enc_x), b)
        x_slice = selectdim(x, ndims(x), sx:ex)
        copyto!(enc_slice, x_slice)

        y_slice = selectdim(y_target, ndims(y_target), b)
        y_src_slice = selectdim(y, ndims(y), sy:ey)
        copyto!(y_slice, y_src_slice)
    end

    return enc_x, nothing, y_target
end
