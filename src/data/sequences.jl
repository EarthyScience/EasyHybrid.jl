export split_into_sequences, filter_sequences

"""
    filter_sequences(x, y) -> (x_filtered, y_filtered)

Drop 3rd-dim samples where any predictor is NaN or all targets are NaN.
"""
function filter_sequences(x, y)
    n = size(x, 3)
    valid = findall(ii -> !any(isnan, @view(x[:, :, ii])) && any(!isnan, @view(y[:, :, ii])), 1:n)
    length(valid) < n && @info "Dropped $(n - length(valid)) / $n sequences with NaN predictors or all-NaN targets"
    return x[:, :, valid], y[:, :, valid]
end

"""
    split_into_sequences(x, y; input_window=5, output_window=1, output_shift=1, lead_time=1)

Slide a (input_window + lead_time) window over 2D `(feature, time)` arrays to produce
3D `(feature, time, batch)` tensors for sequence-to-sequence training.

- `input_window`: number of input time steps per sample.
- `output_window`: number of target time steps per sample.
- `output_shift`: stride between consecutive samples.
- `lead_time`: gap between end of input window and end of output window.

Returns `(X, Y)` as KeyedArrays or DimArrays matching the input type.
"""
function split_into_sequences(x, y; input_window = 5, output_window = 1, output_shift = 1, lead_time = 1)
    ndims(x) == 2 || throw(ArgumentError("expected x to be (feature, time); got ndims(x) = $(ndims(x))"))
    ndims(y) == 2 || throw(ArgumentError("expected y to be (target, time); got ndims(y) = $(ndims(y))"))

    Lx, Ly = size(x, 2), size(y, 2)
    Lx == Ly || throw(ArgumentError("x and y must have same time length; got $Lx vs $Ly"))
    lead_time ≥ 0 || throw(ArgumentError("lead_time must be ≥ 0 (0 = instantaneous end)"))

    nfeat, ntarget = size(x, 1), size(y, 1)
    L = Lx

    featkeys = axiskeys(x, 1)
    timekeys = axiskeys(x, 2)
    targetkeys = axiskeys(y, 1)

    lead_start = lead_time - output_window + 1

    lag_keys = Symbol.(["x$(input_window + lead_time - 1)_to_x$(lag)" for lag in (input_window + lead_time - 1):-1:lead_time])
    lead_keys = Symbol.(["_y$(lead)" for lead in ((output_window - 1):-1:0)])
    lead_keys = Symbol.(lag_keys[(end - length(lead_keys) + 1):end], lead_keys)
    lag_keys[(end - length(lead_keys) + 1):end] .= lead_keys

    sx_min = max(1, 1 - (input_window + lead_time - output_window))
    sx_max = L - input_window - lead_time + 1
    sx_min <= sx_max || throw(ArgumentError("windows too long for series length"))

    sx_vals = collect(sx_min:output_shift:sx_max)
    num_samples = length(sx_vals)
    num_samples ≥ 1 || throw(ArgumentError("no samples with given output_shift/windows"))

    samplekeys = timekeys[sx_vals]

    Xd = zeros(Float32, nfeat, input_window, num_samples)
    Yd = zeros(Float32, ntarget, output_window, num_samples)

    @inbounds @views for (ii, sx) in enumerate(sx_vals)
        ex = sx + input_window - 1
        sy = ex + lead_start
        ey = ex + lead_time
        Xd[:, :, ii] .= x[:, sx:ex]
        Yd[:, :, ii] .= y[:, sy:ey]
    end
    if x isa KeyedArray
        Xk = KeyedArray(Xd; variable = featkeys, time = lag_keys, batch_size = samplekeys)
        Yk = KeyedArray(Yd; variable = targetkeys, time = lead_keys, batch_size = samplekeys)
        return Xk, Yk
    elseif x isa AbstractDimArray
        Xk = DimArray(Xd, (variable = featkeys, time = lag_keys, batch_size = samplekeys))
        Yk = DimArray(Yd, (variable = targetkeys, time = lead_keys, batch_size = samplekeys))
        return Xk, Yk
    else
        throw(ArgumentError("expected Xd to be KeyedArray or AbstractDimArray; got $(typeof(Xd))"))
    end
end
