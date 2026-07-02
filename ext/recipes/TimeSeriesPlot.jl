import EasyHybrid: timeseriesplot, timeseriesplot!

@recipe TimeSeriesPlot (predictions, observations) begin
    "Marker colour for predictions"
    pred_color = :tomato
    "Marker colour for observations"
    obs_color = :grey25
    "Marker size"
    markersize = 6
    "Marker alpha (transparency)"
    alpha = 0.6
end

function _align_timeseries(preds::AbstractArray, obs::AbstractArray)
    if length(preds) != length(obs)
        if ndims(preds) == 2
            batch_size = size(preds, 2)
            if ndims(obs) == 2 && size(obs, 2) == batch_size
                nout = size(obs, 1)
                preds = preds[(end - nout + 1):end, :]
            elseif ndims(obs) == 1 && length(obs) == batch_size
                preds = preds[end:end, :]
            end
        end
    end
    return vec(preds), vec(obs)
end

Makie.convert_arguments(::Type{<:TimeSeriesPlot}, preds::AbstractArray, obs::AbstractArray) = _align_timeseries(preds, obs)

function Makie.plot!(p::TimeSeriesPlot)
    preds = p[:predictions]
    obs = p[:observations]

    Makie.scatter!(
        p, obs;
        color = p.obs_color,
        markersize = p.markersize,
        alpha = p.alpha,
        label = "Observed"
    )

    Makie.scatter!(
        p, preds;
        color = p.pred_color,
        markersize = p.markersize,
        alpha = p.alpha,
        label = "Predicted"
    )

    return p
end

function Makie.update!(plt::TimeSeriesPlot, predictions)
    obs = plt[:observations][]
    preds_aligned, _ = _align_timeseries(predictions, obs)
    Makie.update!(plt, arg1 = preds_aligned)
    return nothing
end

function Makie.update!(plt::TimeSeriesPlot, predictions, observations)
    preds_aligned, obs_aligned = _align_timeseries(predictions, observations)
    Makie.update!(plt, arg1 = preds_aligned, arg2 = obs_aligned)
    return nothing
end
