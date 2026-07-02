import EasyHybrid: predictionplot, predictionplot!

@recipe PredictionPlot (predictions, observations) begin
    "Label shown in the panel header"
    target_name = "target"
    "Maximum scatter points drawn (random subsample when exceeded)"
    maxpoints = 10_000
    "Marker colour for training scatter"
    color = :grey25
    "Marker size"
    markersize = 8
    "Marker alpha (transparency)"
    alpha = 0.6
    "Line style for the 1:1 reference line"
    linestyle = :solid
    "Line width for the 1:1 reference line"
    linewidth = 0.85
end

function _align_predictions(preds::AbstractArray, obs::AbstractArray)
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

Makie.convert_arguments(::Type{<:PredictionPlot}, preds::AbstractArray, obs::AbstractArray) = _align_predictions(preds, obs)

function Makie.plot!(p::PredictionPlot)
    Makie.scatter!(
        p, p[:predictions], p[:observations];
        color = p.color,
        markersize = p.markersize,
        alpha = p.alpha,
    )

    Makie.ablines!(
        p, 0, 1;
        color = :black, linestyle = p.linestyle, linewidth = p.linewidth
    )

    return p
end

function Makie.update!(plt::PredictionPlot, predictions)
    obs = plt[:observations][]
    preds_aligned, _ = _align_predictions(predictions, obs)
    Makie.update!(plt, arg1 = preds_aligned)
    return nothing
end

function Makie.update!(plt::PredictionPlot, predictions, observations)
    preds_aligned, obs_aligned = _align_predictions(predictions, observations)
    Makie.update!(plt, arg1 = preds_aligned, arg2 = obs_aligned)
    return nothing
end
