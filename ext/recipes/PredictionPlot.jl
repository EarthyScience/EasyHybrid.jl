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

function Makie.plot!(p::PredictionPlot)
    Makie.scatter!(p, p[:predictions], p[:observations];
        color = p.color,
        markersize = p.markersize,
        alpha = p.alpha,
        label = "Training"
        )
    # Add a 1:1 reference line
    max_val = max(maximum(p[:observations][]), maximum(p[:predictions][]))
    min_val = min(minimum(p[:observations][]), minimum(p[:predictions][]))
    Makie.lines!(p, [min_val, max_val], [min_val, max_val];
        color = :black, linestyle = p.linestyle, linewidth = p.linewidth
    )
    return p
end
