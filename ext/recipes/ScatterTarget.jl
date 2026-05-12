@recipe ScatterTarget (predictions_training, observations_training, predictions_validation, obs_val) begin
    "Label shown in the panel header"
    target_name = "target"
    "Maximum scatter points drawn (random subsample when exceeded)"
    maxpoints = 10_000
    "Marker colour for training scatter"
    training_color = :grey25
    "Marker colour for validation scatter"
    validation_color = :tomato
    "Marker size for both panels"
    markersize = 6
    "Marker alpha for both panels"
    alpha = 0.6
    "Line style for the 1:1 reference line"
    diagonal_style = :dash
    "Extra padding added to both sides of the shared axis limits"
    axis_padding = 0.1
end

function Makie.plot!(p::ScatterTarget)
    return p
end
