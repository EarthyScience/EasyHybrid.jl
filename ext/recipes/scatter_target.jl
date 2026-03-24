"""
    ScatterTarget

Makie recipe for a paired train/validation scatter plot for a single model target.

# Attributes
- `target_name = "target"` : Label shown in the panel header
- `maxpoints = 10_000`     : Maximum scatter points drawn (random subsample when exceeded)
- `train_color = :grey25`  : Marker colour for training scatter
- `val_color = :tomato`    : Marker colour for validation scatter
- `markersize = 6`         : Marker size for both panels
- `alpha = 0.6`            : Marker alpha for both panels
- `diagonal_style = :dash` : Line style for the 1:1 reference line
- `axis_padding = 0.1`     : Extra padding added to both sides of the shared axis limits
"""
@recipe ScatterTarget (preds_train, obs_train, preds_val, obs_val) begin
    target_name = "target"
    maxpoints = 10_000
    train_color = :grey25
    val_color = :tomato
    markersize = 6
    alpha = 0.6
    diagonal_style = :dash
    axis_padding = 0.1
end

function Makie.plot!(p::ScatterTarget)
    return p
end
