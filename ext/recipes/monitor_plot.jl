"""
    MonitorPlot

Makie recipe for plotting additional monitored model outputs over training epochs.

# Attributes
- `train_color = :grey25` : Colour for training lines
- `val_color = :tomato`   : Colour for validation lines (dashed)
- `linewidth = 2`         : Base line width (q50 uses this; q25/q75 use `linewidth / 2`)
"""
@recipe MonitorPlot (monitor_name, train_monitor, val_monitor) begin
    train_color = :grey25
    val_color = :tomato
    linewidth = 2
end

function Makie.plot!(p::MonitorPlot)
    return p
end
