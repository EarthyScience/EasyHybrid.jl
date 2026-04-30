@recipe MonitorPlot (monitor_name, training_monitor, validation_monitor) begin
    "Colour for training lines"
    training_color = :grey25
    "Colour for validation lines (dashed)"
    validation_color = :tomato
    "Base line width (q50 uses this; q25/q75 use `linewidth / 2`)"
    linewidth = 2
end

function Makie.plot!(p::MonitorPlot)
    return p
end
