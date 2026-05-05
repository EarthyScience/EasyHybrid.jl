@recipe MonitorPlot (epochs_range, training_monitor, validation_monitor, monitor_name) begin
    "Colour for training lines"
    training_color = :grey25
    "Colour for validation lines (dashed)"
    validation_color = :tomato
    training_label = "Training"
    "Validation label"
    validation_label = "Validation"
    "Base line width (q50 uses this; q25/q75 use `linewidth / 2`)"
    linewidth = 2
end

function Makie.plot!(p::MonitorPlot)
    entry = p[:training_monitor][p[:monitor_name]]
    haskey(entry, :quantile) ? _plot_monitor_quantiles!(p) : _plot_monitor_lines!(p)
    return p
end

function _plot_monitor_quantiles!(p)
    monitor_name = p[:monitor_name]
    tr_entry = p[:training_monitor][monitor_name]
    val_entry = p[:validation_monitor][monitor_name]
    labels = keys(tr_entry[:quantile])
    mid = haskey(tr_entry[:quantile], :q50) ? :q50 : labels[length(labels) ÷ 2 + 1]
    for qntl in labels
        lw = qntl == mid ? p.linewidth : p.linewidth / 2
        Makie.lines!(p, p[:epochs_range], tr_entry[:quantile][qntl];
            color = p.training_color, linewidth = lw, label = string(qntl))
        Makie.lines!(p, p[:epochs_range], val_entry[:quantile][qntl];
            color = p.validation_color, linewidth = lw, linestyle = :dash)
    end
    return p
end

function _plot_monitor_lines!(p)
    monitor_name = p[:monitor_name]
    Makie.lines!(p, p[:epochs_range], p[:training_monitor][monitor_name][:scalar];
        color = p.training_color, linewidth = p.linewidth, label = p.training_label)
    Makie.lines!(p, p[:epochs_range], p[:validation_monitor][monitor_name][:scalar];
        color = p.validation_color, linewidth = p.linewidth, linestyle = :dash, label = p.validation_label)
    return p
end
