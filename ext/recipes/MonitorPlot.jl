import EasyHybrid: monitorplot, monitorplot!

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
    training_monitor = to_value(p[:training_monitor])
    monitor_name = to_value(p[:monitor_name])
    entry = training_monitor[monitor_name]
    haskey(entry, :quantile) ? _plot_monitor_quantiles!(p) : _plot_monitor_lines!(p)
    return p
end

function _plot_monitor_quantiles!(p)
    monitor_name = to_value(p[:monitor_name])
    tr_entry = to_value(p[:training_monitor])[monitor_name]
    val_entry = to_value(p[:validation_monitor])[monitor_name]
    labels = keys(tr_entry[:quantile])
    n = length(labels)
    mid_idx = haskey(tr_entry[:quantile], :q50) ? findfirst(==(:q50), labels) : n ÷ 2 + 1
    mid = labels[mid_idx]

    for (i, qntl) in enumerate(labels)
        distance = abs(i - mid_idx)
        alpha = 1.0 - 0.25 * distance
        lw = p.linewidth[] / (1 + 0.5 * distance)
        # label only the training line with the quantile name;
        # validation is implied by the dashed style, no extra legend entry
        tr_label = string(qntl)
        val_label = ""

        Makie.lines!(
            p, p[:epochs_range], tr_entry[:quantile][qntl];
            color = (p.training_color[], alpha), linewidth = lw, label = tr_label
        )
        Makie.lines!(
            p, p[:epochs_range], val_entry[:quantile][qntl];
            color = (p.validation_color[], alpha), linewidth = lw,
            linestyle = :dash, label = val_label
        )
    end
    return p
end

function _plot_monitor_lines!(p)
    monitor_name = to_value(p[:monitor_name])
    tr = to_value(p[:training_monitor])[monitor_name][:scalar]
    val = to_value(p[:validation_monitor])[monitor_name][:scalar]

    Makie.lines!(
        p, p[:epochs_range], tr;
        color = p.training_color, linewidth = p.linewidth, label = p.training_label
    )
    Makie.lines!(
        p, p[:epochs_range], val;
        color = p.validation_color, linewidth = p.linewidth, linestyle = :dash,
        label = p.validation_label
    )
    return p
end

function _legend_entries(ax::Makie.Axis, plt::MonitorPlot)
    child_plots_with_labels = [
        p for p in plt.plots
            if haskey(p.attributes, :label) && p.label[] != ""
    ]
    child_labels = [p.label[] for p in child_plots_with_labels]

    other_plots = filter(ax.scene.plots) do p
        haskey(p.attributes, :label) && p.label[] != "" && p ∉ plt.plots
    end
    other_labels = String[p.label[] for p in other_plots]

    # Only add solid/dashed explanation dummies in the quantile case,
    # where training_label/validation_label are not already in the child labels
    has_quantiles = !any(==(plt.training_label[]), child_labels)
    extras = has_quantiles ? [
            LineElement(color = plt.training_color[], linestyle = :solid),
            LineElement(color = plt.validation_color[], linestyle = :dash),
        ] : []
    extra_labels = has_quantiles ? [plt.training_label[], plt.validation_label[]] : String[]

    return (
        plots = [child_plots_with_labels; other_plots],
        labels = [child_labels; other_labels],
        extras = extras,
        extra_labels = extra_labels,
    )
end

function Makie.axislegend(ax::Makie.Axis, plt::MonitorPlot; title = nothing, kwargs...)
    leg_entry = _legend_entries(ax, plt)
    return Makie.axislegend(ax, [leg_entry.extras; leg_entry.plots], [leg_entry.extra_labels; leg_entry.labels], title; kwargs...)
end

function Makie.Legend(gp::Makie.GridPosition, ax::Makie.Axis, plt::MonitorPlot; title = nothing, kwargs...)
    leg_entry = _legend_entries(ax, plt)
    return Makie.Legend(gp, [leg_entry.extras; leg_entry.plots], [leg_entry.extra_labels; leg_entry.labels], title; kwargs...)
end

function Makie.update!(plt::MonitorPlot, epochs_range, training_monitor, validation_monitor, monitor_name)
    Makie.update!(
        plt,
        arg1 = epochs_range,
        arg2 = training_monitor,
        arg3 = validation_monitor,
        arg4 = monitor_name,
    )
    return nothing
end
