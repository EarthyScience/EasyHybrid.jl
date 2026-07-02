import EasyHybrid: monitorplot, monitorplot!

@recipe MonitorPlot (epochs_range, y_train, y_val) begin
    "Colour for training curves"
    training_color = :grey25
    "Colour for validation curves"
    validation_color = :tomato
    "Training label"
    training_label = "Training"
    "Validation label"
    validation_label = "Validation"
    "Line width for both curves"
    linewidth = 2
    "Whether the monitor data is quantile-based (i.e. contains multiple quantiles per monitor) or scalar-based (one value per monitor)"
    is_quantile = false
    "If `is_quantile` is true, the keys of the quantiles to plot, e.g. `[:q25, :q50, :q75]`. Ignored if `is_quantile` is false."
    quantile_keys = Symbol[]
end

function Makie.plot!(p::MonitorPlot)
    if p.is_quantile[]
        _plot_monitor_quantiles!(p)
    else
        _plot_monitor_lines!(p)
    end
    return p
end

function _plot_monitor_lines!(p)
    Makie.lines!(
        p, p[:epochs_range], p[:y_train];
        color = p.training_color, linewidth = p.linewidth, label = p.training_label
    )
    Makie.lines!(
        p, p[:epochs_range], p[:y_val];
        color = p.validation_color, linewidth = p.linewidth,
        linestyle = :dash, label = p.validation_label
    )
    return p
end

function _plot_monitor_quantiles!(p)
    qkeys = p.quantile_keys[]
    n = length(qkeys)
    mid_idx = something(findfirst(==(:q50), qkeys), n ÷ 2 + 1)

    for (i, qntl) in enumerate(qkeys)
        distance = abs(i - mid_idx)
        alpha = 1.0 - 0.25 * distance
        lw = p.linewidth[] / (1 + 0.5 * distance)

        # Create a plain Observable for this slot's data
        tr_obs = Observable(p[:y_train][][qntl])
        val_obs = Observable(p[:y_val][][qntl])

        # Wire them to update whenever y_train / y_val change
        on(p[:y_train]) do yt
            tr_obs[] = yt[qntl]
        end
        on(p[:y_val]) do yv
            val_obs[] = yv[qntl]
        end

        Makie.lines!(
            p, p[:epochs_range], tr_obs;
            color = (p.training_color[], alpha), linewidth = lw, label = string(qntl)
        )
        Makie.lines!(
            p, p[:epochs_range], val_obs;
            color = (p.validation_color[], alpha), linewidth = lw,
            linestyle = :dash, label = ""
        )
    end
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
