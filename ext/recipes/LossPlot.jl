import EasyHybrid: lossplot, lossplot!

@recipe LossPlot (epochs_range, training_loss, validation_loss) begin
    "Y-axis scale function, e.g. `log10`"
    yscale = identity
    "Number of recent epochs shown in the zoom panel"
    zoom_epochs = 50
    "Training label"
    training_label = "Training"
    "Validation label"
    validation_label = "Validation"
    "Colour for training curves"
    training_color = :grey25
    "Colour for validation curves"
    validation_color = :tomato
    "Line width for both curves"
    linewidth = 2
end

function Makie.plot!(p::LossPlot)
    Makie.lines!(p, p[:epochs_range], p[:training_loss];
        color = p.training_color,
        linewidth = p.linewidth,
        label = p.training_label
        )
    Makie.lines!(p, p[:epochs_range], p[:validation_loss];
        color = p.validation_color,
        linewidth = p.linewidth,
        label = p.validation_label)
    return p
end

function _lossplot_legend_entries(ax::Makie.Axis, plt::LossPlot)
    loss_plots = collect(plt.plots)
    loss_labels = [plt.training_label[], plt.validation_label[]]

    other_plots = filter(ax.scene.plots) do p
        haskey(p.attributes, :label) &&
        p.label[] != "" &&
        p ∉ loss_plots
    end
    other_labels = String[p.label[] for p in other_plots]

    return [loss_plots; other_plots], [loss_labels; other_labels]
end

function Makie.axislegend(ax::Makie.Axis, plt::LossPlot; kwargs...)
    plots, labels = _lossplot_legend_entries(ax, plt)
    Makie.axislegend(ax, plots, labels; kwargs...)
end

function Makie.Legend(gp::Makie.GridPosition, ax::Makie.Axis, plt::LossPlot; kwargs...)
    plots, labels = _lossplot_legend_entries(ax, plt)
    Makie.Legend(gp, plots, labels; kwargs...)
end