@recipe LossPlot (training_loss, validation_loss) begin
    "Y-axis scale function, e.g. `log10`"
    yscale = identity
    "Number of recent epochs shown in the zoom panel"
    zoom_epochs = 50
    "Y-axis label (set to the loss type name)"
    loss_label = "Loss"
    "Colour for training curves"
    training_color = :grey25
    "Colour for validation curves"
    validation_color = :tomato
    "Line width for both curves"
    linewidth = 2
end

function Makie.plot!(p::LossPlot)
    return p
end
