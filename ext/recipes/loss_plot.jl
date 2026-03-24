"""
    LossPlot

Makie recipe for plotting training and validation loss curves, with an optional
zoomed view of the last `zoom_epochs` epochs.

# Usage
    loss_plot(train_loss, val_loss)
    loss_plot!(ax, train_loss, val_loss)

# Attributes
- `yscale = identity`      : Y-axis scale function, e.g. `log10`
- `zoom_epochs = 50`       : Number of recent epochs shown in the zoom panel
- `loss_label = "Loss"`    : Y-axis label (set to the loss type name)
- `train_color = :grey25`  : Colour for training curves
- `val_color = :tomato`    : Colour for validation curves
- `linewidth = 2`          : Line width for both curves
"""
@recipe LossPlot (train_loss, val_loss) begin
    yscale = identity
    zoom_epochs = 50
    loss_label = "Loss"
    train_color = :grey25
    val_color = :tomato
    linewidth = 2
end

function Makie.plot!(p::LossPlot)
    return p
end
