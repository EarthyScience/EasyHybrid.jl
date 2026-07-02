using EasyHybrid
using GLMakie
using GLMakie.Makie.GeometryBasics: AbstractPoint
using DataStructures: CircularBuffer

epochs = 1:20
# Build fake monitor data matching the expected structure
# Scalar monitors are expected to be tuples of the form (scalar = <vector>,)
training_monitor = (
    loss = (scalar = rand(20) .* 0.5 .+ 0.1,),
    accuracy = (scalar = rand(20) .* 0.3 .+ 0.6,),
)
validation_monitor = (
    loss = (scalar = rand(20) .* 0.5 .+ 0.2,),
    accuracy = (scalar = rand(20) .* 0.3 .+ 0.5,),
)

# Quantile monitors are expected to be tuples of the form (quantile = (q25 = <vector>, q50 = <vector>, q75 = <vector>),)
training_monitor_q = (
    loss = (
        quantile = (
            q25 = rand(20) .* 0.3 .+ 0.05,
            q50 = rand(20) .* 0.3 .+ 0.15,
            q75 = rand(20) .* 0.3 .+ 0.25,
        ),
    ),
)
validation_monitor_q = (
    loss = (
        quantile = (
            q25 = rand(20) .* 0.3 .+ 0.1,
            q50 = rand(20) .* 0.3 .+ 0.2,
            q75 = rand(20) .* 0.3 .+ 0.3,
        ),
    ),
)

# 1. Scalar, standalone figure
fig1, ax1, plt1 = monitorplot(epochs, training_monitor, validation_monitor, :loss; axis = (xlabel = "Epoch", ylabel = "Loss"))
# axislegend(ax1, plt1)
hidespines!(ax1, :r, :t)
Legend(fig1[1, 1, Top()], ax1, plt1; orientation = :horizontal, titleposition = :left)
fig1

# 2. Quantile, standalone figure
fig2, ax2, plt2 = monitorplot(epochs, training_monitor_q, validation_monitor_q, :loss)
# axislegend(ax2, plt2)
hidespines!(ax2, :r, :t)
Legend(fig2[1, 1, Top()], ax2, plt2; orientation = :horizontal)
fig2

# 3. Mutating form + attribute overrides
fig3 = Figure()
ax3 = Axis(fig3[1, 1], title = "Loss (custom style)", xlabel = "Epoch", ylabel = "Loss")
plt3 = monitorplot!(
    ax3, epochs, training_monitor, validation_monitor, :loss;
    training_color = :steelblue,
    validation_color = :crimson,
    linewidth = 3,
    training_label = "Train",
    validation_label = "Val",
)
axislegend(ax3, plt3)
fig3

# 4. Multi-panel figure
fig4 = Figure(size = (900, 400))
for (col, name) in enumerate([:loss, :accuracy])
    ax = Axis(fig4[1, col], title = string(name), xlabel = "Epoch")
    plt = monitorplot!(ax, epochs, training_monitor, validation_monitor, name)
    axislegend(ax, plt)
end
fig4
