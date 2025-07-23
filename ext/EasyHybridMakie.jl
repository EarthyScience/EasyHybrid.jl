module EasyHybridMakie

using EasyHybrid
using Makie
using Makie.Colors
import Makie
import EasyHybrid

include("HybridTheme.jl")

@debug "Extension loaded!"

Makie.convert_single_argument(wt::WrappedTuples) = Matrix(wt)

function Makie.series(wt::WrappedTuples; axislegend = (;) , attributes...)
    data_matrix, merged_attributes = _series(wt, attributes)
    p = Makie.series(data_matrix; merged_attributes...)
    Makie.axislegend(p.axis; merge=true, axislegend...)
    return p
end

function _series(wt::WrappedTuples, attributes)
    data_matrix = Matrix(wt)'
    plot_attributes = Makie.Attributes(;
        labels = string.(keys(wt))
        )
    user_attributes = Makie.Attributes(; attributes...)
    merged_attributes = merge(user_attributes, plot_attributes)
    return data_matrix, merged_attributes
end

# =============================================================================
# Prediction vs Observed Plotting Functions
# =============================================================================

"""
    plot_pred_vs_obs(ax, pred, obs, title_prefix)

Create a scatter plot comparing predicted vs observed values with performance metrics.

# Arguments
- `ax`: Makie axis to plot on
- `pred`: Vector of predicted values
- `obs`: Vector of observed values  
- `title_prefix`: Title prefix for the plot

# Returns
- Updates the axis with the plot and adds modeling efficiency to title
"""
function EasyHybrid.poplot(pred, obs, title_prefix)

    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])

    EasyHybrid.plot_pred_vs_obs!(ax, pred, obs, title_prefix)

    return fig

end

"""
    plot_pred_vs_obs!(fig, pred, obs, title_prefix, row::Int, col::Int)

Add a prediction vs observed plot to a figure at the specified position.

# Arguments
- `fig`: Makie figure to add plot to
- `pred`: Vector of predicted values
- `obs`: Vector of observed values
- `title_prefix`: Title prefix for the plot
- `row`: Row position in figure grid
- `col`: Column position in figure grid

# Returns
- Updated figure with the new plot
"""
function EasyHybrid.poplot!(fig, pred, obs, title_prefix, row::Int, col::Int)
    ax = Makie.Axis(fig[row, col])
    EasyHybrid.plot_pred_vs_obs!(ax, pred, obs, title_prefix)
end

function EasyHybrid.plot_pred_vs_obs!(ax, pred, obs, title_prefix)
    ss_res = sum((obs .- pred).^2)
    ss_tot = sum((obs .- mean(obs)).^2)
    modeling_efficiency = 1 - ss_res / ss_tot

    ax.title = "$title_prefix\nModeling Efficiency: $(round(modeling_efficiency, digits=3))"
    ax.xlabel = "Predicted θ"
    ax.ylabel = "Observed θ"
    ax.aspect = 1

    Makie.scatter!(ax, pred, obs, color=:purple, alpha=0.6, markersize=8)

    max_val = max(maximum(obs), maximum(pred))
    min_val = min(minimum(obs), minimum(pred))
    Makie.lines!(ax, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=1, label="1:1 line")

    Makie.axislegend(ax; position=:lt)
end

function EasyHybrid.plot_loss(loss, yscale)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; yscale=yscale, xlabel = "epoch", ylabel="loss")
    Makie.lines!(ax, loss; color = :grey25,label="Training Loss")
    on(loss) do _
        autolimits!(ax)
    end
    display(fig; title="EasyHybrid.jl", focus_on_show = true)
end

function EasyHybrid.plot_loss!(loss)
    if nameof(Makie.current_backend()) == :WGLMakie # TODO for our CPU cluster - alternatives?
        sleep(2.0) 
    end
    ax = Makie.current_axis()
    Makie.lines!(ax, loss; color = :tomato, label="Validation Loss")
    Makie.axislegend(ax; position=:rt)
end

function EasyHybrid.train_board(train_loss, val_loss, train_pred, train_obs, val_pred, val_obs, yscale)
    
    fig=Makie.Figure(resolution=(1600,900))
    scat1 = Makie.Axis(fig[1, 1:2], title="Training -- y: observed, x modelled")
    scat2 = Makie.Axis(fig[1, 3:4], title="Validation -- y: observed, x: modelled")
    line = Makie.Axis(fig[2, 1:2], yscale=yscale, ylabel="loss")
    lineZoom = Makie.Axis(fig[2,3:4], ylabel="Loss (zoomed on x)")


    Makie.lines!(line, train_loss; color = :grey25, label="Training Loss")
    Makie.lines!(line, val_loss; color = :tomato, label="Validation Loss")
    
    zoomindex = @lift(max(1, length($train_loss) - 100))
    train_loss_zoomed = @lift($train_loss[$zoomindex:end])
    val_loss_zoomed = @lift($val_loss[$zoomindex:end])

    on(train_loss) do _
        autolimits!(line); autolimits!(lineZoom)
    end
    

    Makie.lines!(lineZoom, train_loss_zoomed; color = :grey25, label="Training Loss (log scale)")
    Makie.lines!(lineZoom, val_loss_zoomed; color = :tomato, label="Validation Loss (log scale)")
    
    Makie.scatter!(scat1, train_pred, train_obs, color=:blue, alpha=0.6, markersize=8, label="Training Data")
    Makie.lines!(scat1, @lift(sort($train_obs)), @lift(sort($train_obs)), color=:black, linestyle=:dash, linewidth=1)  # y = x line

    Makie.scatter!(scat2, val_pred, val_obs, color=:red, alpha=0.6, markersize=8, label="Validation Data")
    Makie.lines!(scat2, @lift(sort($val_obs)), @lift(sort($val_obs)), color=:black, linestyle=:dash, linewidth=1)  # y = x line
    #EasyHybrid.plot_pred_vs_obs!(ax2, train_pred[], train_obs[], "Predictions vs Observations")
    on(train_pred) do _
        autolimits!(scat1); autolimits!(scat2)
    end
    display(fig;focus_on_show = true)
end

function EasyHybrid.to_obs(o)
    Makie.Observable(o)
end

function EasyHybrid.to_point2f(i, p)
    Makie.Point2f(i, p)
end

function __init__()
    @debug "setting theme_easy_hybrid"
    # hybrid_latex = merge(theme_easy_hybrid(), theme_latexfonts())
    hybrid_latex = theme_easy_hybrid()
    set_theme!(hybrid_latex)
end

end