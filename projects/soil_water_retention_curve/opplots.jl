function opplot(ax, pred, obs, title_prefix)
    ss_res = sum((obs .- pred).^2)
    ss_tot = sum((obs .- mean(obs)).^2)
    modelling_efficiency = 1 - ss_res / ss_tot

    ax.title = "$title_prefix\nModelling Efficiency: $(round(modelling_efficiency, digits=3))"
    ax.xlabel = "Predicted θ"
    ax.ylabel = "Observed θ"
    ax.aspect = 1

    scatter!(ax, pred, obs, color=:purple, alpha=0.6, markersize=8)

    max_val = max(maximum(obs), maximum(pred))
    min_val = min(minimum(obs), minimum(pred))
    lines!(ax, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=1, label="1:1 line")

    axislegend(ax; position=:lt)
end

function opplot!(fig, pred, obs, title_prefix, row::Int, col::Int)
    ax = Makie.Axis(fig[row, col])
    opplot(ax, pred, obs, title_prefix)
    return fig
end

