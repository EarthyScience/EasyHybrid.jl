using CairoMakie               # or GLMakie/WGLMakie
using DataFrames, Statistics   # just to prep the bin summaries

# --- compute per-bin summaries -----------------------------------------------
function summarise_bins(df::DataFrame; binw::Real)
    # left-aligned bins: (..., -3.5], (0, 3.5], (3.5, 7.0], ...
    bin_left = floor.(df.MAT ./ binw) .* binw
    df2 = copy(df)
    df2.bin_left  = bin_left
    df2.bin_right = bin_left .+ binw

    g = groupby(df2, :bin_left)
    dfout = DataFrame(
        bin_left  = first.(keys(g)),
        bin_right = [first(gg.bin_right) for gg in g],
        y_med  = [median(gg.Q10) for gg in g],
        y_q25  = [quantile(gg.Q10, 0.25) for gg in g],
        y_q75  = [quantile(gg.Q10, 0.75) for gg in g],
        y_q025 = [quantile(gg.Q10, 0.025) for gg in g],
        y_q975 = [quantile(gg.Q10, 0.975) for gg in g],
    ) 
    sort(dfout, :bin_left)
end

function build_step_data(df_bins)
    x_edges = vcat(df_bins.bin_left, last(df_bins.bin_right))
    y_steps = vcat(df_bins.y_med,   last(df_bins.y_med))
    return x_edges, y_steps
end

function plot_Q10_vs_MAT(dfQ10::DataFrame, binw::Real)
    df_bins = summarise_bins(dfQ10; binw)
    x_edges, y_steps = build_step_data(df_bins)

    fig = Figure()
    ax  = Makie.Axis(fig[1, 1], xlabel="MAT (°C)", ylabel="Q₁₀,hybrid",
               xminorticksvisible=true, yminorticksvisible=true)

    # 95% band as per-bin rectangles
    for r in eachrow(df_bins)
        poly!(ax, Rect(r.bin_left, r.y_q025, r.bin_right - r.bin_left, r.y_q975 - r.y_q025);
              color = (:gray, 0.35), strokecolor=:transparent)
    end

    # 50% band rectangles on top (darker)
    for r in eachrow(df_bins)
        poly!(ax, Rect(r.bin_left, r.y_q25, r.bin_right - r.bin_left, r.y_q75 - r.y_q25);
              color = (:gray, 0.6), strokecolor=:transparent)
    end

    # ladder / step median per bin
    stairs!(ax, x_edges, y_steps; step=:post, linewidth=2)
    scatter!(ax, dfQ10.MAT, dfQ10.Q10; markersize=6)

    xlims!(ax, minimum(df_bins.bin_left), maximum(df_bins.bin_right))
    ylims!(ax, 0, 7)   # adjust if needed

    axislegend(ax,
      [
        PolyElement(color=(:gray, 0.35), strokecolor=:transparent),
        PolyElement(color=(:gray, 0.6),  strokecolor=:transparent),
        LineElement(color=:black, linewidth=2),
        MarkerElement(marker=:circle, markersize=6, color=:black)
      ],
      ["5-95% percentile", "25-75% percentile", "Bin median", "Site Q₁₀"],
      position = :rt
    )

    return fig
end

# Example usage (uncomment and provide dfQ10, binw):
# fig = plot_Q10_vs_MAT(dfQ10, binw)
# display(fig)
