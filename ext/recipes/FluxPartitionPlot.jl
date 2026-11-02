# Partitioned NEE / RECO / GPP with the meteorological drivers.
# Measured NEE is the gap-filled aggregate; the QC observations themselves are not marked.

const _OUTPUT_COLORS = [:purple, :teal, :brown, :magenta, :olivedrab, :gray]
const _SWC_COLORS = [:royalblue, :midnightblue, :steelblue, :navy, :cornflowerblue]

function EasyHybrid.plot_flux_partition(
        raw::DataFrame, outputs::DataFrame...;
        size = (1000, 1060), kwargs...,
    )
    fig = Makie.Figure(; size)
    EasyHybrid.plot_flux_partition!(fig[1, 1], raw, outputs...; kwargs...)
    return fig
end

function EasyHybrid.plot_flux_partition!(
        pos, raw::DataFrame, outputs::DataFrame...;
        scale::Union{Symbol, AbstractString} = :daily,
        labels = nothing,
        swc_caps = (25, 625),
    )
    scale = EasyHybrid._normalize_scale(scale)
    labs = EasyHybrid._partition_labels(length(outputs), labels)
    frame = EasyHybrid.combine_flux_partition(raw, outputs...; scale, labels = labs, swc_caps)
    gl = pos isa Makie.GridLayout ? pos : Makie.GridLayout(pos)
    caps = EasyHybrid._swc_caps(swc_caps)
    present_caps = [cap for cap in caps if EasyHybrid._hascol(frame, EasyHybrid._cap_name(cap))]
    has_drivers = EasyHybrid._hascol(frame, :TA) || EasyHybrid._hascol(frame, :P) || !isempty(present_caps)

    axes = Makie.Axis[]
    push!(axes, _flux_panel!(gl, 1, :NEE, frame, labs))
    push!(axes, _flux_panel!(gl, 2, :RECO, frame, labs))
    if has_drivers
        append!(axes, _driver_panel!(gl, 3, frame, scale, present_caps))
    end
    push!(axes, _flux_panel!(gl, has_drivers ? 4 : 3, :GPP, frame, labs; ticks = true))
    Makie.linkxaxes!(axes...)
    return gl
end

function _flux_series(frame, flux, labels)
    series = NamedTuple{(:column, :label, :color, :interpolate), Tuple{Symbol, String, Symbol, Bool}}[]
    function add(column, label, color; interpolate = false)
        EasyHybrid._hascol(frame, column) || return nothing
        push!(series, (; column = Symbol(column), label = String(label), color = Symbol(color), interpolate))
        return nothing
    end
    if flux === :NEE
        add(:NEE, "measured", :green; interpolate = true)
        add(:NEE_DT, "DT", :orange)
    else
        add(Symbol(flux, :_NT_VUT_USTAR50), "NT", :black)
        add(Symbol(flux, :_DT_VUT_USTAR50), "DT", :orange)
    end
    for (i, label) in enumerate(labels)
        column = i == 1 ? Symbol(flux, :_pred) : Symbol(flux, :_pred_, i)
        add(column, label, _OUTPUT_COLORS[mod1(i, length(_OUTPUT_COLORS))])
    end
    return series
end

function _flux_panel!(gl, row, flux, frame, labels; ticks = false)
    ax = Makie.Axis(
        gl[row, 1];
        ylabel = "$flux [μmol/m²/s]",
        xlabelvisible = false,
        xticklabelsvisible = ticks,
    )
    plots, legend = Any[], String[]
    for series in _flux_series(frame, flux, labels)
        y = collect(Float64, frame[!, series.column])
        series.interpolate && (y = fillgaps(y))
        push!(plots, Makie.lines!(ax, frame.time, y; color = series.color))
        push!(legend, series.label)
    end
    if !isempty(plots)
        Makie.axislegend(
            ax, plots, legend;
            position = :lt, framevisible = false,
            orientation = flux === :NEE ? :horizontal : :vertical,
        )
    end
    return ax
end

function _driver_panel!(gl, row, frame, scale, caps)
    has_ta = EasyHybrid._hascol(frame, :TA)
    has_water = EasyHybrid._hascol(frame, :P) || !isempty(caps)
    axes = Makie.Axis[]
    ta_line = nothing
    if has_ta
        ax_ta = Makie.Axis(
            gl[row, 1];
            ylabel = "TA [°C]", ylabelcolor = :red,
            xlabelvisible = false, xticklabelsvisible = false,
        )
        ta_line = Makie.lines!(ax_ta, frame.time, frame.TA; color = :red)
        push!(axes, ax_ta)
        if !has_water
            Makie.axislegend(ax_ta, [ta_line], ["TA"]; position = :lt, framevisible = false, orientation = :horizontal)
        end
    end
    has_water || return axes

    colors = [_SWC_COLORS[mod1(i, length(_SWC_COLORS))] for i in eachindex(caps)]
    cap_text = EasyHybrid._cap_label.(caps)
    ax_water = Makie.Axis(
        gl[row, 1];
        yaxisposition = has_ta ? :right : :left,
        ylabelcolor = :black,
        ylabel = _water_ylabel(EasyHybrid._hascol(frame, :P) ? EasyHybrid._precip_unit(scale) : nothing, cap_text, colors),
        xlabelvisible = false,
        xticklabelsvisible = false,
    )
    if has_ta
        Makie.hidexdecorations!(ax_water)
        Makie.hidespines!(ax_water, :t, :l, :b)
        ax_water.xgridvisible = false
        ax_water.ygridvisible = false
    end
    push!(axes, ax_water)

    legend_plots, legend_labels = Any[], String[]
    limit_values = Float64[]
    if ta_line !== nothing
        push!(legend_plots, ta_line)
        push!(legend_labels, "TA")
    end
    if EasyHybrid._hascol(frame, :P)
        rain = collect(Float64, frame.P)
        rain[.!isfinite.(rain)] .= 0
        push!(
            legend_plots, Makie.barplot!(
                ax_water, frame.time, rain;
                width = EasyHybrid._bar_width(scale), gap = 0,
                color = (:dodgerblue, 0.45),
            )
        )
        push!(legend_labels, "P")
        append!(limit_values, rain)
    end
    for (cap, text, color) in zip(caps, cap_text, colors)
        storage = 100 .* frame[!, EasyHybrid._cap_name(cap)] ./ cap
        push!(legend_plots, Makie.lines!(ax_water, frame.time, storage; color))
        push!(legend_labels, "SWC $(text) mm")
        append!(limit_values, collect(Float64, storage))
    end
    finite = filter(isfinite, limit_values)
    hi = isempty(finite) ? 100.0 : maximum(finite)
    Makie.ylims!(ax_water, -5, max(hi, 100) + 25)
    Makie.axislegend(
        ax_water, legend_plots, legend_labels;
        position = :lt, framevisible = false, orientation = :horizontal,
    )
    return axes
end

function _water_ylabel(unit, caps, colors)
    pieces = Any[]
    if !isnothing(unit)
        push!(pieces, Makie.rich(isempty(caps) ? "P [$unit]" : "P [$unit], "; color = :dodgerblue))
    end
    if !isempty(caps)
        push!(pieces, "SWC in ")
        for (i, cap) in enumerate(caps)
            i == 1 || push!(pieces, "/")
            push!(pieces, Makie.rich(cap, color = colors[i]))
        end
        push!(pieces, " mm [%]")
    end
    return Makie.rich(pieces...)
end
