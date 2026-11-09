# Month-faceted mean diurnal cycle. The aggregation and the skill numbers live in
# EasyHybrid; this file draws them once Makie is loaded.

using Dates

const _BAND_COLOR = (
    night = (:steelblue, 0.18),
    day = (:gold, 0.16),
    midday = (:darkorange, 0.32),
)

function _diurnal_month_hours(cycle::DataFrame, month, cols)
    sub = cycle[cycle.Month .== month, :]
    out = DataFrame(hour = 0:23)
    for col in cols
        y = fill(NaN, 24)
        name = string(col)
        for row in eachrow(sub)
            h = Int(row.hour)
            0 <= h <= 23 || continue
            y[h + 1] = row[name]
        end
        out[!, name] = y
    end
    return out
end

function _plotted_yrange(cycle, cols)
    ys = Float64[]
    for col in cols
        name = string(col)
        name in names(cycle) || continue
        append!(ys, filter(isfinite, Float64.(cycle[!, name])))
    end
    isempty(ys) && return (0.0, 1.0, 0.0, 1.0)
    lo, hi = extrema(ys)
    lo == hi && return (lo - 1.0, hi + 1.0, lo, hi)
    pad = 0.08 * (hi - lo)
    return (lo - pad, hi + pad, lo, hi)
end

# Same clock hours as the other fluxes. Night hours stay in the series; the rest
# are missing so the faint full-day line shows through.
function _night_line(y, spans)
    xs = Float64[]
    ys = Float64[]
    drawing = false
    for h in 0:23
        inside = any(spans) do (start, stop)
            start <= h < stop
        end
        valid = inside && isfinite(y[h + 1])
        if !valid
            if drawing
                push!(xs, NaN)
                push!(ys, NaN)
                drawing = false
            end
            continue
        end
        push!(xs, float(h))
        push!(ys, float(y[h + 1]))
        drawing = true
    end
    return xs, ys
end

function _shade_bands!(ax, bands)
    for window in (:day, :night, :midday)
        spans = bands[window]
        isempty(spans) && continue
        Makie.vspan!(
            ax, first.(spans), last.(spans);
            color = _BAND_COLOR[window], strokewidth = 0, strokecolor = :transparent,
        )
    end
    return nothing
end

function _place_window_labels!(ax, skill, month, methods, bands, ymin, ymax)
    isempty(methods) && return nothing
    for window in (:night, :day, :midday)
        x, y, align = if window === :night
            # Left edge at 01:00, on the lowest NEE, extending to the right.
            (1.0, ymin, (:left, :bottom))
        elseif window === :day
            # Far right, just inside the frame, so the box does not cover the curves.
            (23.6, ymin, (:right, :bottom))
        else
            center = EasyHybrid._longest_span_center(bands.midday)
            isnothing(center) && continue
            (center, ymax, (:center, :top))
        end
        note = EasyHybrid._window_annotation(skill, month, methods, window)
        isempty(note) && continue
        Makie.textlabel!(
            ax, x, y;
            text = note,
            space = :data,
            text_align = align,
            justification = :left,
            fontsize = 10,
            font = "DejaVu Sans Mono",
            background_color = (:white, 0.88),
            strokewidth = 0,
            padding = 3,
        )
    end
    return nothing
end

function EasyHybrid.diurnalPlot(
        df::DataFrame;
        yvars = r"^RECO_hyb$|RECO_[ND]T_VUT_USTAR50",
        obs::Symbol = :NEE,
        time::Symbol = :DateTime,
        sw_in::Symbol = :SW_IN,
        sw_pot::Symbol = :SW_IN_POT,
        ylabel = string(obs),
        size = (1680, 1200),
    )
    cols = EasyHybrid._diurnal_yvars(df, yvars)
    cycle = EasyHybrid.diurnal_cycle(df; time)
    months = sort(unique(Int.(cycle.Month)))
    isempty(months) && throw(ArgumentError("diurnalPlot found no month-hour groups"))
    methods = [col for col in cols if col != obs]
    can_score = string(obs) in names(df) && string(sw_in) in names(df) && string(sw_pot) in names(df)
    # Score the half-hourly rows. `cycle` is only the line that is drawn.
    skill = if can_score && !isempty(methods)
        EasyHybrid.diurnal_skill(df; time, obs, methods, sw_in, sw_pot)
    else
        EasyHybrid._empty_diurnal_skill()
    end
    ylo, yhi, ymin, ymax = _plotted_yrange(cycle, cols)

    ncols = 3
    nrows = cld(length(months), ncols)
    fig = Makie.Figure(; size, fontsize = 12)
    Makie.Label(
        fig[0, 1:ncols],
        "Blue is night, gold is day, orange is midday. Numbers are the half-hourly r², bias ratio β, and α.";
        fontsize = 14,
        tellwidth = false,
    )
    handles, labels, axes = Any[], String[], Makie.Axis[]
    for (i, month) in enumerate(months)
        r, c = fldmod1(i, ncols)
        below = i + ncols <= length(months)
        ax = Makie.Axis(
            fig[r, c];
            title = Dates.monthabbr(month),
            xlabel = below ? "" : "hour",
            ylabel = c == 1 ? ylabel : "",
            xticks = 0:6:24,
        )
        below && Makie.hidexdecorations!(ax; grid = false, ticks = false, minorgrid = false, minorticks = false)
        push!(axes, ax)
        bands = EasyHybrid._diurnal_bands(df, month; time, sw_in, sw_pot)
        _shade_bands!(ax, bands)
        hours = _diurnal_month_hours(cycle, month, cols)
        for (j, col) in enumerate(cols)
            color = EasyHybrid._diurnal_color(col, j)
            y = hours[!, string(col)]
            # Full RECO_NT line on the same hours as the other fluxes.
            # Whitened outside the night hours, solid inside them.
            ln = if string(col) == "RECO_NT"
                Makie.lines!(ax, hours.hour, y; color = (:black, 0.18), linewidth = 2)
                xs, ys = _night_line(y, bands.night)
                Makie.lines!(ax, xs, ys; color = :black, linewidth = 2)
            else
                Makie.lines!(ax, hours.hour, y; color, linewidth = 2)
            end
            if i == 1
                push!(handles, ln)
                push!(labels, EasyHybrid._diurnal_label(col))
            end
        end
        Makie.xlims!(ax, 0, 24)
        Makie.ylims!(ax, ylo, yhi)
        _place_window_labels!(ax, skill, month, methods, bands, ymin, ymax)
    end
    Makie.linkaxes!(axes...)
    if !isempty(handles)
        append!(
            handles, [
                Makie.PolyElement(color = _BAND_COLOR.night, strokecolor = :transparent),
                Makie.PolyElement(color = _BAND_COLOR.day, strokecolor = :transparent),
                Makie.PolyElement(color = _BAND_COLOR.midday, strokecolor = :transparent),
            ],
        )
        append!(labels, ["night", "day", "midday"])
        Makie.Legend(fig[1:nrows, ncols + 1], handles, labels; framevisible = false, tellheight = false)
    end
    return fig
end
