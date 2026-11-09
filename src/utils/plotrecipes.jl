export lossplot, lossplot!
export monitorplot, monitorplot!
export predictionplot, predictionplot!
export timeseriesplot, timeseriesplot!
export train_dashboard, update_step_dashboard!
export build_dashboards, update_step_dashboards!
export combine_flux_partition, plot_flux_partition, plot_flux_partition!
export diurnal_cycle, diurnal_skill, diurnalPlot

export cv_scatter

function poplot()
    return @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `poplot` with no arguments!")
end

function cv_scatter(args...; kwargs...)
    return @error("Please load `Makie.jl` (e.g. `using CairoMakie`) before calling `cv_scatter`.")
end

function poplot!()
    return @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `poplot!` with no arguments!")
end

function plot_pred_vs_obs!()
    return @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_pred_vs_obs!` with no arguments!")
end

function train_board()
    return @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_pred_vs_obs!` with no arguments!")
end


function plot_parameters!()
    return @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_parameters!` with no arguments!")
end

function plot_loss!()
    return @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_loss!` with no arguments!")
end

function plot_parameters()
    return @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_parameters` with no arguments!")
end

function plot_training_summary()
    return @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_training_summary` with no arguments!")
end
function update_plotting_observables()
    return @error("Please load `Makie.jl` and then call this function. If Makie is loaded, then you can't call `plot_training_summary` with no arguments!")
end

function plot_loss end
function to_obs end
function to_point2f end
function update_plotting_observables end
function record_history end
function dashboard_figure end
function recordframe! end
function save_fig end
function VideoStream end
function save_video end
function train_dashboard end
function update_step_dashboard! end
function build_dashboards end
function update_step_dashboards! end

"""
    initialize_plotting_observables(init_ŷ_train, init_ŷ_val, y_train, y_val, l_init_train, l_init_val, training_loss, agg, monitor_names, target_names)

Initialize plotting observables for training visualization if the Makie extension is loaded.
"""
function initialize_plotting_observables(init_ŷ_train, init_ŷ_val, y_train, y_val, l_init_train, l_init_val, training_loss, agg, target_names; monitor_names)
    # Initialize loss history observables
    l_value = get_loss_value(l_init_train, training_loss, Symbol("$agg"))
    p = to_point2f(0, l_value)
    train_h_obs = to_obs([p])

    l_value_val = get_loss_value(l_init_val, training_loss, Symbol("$agg"))
    p_val = to_point2f(0, l_value_val)
    val_h_obs = to_obs([p_val])

    # build NamedTuples of Observables for preds and obs
    train_preds = to_obs_tuple(init_ŷ_train, target_names)
    val_preds = to_obs_tuple(init_ŷ_val, target_names)
    train_obs = map(Array, y_train)
    val_obs = map(Array, y_val)

    # --- monitored parameters/state as Observables ---
    train_monitor = !isempty(monitor_names) ? monitor_to_obs(init_ŷ_train, monitor_names) : nothing
    val_monitor = !isempty(monitor_names) ? monitor_to_obs(init_ŷ_val, monitor_names) : nothing

    observables = (; train_h_obs, val_h_obs, train_preds, val_preds, train_monitor, val_monitor)
    observations = (; train_obs, val_obs) # observations

    return (; observables, observations)
end

"""
    get_loss_value(losses, loss_spec, agg)

Extract loss value from losses based on the loss specification type.

# Arguments
- `losses`: NamedTuple containing loss values
- `loss_spec`: Loss specification (Symbol, Function or Tuple)
- `agg`: Aggregation function name as Symbol

# Returns
- Loss value for the specified loss function

# Examples
```julia
# Symbol case
val = get_loss_value(losses, :mse, :sum)

# Function case
custom_loss(ŷ, y) = mean(abs2, ŷ .- y)
val = get_loss_value(losses, custom_loss, :mean)

# Tuple case
weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
val = get_loss_value(losses, (weighted_loss, (0.5,)), :sum)
```
"""
function get_loss_value(losses, loss_spec, agg::Symbol)
    return losses[_loss_name(loss_spec)][agg]
end

function get_loss_entries(losses, loss_spec)
    return losses[_loss_name(loss_spec)]
end

function to_obs_tuple(y, target_names)
    return (; (t => to_obs(vec(getfield(y, t))) for t in target_names)...)
end

function monitor_to_obs(ŷ, monitor_names; cuts = (0.25, 0.5, 0.75))
    return (;
        (
            m => begin
                v = vec(getfield(ŷ, m))
                if length(v) > 1
                    (;
                        (
                            qx_ = Symbol("q$(Int(q * 100))") => to_obs([to_point2f(0, quantile(v, q))])
                                for q in cuts
                        )...,
                    )
                else
                    (; :scalar => to_obs([to_point2f(0, v[1])]))
                end
            end for m in monitor_names
        )...,
    )
end

function get_monitor_values(ŷ, monitor_names; cuts = (0.25, 0.5, 0.75))
    labels = map(q -> Symbol("q$(Int(q * 100))"), cuts)
    return (; (m => _monitor_entry(getfield(ŷ, m), cuts, labels) for m in monitor_names)...)
end

function _monitor_entry(v, cuts, labels)
    v = vec(v)
    if length(v) > 1
        return (; :quantile => (; zip(labels, quantile(v, collect(cuts)))...))
    else
        return (; :scalar => only(v))
    end
end

# for recipes
function lossplot end
function lossplot! end
function monitorplot end
function monitorplot! end
function predictionplot end
function predictionplot! end
function timeseriesplot end
function timeseriesplot! end
#

"""
    combine_flux_partition(raw, outputs...; scale = :daily, labels = nothing, swc_caps = (25, 625)) -> DataFrame

Join a FLUXNET site table `raw` with any number of hybrid-model `outputs` and aggregate
them to one scale.

`raw` is the table from [`load_fluxnet_shuttle`](@ref) or [`flux_features`](@ref). It
must have a sorted `DateTime` column (`TIMESTAMP_START`). Observations are read from
`raw` when present, and from the first output that has them otherwise:

- measured `NEE`, or `NEE_VUT_USTAR50` kept where `NEE_VUT_USTAR50_QC == 0`;
- `TA` or `TA_F`, and `P` or `P_F` (both gap-filled before aggregation);
- `RECO_*` / `GPP_*` reference fluxes;
- `SWC_capa_*` soil-water buckets. Missing buckets are computed with [`soil_bucket`](@ref)
  from half-hourly `P`/`P_F` and `LE`/`LE_F_MDS` for the capacities in `swc_caps`.
  The default `(25, 625)` keeps the 25 mm and 625 mm buckets. `:all` requests every
  depth used by [`flux_features`](@ref) (1, 5, 25, 125 and 625 mm). A single number,
  such as `swc_caps = 25`, requests that depth only.

Each output contributes its `*_pred` columns. The first output keeps those names
(`NEE_pred`, …). Further outputs are renamed `NEE_pred_2`, `NEE_pred_3`, …, in the
order of `labels` (legend text; default `"hybrid"`, `"hybrid 2"`, …). An output with
a `DateTime` column is aligned to `raw`; without one it must have the same row order.

`scale` is `:halfhourly`, `:hourly`, or `:daily` (`:hh`, `:hour`, `:day`, and those
strings are accepted). Fluxes, temperature, and soil water are means of the finite
values in each bin. Precipitation is the sum, so the bin width is the unit of `P`.
Hourly bins group `TIMESTAMP_START` timestamps with `Dates.floor` to the hour.
The aggregated time is the column `time` (`Date` at daily scale, `DateTime` otherwise).

# Example
```julia
daily = combine_flux_partition(raw, pred; scale = :daily)
hourly = combine_flux_partition(raw, pred_a, pred_b; labels = ["hybrid", "refit"], scale = :hourly)
```
"""
function combine_flux_partition(
        raw::DataFrame, outputs::DataFrame...;
        scale::Union{Symbol, AbstractString} = :daily,
        labels = nothing,
        swc_caps = (25, 625),
    )
    scale = _normalize_scale(scale)
    _partition_labels(length(outputs), labels)
    caps = _swc_caps(swc_caps)
    _check_time_axis(raw)
    aligned = map(output -> _align_output(raw.DateTime, output), outputs)
    frames = (raw, aligned...)

    frame = DataFrame(DateTime = raw.DateTime)
    _put!(frame, :NEE, _first_nee(frames))
    _put!(frame, :TA, _filled(_first_series(frames, (:TA, :TA_F))))
    _put!(frame, :P, _filled(_first_series(frames, (:P, :P_F))))
    for col in (:RECO_NT_VUT_USTAR50, :GPP_NT_VUT_USTAR50, :RECO_DT_VUT_USTAR50, :GPP_DT_VUT_USTAR50)
        _put!(frame, col, _first_series(frames, (col,)))
    end
    if _hascol(frame, :RECO_DT_VUT_USTAR50) && _hascol(frame, :GPP_DT_VUT_USTAR50)
        frame.NEE_DT = frame.RECO_DT_VUT_USTAR50 .- frame.GPP_DT_VUT_USTAR50
    end
    for (i, output) in enumerate(aligned)
        for col in names(output)
            endswith(col, "_pred") || continue
            values = _as_float(output[!, col])
            isnothing(values) && continue
            dest = _pred_dest(col, i)
            _hascol(frame, dest) && throw(ArgumentError("column $dest is produced by more than one output"))
            frame[!, dest] = values
        end
    end
    _add_soil_water!(frame, frames, caps)
    return _aggregate_fluxes(frame, scale)
end

"""
    plot_flux_partition(raw, outputs...; scale = :daily, labels = nothing, swc_caps = (25, 625), size = (1000, 1060))
    plot_flux_partition!(grid, raw, outputs...; scale = :daily, labels = nothing, swc_caps = (25, 625))

Draw the partitioned NEE, RECO, and GPP together with temperature, precipitation, and
soil water. `raw` and `outputs` are the inputs of [`combine_flux_partition`](@ref), and
`scale` is `:halfhourly`, `:hourly`, or `:daily`.

NEE, RECO, and GPP each get an axis, with the drivers between respiration and GPP.
Measured NEE is the gap-filled aggregate (a line, without the observation markers).
NT and DT are the FLUXNET reference products. Each output is one more line; `labels`
are the legend entries. Soil water is drawn as storage over bucket capacity, in percent.
`swc_caps` selects those depths: `(25, 625)` by default, `:all` for 1, 5, 25, 125 and
625 mm, or one number for a single depth.
Precipitation is a bar of one bin width, labelled `mm/half-hour`, `mm/h`, or `mm/day`.
The time axis has a tick on the first of each month inside the series. January is
labelled with the year in front of the month (`2018 Jan`); every other month is
the month abbreviation alone.

Load a Makie backend first, for example `using CairoMakie`. The plain method returns
the `Figure`. The `!` method draws into a figure cell or `GridLayout`, so several
partitions can sit in one figure:

```julia
using CairoMakie
fig = plot_flux_partition(raw, pred; scale = :daily)
fig = plot_flux_partition(raw, pred; scale = :daily, swc_caps = :all)
fig = plot_flux_partition(raw, pred; scale = :daily, swc_caps = 25)

board = Figure()
plot_flux_partition!(board[1, 1], raw, pred_a; scale = :hourly)
plot_flux_partition!(board[1, 2], raw, pred_b; labels = ["refit"], scale = :hourly)
```
"""
function plot_flux_partition end

function plot_flux_partition(args...; kwargs...)
    return @error("Please load `Makie.jl` (e.g. `using CairoMakie`) before calling `plot_flux_partition`.")
end

function plot_flux_partition! end

function plot_flux_partition!(args...; kwargs...)
    return @error("Please load `Makie.jl` (e.g. `using CairoMakie`) before calling `plot_flux_partition!`.")
end

_as_datetime(time::DateTime) = time
_as_datetime(time::Dates.Date) = DateTime(time)

# First of each month inside the series. A timestamp later on the 1st still
# keeps that month, so a half-hour starting at 00:30 is ticked at midnight.
function _month_ticks(time)
    isempty(time) && return DateTime[], String[]
    lo_dt = _as_datetime(minimum(time))
    hi_dt = _as_datetime(maximum(time))
    cursor = DateTime(year(lo_dt), Dates.month(lo_dt))
    if cursor < lo_dt && Dates.Date(cursor) != Dates.Date(lo_dt)
        cursor += Dates.Month(1)
    end
    stop = DateTime(year(hi_dt), Dates.month(hi_dt))
    ticks = DateTime[]
    while cursor <= hi_dt && cursor <= stop
        push!(ticks, cursor)
        cursor += Dates.Month(1)
    end
    return ticks, _month_ticklabel.(ticks)
end

function _month_ticklabel(time::DateTime)
    abbr = Dates.monthabbr(Dates.month(time))
    return Dates.month(time) == 1 ? string(year(time), " ", abbr) : abbr
end

function _normalize_scale(scale::Symbol)
    if scale === :halfhourly || scale === :half_hourly || scale === :hh
        return :halfhourly
    elseif scale === :hourly || scale === :hour
        return :hourly
    elseif scale === :daily || scale === :day
        return :daily
    else
        throw(ArgumentError("scale must be :halfhourly, :hourly, or :daily, got :$scale"))
    end
end
_normalize_scale(scale::AbstractString) = _normalize_scale(Symbol(lowercase(scale)))

function _partition_labels(n, labels)
    if isnothing(labels)
        n == 0 && return String[]
        return n == 1 ? ["hybrid"] : ["hybrid"; ["hybrid $i" for i in 2:n]]
    end
    labs = collect(String, labels)
    length(labs) == n || throw(ArgumentError("got $(length(labs)) labels for $n outputs"))
    return labs
end

const _ALL_SWC_CAPS = (1, 5, 25, 125, 625)

function _swc_caps(swc_caps::Union{Symbol, AbstractString})
    Symbol(lowercase(string(swc_caps))) === :all && return collect(_ALL_SWC_CAPS)
    throw(ArgumentError("swc_caps must be positive numbers or :all, got $(repr(swc_caps))"))
end

function _swc_caps(swc_caps::Real)
    _valid_swc_cap(swc_caps) ||
        throw(ArgumentError("swc_caps must be positive numbers or :all, got $(repr(swc_caps))"))
    return [swc_caps]
end

function _swc_caps(swc_caps)
    caps = collect(swc_caps)
    all(_valid_swc_cap, caps) ||
        throw(ArgumentError("swc_caps must be positive numbers or :all, got $(repr(swc_caps))"))
    return caps
end

_valid_swc_cap(cap) = cap isa Real && isfinite(cap) && cap > 0

_cap_label(cap) = isinteger(cap) ? string(round(Int, cap)) : string(cap)
_cap_name(cap) = "SWC_capa_" * _cap_label(cap)

function _check_time_axis(raw)
    "DateTime" in names(raw) || throw(ArgumentError("raw needs a DateTime column"))
    eltype(raw.DateTime) <: Dates.TimeType ||
        throw(ArgumentError("raw.DateTime must be a Dates.TimeType, got $(eltype(raw.DateTime))"))
    issorted(raw.DateTime) || throw(ArgumentError("raw.DateTime must be sorted"))
    return nothing
end

function _align_output(time, output::DataFrame)
    if !("DateTime" in names(output))
        nrow(output) == length(time) || throw(
            ArgumentError(
                "an output without DateTime must have the same number of rows as raw ($(length(time))), got $(nrow(output))",
            )
        )
        return output
    end
    eltype(output.DateTime) == eltype(time) || throw(
        ArgumentError(
            "output DateTime has eltype $(eltype(output.DateTime)), raw has $(eltype(time))",
        )
    )
    allunique(output.DateTime) || throw(ArgumentError("output DateTime values must be unique"))
    joined = DataFrames.leftjoin(DataFrame(DateTime = time), output; on = :DateTime, order = :left)
    nrow(joined) == length(time) || throw(ArgumentError("aligning an output on DateTime changed the number of rows"))
    return joined
end

function _as_float(x)
    eltype(x) <: Union{Missing, Real} || return nothing
    return nan(x)
end

function _first_series(frames, candidates)
    for frame in frames
        for name in candidates
            string(name) in names(frame) || continue
            values = _as_float(frame[!, name])
            isnothing(values) || return values
        end
    end
    return nothing
end

function _first_nee(frames)
    for frame in frames
        if "NEE" in names(frame)
            values = _as_float(frame.NEE)
            isnothing(values) || return values
        elseif "NEE_VUT_USTAR50" in names(frame)
            nee = _as_float(frame.NEE_VUT_USTAR50)
            isnothing(nee) && continue
            "NEE_VUT_USTAR50_QC" in names(frame) || return nee
            keep = coalesce.(frame.NEE_VUT_USTAR50_QC .== 0, false)
            return ifelse.(keep, nee, NaN)
        end
    end
    return nothing
end

_hascol(frame, name) = string(name) in names(frame)

_filled(values) = isnothing(values) ? nothing : fillgaps(values)

function _put!(frame, name, values)
    isnothing(values) && return nothing
    frame[!, name] = values
    return nothing
end

_pred_dest(col::AbstractString, i::Int) = i == 1 ? col : col[1:(end - length("_pred"))] * "_pred_$(i)"

function _is_halfhour(time)
    length(time) < 2 && return true
    step = Statistics.median(Dates.value.(diff(time)))
    return 1_500_000 <= step <= 2_100_000
end

function _add_soil_water!(frame, frames, caps)
    for cap in caps
        _put!(frame, _cap_name(cap), _first_series(frames, (_cap_name(cap),)))
    end
    any(cap -> !_hascol(frame, _cap_name(cap)), caps) || return nothing
    _hascol(frame, :P) && _is_halfhour(frame.DateTime) || return nothing
    latent = _filled(_first_series(frames, (:LE, :LE_F_MDS)))
    isnothing(latent) && return nothing
    for cap in caps
        name = _cap_name(cap)
        _hascol(frame, name) && continue
        frame[!, name] = soil_bucket(frame.P, latent, float(cap))
    end
    return nothing
end

_finite_mean(x) = (v = filter(isfinite, skipmissing(x)); isempty(v) ? NaN : mean(v))
_finite_sum(x) = (v = filter(isfinite, skipmissing(x)); isempty(v) ? NaN : sum(v))

function _bin_time(time, scale::Symbol)
    return if scale === :daily
        Dates.Date(time)
    elseif scale === :hourly
        Dates.floor(time, Dates.Hour)
    else
        Dates.floor(time, Dates.Minute(30))
    end
end

function _precip_unit(scale::Symbol)
    return if scale === :daily
        "mm/day"
    elseif scale === :hourly
        "mm/h"
    else
        "mm/half-hour"
    end
end

function _bar_width(scale::Symbol)
    period = if scale === :daily
        Dates.Day(1)
    elseif scale === :hourly
        Dates.Hour(1)
    else
        Dates.Minute(30)
    end
    return Float64(Dates.value(Dates.Millisecond(period)))
end

function _aggregate_fluxes(frame::DataFrame, scale::Symbol)
    work = DataFrame(time = _bin_time.(frame.DateTime, scale))
    specs = Pair[]
    for col in names(frame)
        col == "DateTime" && continue
        work[!, col] = frame[!, col]
        col == "P" || push!(specs, col => _finite_mean)
    end
    "P" in names(work) && push!(specs, "P" => _finite_sum)
    out = if isempty(specs)
        DataFrame(time = unique(work.time))
    else
        DataFrames.combine(DataFrames.groupby(work, :time), specs...; renamecols = false)
    end
    return DataFrames.sort!(out, :time)
end

"""
    diurnal_cycle(df; time = :DateTime) -> DataFrame

Mean of each numeric column by month and clock hour.

`time` is a `DateTime` column. `Month` is that column's calendar month, or the
integer `Month` column when the frame already has one. The two half-hours of a
clock hour are averaged together. Non-finite values are left out of the mean.
The result is sorted by `Month` and `hour`.

# Example
```julia
cycle = diurnal_cycle(paired)
```
"""
function diurnal_cycle(df::DataFrame; time::Symbol = :DateTime)
    _require_column(df, time)
    value_cols = _diurnal_value_columns(df)
    isempty(value_cols) && throw(ArgumentError("diurnal_cycle found no numeric columns to average"))
    work = DataFrame(Month = _month_numbers(df, time), hour = Dates.hour.(df[!, time]))
    for col in value_cols
        work[!, col] = df[!, col]
    end
    specs = [col => _finite_mean for col in value_cols]
    out = DataFrames.combine(DataFrames.groupby(work, [:Month, :hour]), specs...; renamecols = false)
    return DataFrames.sort!(out, [:Month, :hour])
end

"""
    diurnal_skill(df; time = :DateTime, obs = :NEE, methods, sw_in = :SW_IN, sw_pot = :SW_IN_POT) -> DataFrame

Pearson `r²`, bias ratio, and α of each series in `methods` against `obs`.

Each pair is one row of `df`, a half-hour. The hourly means from [`diurnal_cycle`](@ref)
are not used. Rows are the calendar month (`Month`), the method column name
(`method`), and the window (`window`: `"night"`, `"midday"`, or `"day"`). `r2` is
the square of the Pearson correlation. `beta` is the bias ratio `μ_method / μ_obs`.
`alpha` is `σ_method / σ_obs`, the Kling–Gupta variability ratio. `n` is the number
of finite pairs. A window with fewer than two pairs is `NaN`.

Night is `sw_in <= 0`. Day is `sw_in >= 20`, every daytime half-hour. Midday is
the four half-hours closest to that calendar day's maximum of `sw_pot`. Two
half-hours at the same distance resolve to the earlier one.

# Example
```julia
skill = diurnal_skill(paired; obs = :NEE, methods = [:NEE_hybrid, :NEE_NT, :NEE_DT])
```
"""
function diurnal_skill(
        df::DataFrame;
        time::Symbol = :DateTime,
        obs::Symbol = :NEE,
        methods,
        sw_in::Symbol = :SW_IN,
        sw_pot::Symbol = :SW_IN_POT,
    )
    _require_column(df, time)
    _require_column(df, obs)
    _require_column(df, sw_in)
    _require_column(df, sw_pot)
    method_cols = Symbol.(collect(methods))
    foreach(col -> _require_column(df, col), method_cols)
    months = _month_numbers(df, time)
    windows = _nee_windows(df[!, time], df[!, sw_in], df[!, sw_pot])
    obs_v = Float64.(df[!, obs])
    rows = NamedTuple[]
    for month in sort(unique(months))
        in_month = months .== month
        for method in method_cols
            pred_v = Float64.(df[!, method])
            for window in (:night, :midday, :day)
                skill = _ratio_skill(pred_v[in_month .& windows[window]], obs_v[in_month .& windows[window]])
                push!(
                    rows,
                    (;
                        Month = month,
                        method = string(method),
                        window = string(window),
                        r2 = skill.r2,
                        beta = skill.β,
                        alpha = skill.α,
                        n = skill.n,
                    ),
                )
            end
        end
    end
    return isempty(rows) ? _empty_diurnal_skill() : DataFrame(rows)
end

"""
    diurnalPlot(df; yvars, obs = :NEE, time = :DateTime, sw_in = :SW_IN, sw_pot = :SW_IN_POT, ylabel = string(obs), size = (1680, 1200))

Mean diurnal cycle of `yvars`, one panel per month, with `r²`, the bias ratio, and
`α` annotated on each panel.

The drawn lines are the [`diurnal_cycle`](@ref) means. Night, day, and midday are
shaded. The numbers in each band are that month's [`diurnal_skill`](@ref) on the
half-hours in the band, not on the mean curve: Pearson `r²`, the bias ratio, and
`α` (`σ_method / σ_obs`) for every `yvars` column other than `obs`.
Load a Makie backend first, for example `using CairoMakie`.

`yvars` is a list of column names or a regular expression, as in the Fluxparti
`diurnalPlot`.

# Example
```julia
using CairoMakie
fig = diurnalPlot(paired; yvars = [:NEE, :NEE_hybrid, :NEE_NT, :NEE_DT], obs = :NEE)
```
"""
function diurnalPlot end

function diurnalPlot(args...; kwargs...)
    return @error("Please load `Makie.jl` (e.g. `using CairoMakie`) before calling `diurnalPlot`.")
end

function _require_column(df, name)
    string(name) in names(df) || throw(ArgumentError("column $name is missing"))
    return nothing
end

function _month_numbers(df, time)
    col = "Month" in names(df) ? df.Month : Dates.month.(df[!, time])
    eltype(col) <: Integer || throw(ArgumentError("Month must be an integer month number"))
    return Int.(col)
end

function _diurnal_value_columns(df)
    skip = Set(["Month", "hour"])
    cols = String[]
    for col in names(df)
        col in skip && continue
        eltype(df[!, col]) <: Union{Missing, Real} || continue
        push!(cols, col)
    end
    return cols
end

function _diurnal_yvars(df, yvars::Regex)
    cols = Symbol.(names(df, yvars))
    isempty(cols) && throw(ArgumentError("no columns matched $yvars"))
    return cols
end

function _diurnal_yvars(df, yvars)
    cols = Symbol.(collect(yvars))
    isempty(cols) && throw(ArgumentError("yvars is empty"))
    foreach(col -> _require_column(df, col), cols)
    return cols
end

_diurnal_label(col) = _diurnal_label(string(col))
function _diurnal_label(name::AbstractString)
    name == "NEE" && return "measured"
    name in ("hybrid", "NEE_hybrid", "pred", "NEE_pred") && return "UN"
    name == "RECO_NT" && return "RECO NT"
    startswith(name, "NEE_") && return String(name[(length("NEE_") + 1):end])
    return String(name)
end

function _diurnal_color(col, i)
    label = _diurnal_label(col)
    label == "measured" && return :green
    label == "UN" && return :purple
    label in ("NT", "RECO NT") && return :black
    label == "DT" && return :orange
    palette = [:purple, :teal, :brown, :magenta, :olivedrab, :gray]
    return palette[mod1(i, length(palette))]
end

function _nee_windows(time, sw_in, sw_pot)
    sw = Float64.(sw_in)
    night = isfinite.(sw) .& (sw .<= 0)
    day = isfinite.(sw) .& (sw .>= 20)
    midday = _midday_mask(time, Float64.(sw_pot))
    return (; night, day, midday)
end

function _midday_mask(time, pot)
    mask = falses(length(time))
    dates = Dates.Date.(time)
    for d in unique(dates)
        idx = findall(==(d), dates)
        finite = findall(i -> isfinite(pot[idx[i]]), eachindex(idx))
        isempty(finite) && continue
        peak_local = finite[argmax(pot[idx[finite]])]
        peak_time = time[idx[peak_local]]
        order = sort(finite; by = j -> (abs(Dates.value(time[idx[j]] - peak_time)), time[idx[j]]))
        take = order[1:min(4, length(order))]
        mask[idx[take]] .= true
    end
    return mask
end

function _ratio_skill(pred, obs)
    m = isfinite.(pred) .& isfinite.(obs)
    n = count(m)
    n < 2 && return (; r2 = NaN, β = NaN, α = NaN, n)
    r = loss_fn(pred, obs, m, Val(:pearson))
    β = loss_fn(pred, obs, m, Val(:β))
    α = loss_fn(pred, obs, m, Val(:α))
    return (; r2 = r * r, β, α, n)
end

function _empty_diurnal_skill()
    return DataFrame(
        Month = Int[],
        method = String[],
        window = String[],
        r2 = Float64[],
        beta = Float64[],
        alpha = Float64[],
        n = Int[],
    )
end

_window_title(window) = window === :night ? "night" : window === :midday ? "midday" : "day"

function _fmt_skill_cell(x)
    isfinite(x) || return "    —"
    return @sprintf("%5.2f", x)
end

# Pad by character count. Printf's `%s` width mis-counts `β`.
_gap(s, w) = (n = w - length(string(s)); n > 0 ? " "^n : "")
_pad_left(s, w) = _gap(s, w) * string(s)
_pad_right(s, w) = string(s) * _gap(s, w)

function _score_row(name, r2, beta, alpha; name_w = 4)
    return _pad_right(name, name_w) * " " * _pad_left(r2, 5) * " " * _pad_left(beta, 5) * " " * _pad_left(alpha, 5)
end

function _window_annotation(skill::DataFrame, month, methods, window)
    shown = Tuple{String, String, String, String}[]
    for method in methods
        hit = skill[
            (skill.Month .== month) .& (skill.method .== string(method)) .& (skill.window .== string(window)),
            :,
        ]
        isempty(hit) && continue
        hit.n[1] < 2 && continue
        string(method) == "RECO_NT" && window !== :night && continue
        push!(
            shown,
            (
                _diurnal_label(method),
                _fmt_skill_cell(hit.r2[1]),
                _fmt_skill_cell(hit.beta[1]),
                _fmt_skill_cell(hit.alpha[1]),
            ),
        )
    end
    isempty(shown) && return ""
    name_w = maximum(length(row[1]) for row in shown)
    lines = String[_window_title(window), _score_row("", "r²", "β", "α"; name_w)]
    for (name, r2, beta, alpha) in shown
        push!(lines, _score_row(name, r2, beta, alpha; name_w))
    end
    return join(lines, "\n")
end

function _empty_diurnal_bands()
    return (;
        night = Tuple{Float64, Float64}[],
        day = Tuple{Float64, Float64}[],
        midday = Tuple{Float64, Float64}[],
    )
end

"""
Clock-hour bands for one month. An hour is night, day, or midday when more than
half of its half-hours fall in that window. Midday wins over day. The spans are
`[first hour, hour after the run)`.
"""
function _diurnal_bands(
        df::DataFrame, month::Integer;
        time::Symbol = :DateTime, sw_in::Symbol = :SW_IN, sw_pot::Symbol = :SW_IN_POT,
    )
    if !(string(time) in names(df) && string(sw_in) in names(df) && string(sw_pot) in names(df))
        return _empty_diurnal_bands()
    end
    keep = _month_numbers(df, time) .== month
    any(keep) || return _empty_diurnal_bands()
    sub_time = df[keep, time]
    windows = _nee_windows(sub_time, df[keep, sw_in], df[keep, sw_pot])
    hours = Dates.hour.(sub_time)
    classes = fill(:none, 24)
    for h in 0:23
        sel = hours .== h
        n = count(sel)
        n == 0 && continue
        if count(windows.midday[sel]) / n > 0.5
            classes[h + 1] = :midday
        elseif count(windows.night[sel]) / n > 0.5
            classes[h + 1] = :night
        elseif count(windows.day[sel]) / n > 0.5
            classes[h + 1] = :day
        end
    end
    return (
        night = _hour_spans(classes, :night),
        day = _hour_spans(classes, :day),
        midday = _hour_spans(classes, :midday),
    )
end

function _hour_spans(classes, which)
    spans = Tuple{Float64, Float64}[]
    h = 0
    while h <= 23
        if classes[h + 1] != which
            h += 1
            continue
        end
        start = h
        h += 1
        while h <= 23 && classes[h + 1] == which
            h += 1
        end
        push!(spans, (float(start), float(h)))
    end
    return spans
end

function _longest_span_center(spans)
    isempty(spans) && return nothing
    start, stop = spans[argmax([stop - start for (start, stop) in spans])]
    return (start + stop) / 2
end
