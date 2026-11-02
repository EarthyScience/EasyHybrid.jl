export lossplot, lossplot!
export monitorplot, monitorplot!
export predictionplot, predictionplot!
export timeseriesplot, timeseriesplot!
export train_dashboard, update_step_dashboard!
export build_dashboards, update_step_dashboards!
export combine_flux_partition, plot_flux_partition, plot_flux_partition!

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
Hourly bins group `TIMESTAMP_START` timestamps with [`Dates.floor`](@ref) to the hour.
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

