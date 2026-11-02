export nan, fillgaps, moving, moving_slope, slope_at_max_x, soil_bucket, esat, flux_features

"""
    nan(x) -> Vector{Float64}

Convert `x` to `Float64`, replacing `missing` with `NaN`.

# Example
```julia
nan([1, missing, 3])  # [1.0, NaN, 3.0]
```
"""
nan(x) = Float64.(coalesce.(x, NaN))

"""
    fillgaps(x; maxgap = typemax(Int)) -> Vector

Fill non-finite values of `x` by linear interpolation between the neighbouring finite
values; leading and trailing gaps take the first and last finite value. A gap longer
than `maxgap` samples is left missing. Returns a copy.

# Example
```julia
fillgaps([NaN, 1.0, NaN, 3.0, NaN])  # [1.0, 1.0, 2.0, 3.0, 3.0]
```
"""
function fillgaps(x; maxgap::Int = typemax(Int))
    y = copy(x)
    ok = findall(isfinite, y)
    isempty(ok) && return y
    if 0 < ok[1] - 1 <= maxgap
        y[1:(ok[1] - 1)] .= y[ok[1]]
    end
    if 0 < length(y) - ok[end] <= maxgap
        y[(ok[end] + 1):end] .= y[ok[end]]
    end
    for (a, b) in zip(ok[1:(end - 1)], ok[2:end])
        b - a - 1 <= maxgap || continue
        for i in (a + 1):(b - 1)
            y[i] = y[a] + (y[b] - y[a]) * (i - a) / (b - a)
        end
    end
    return y
end

"""
    moving(f, x; window = 49, min_count = 1) -> Vector

Apply `f` to a centered moving window of `window` values of `x`, ignoring non-finite
values; the window is truncated at the edges. Returns `NaN` where a window holds fewer
than `min_count` finite values. The default of 49 half-hours covers about one day.

# Example
```julia
moving(median, NEE_night)
```
"""
function moving(f, x; window = 49, min_count = 1)
    h, n = window ÷ 2, length(x)
    return map(1:n) do i
        v = filter(isfinite, view(x, max(1, i - h):min(n, i + h)))
        length(v) < min_count ? NaN : f(v)
    end
end

"""
    moving_slope(x, y; window = 49, min_range = 50) -> Vector

Least-squares slope of `y` against `x` in a centered moving window of `window` values,
using only pairs where both are finite. Returns `NaN` where a window holds fewer than
four pairs, `x` does not vary, or the range of finite `x` is below `min_range`.

# Example
```julia
moving_slope(SW_IN_lowlight, NEE)  # local light response of NEE
```
"""
function moving_slope(x, y; window = 49, min_range = 50)
    h, n = window ÷ 2, length(x)
    return map(1:n) do i
        ok = filter(j -> isfinite(x[j]) && isfinite(y[j]), max(1, i - h):min(n, i + h))
        length(ok) > 3 || return NaN
        xv = x[ok]
        maximum(xv) - minimum(xv) >= min_range || return NaN
        v = var(xv)
        v > 0 ? Statistics.cov(xv, y[ok]) / v : NaN
    end
end

"""
    slope_at_max_x(x, y; min_x = 200, min_range = 50) -> Float64

Response of `y` at high `x`: `(y[i] - quantile(y, 0.90)) / x[i]`, where `i` is the first
finite pair whose `x` is at or above the 95th percentile. Non-finite pairs are dropped.
Returns `NaN` when fewer than four finite pairs remain, the range of finite `x` is below
`min_range`, or the high-`x` point is below `min_x`. The defaults reject a window that
never reaches a few hundred units of `x`, such as dawn radiation of a few W/m².

This is the inner window statistic of `NEE_RUE_at_max_SW` in [`flux_features`](@ref).

# Example
```julia
slope_at_max_x(SW_IN, NEE)
```
"""
function slope_at_max_x(x, y; min_x = 200, min_range = 50)
    ok = findall(i -> isfinite(x[i]) && isfinite(y[i]), eachindex(x, y))
    length(ok) < 4 && return NaN
    xv, yv = x[ok], y[ok]
    maximum(xv) - minimum(xv) >= min_range || return NaN
    xmax = Statistics.quantile(xv, 0.95)
    idx = findfirst(≥(xmax), xv)
    xv[idx] >= min_x && xv[idx] != 0 || return NaN
    return (yv[idx] - Statistics.quantile(yv, 0.9)) / xv[idx]
end

"""
    soil_bucket(precip, LE, cap) -> Vector

Soil-moisture proxy [mm] from half-hourly precipitation `precip` [mm] and latent heat
flux `LE` [W/m²]: a bucket of capacity `cap` [mm], starting full, filled by
precipitation and emptied by evapotranspiration (latent heat over the half-hour divided
by the latent heat of vaporization, 2.45 MJ/kg).

# Example
```julia
soil_bucket(P_F, LE_F_MDS, 25)
```
"""
function soil_bucket(precip, LE, cap)
    s = cap
    return map(precip, LE) do p, le
        s = clamp(s + p - le * 1800 / 2.45e6, 0, cap)
    end
end

"""
    esat(T) -> Real

Saturation vapour pressure [kPa] at air temperature `T` [°C] (Sonntag 1990).
"""
esat(T) = 0.6112 * exp(17.62 * T / (243.12 + T))

"""
    flux_features(raw::DataFrame; min_x = 200, min_range = 50) -> DataFrame

Engineer the predictors of the NEE flux-partitioning example from half-hourly FLUXNET
data, e.g. the output of [`load_fluxnet_shuttle`](@ref).

`min_x` and `min_range` are passed to [`slope_at_max_x`](@ref), and `min_range` to
[`moving_slope`](@ref). Both are in the units of `SW_IN`. The defaults drop a
light-response slope when the window never reaches a few hundred W/m². `min_x = 0` and
`min_range = 0` keep a slope from every window with four finite pairs.

`raw` needs the columns `DateTime`, `TA_F`, `SW_IN_F`, `SW_IN_POT`, `VPD_F`, `P_F`,
`LE_F_MDS`, `NEE_VUT_USTAR50`, `NEE_VUT_USTAR50_QC` and the NT/DT partitioning results
`RECO_NT_VUT_USTAR50`, `GPP_NT_VUT_USTAR50`, `RECO_DT_VUT_USTAR50`, `GPP_DT_VUT_USTAR50`.

The returned `Float32` columns are
- `NEE`: measured NEE (`NEE_VUT_USTAR50_QC == 0`), `NaN` otherwise;
- `TA`, `SW_IN`, `SW_IN_POT`, `SW_IN_POT_diff`, `RH_F`: meteorology, with `RH_F` derived from
  `VPD_F` [hPa] and [`esat`](@ref);
- `NEE_night_movmean`, `TA_night_movmean`, `NEE_light_night_diff`: daily
  [`moving`](@ref) medians of night-time NEE and temperature and the night–day NEE
  difference;
- `NEE_SWin_lowlight_lm_slope`: daily median of the local light response of NEE below
  300 W/m² ([`moving_slope`](@ref)); a window whose radiation range is below `min_range`
  contributes no slope;
- `NEE_RUE_at_max_SW`: 95-step moving median of [`slope_at_max_x`](@ref) applied to
  `NEE` against `SW_IN` in a 49-step window; the high-radiation point must reach `min_x`;
- `SWC_capa_1` … `SWC_capa_625`: [`soil_bucket`](@ref) soil-moisture proxies with
  capacities of 1, 5, 25, 125 and 625 mm;
- `cosHR`, `sinHR`: diurnal harmonics;
- `sin_1.0_year` … `sin_5.0_year`, `cos_1.0_year` … `cos_5.0_year`: harmonics of
  decimal year (`year + doy/366`) over the length of the record;
- the NT/DT reference fluxes, unchanged.

All columns except `NEE` and the references are gap-filled with [`fillgaps`](@ref).

# Example
```julia
raw = load_fluxnet_shuttle("DE-Hai"; years = 2018:2018)
df = flux_features(raw)
```
"""
function flux_features(raw::DataFrame; min_x = 200, min_range = 50)
    TA, SW_IN, SW_IN_POT = fillgaps(nan(raw.TA_F)), fillgaps(nan(raw.SW_IN_F)), nan(raw.SW_IN_POT)
    NEE_obs = ifelse.(coalesce.(raw.NEE_VUT_USTAR50_QC .== 0, false), nan(raw.NEE_VUT_USTAR50), NaN)
    night, day = SW_IN .<= 0, SW_IN .>= 20

    df = DataFrame(DateTime = raw.DateTime, NEE = NEE_obs, TA = TA, SW_IN = SW_IN, SW_IN_POT = SW_IN_POT)
    df.SW_IN_POT_diff = [0.0; diff(SW_IN_POT)]
    df.RH_F = clamp.(1 .- fillgaps(nan(raw.VPD_F)) ./ (10 .* esat.(TA)), 0, 1)

    # daily context from the NEE observations themselves
    df.NEE_night_movmean = moving(Statistics.median, ifelse.(night, NEE_obs, NaN))
    df.NEE_light_night_diff = df.NEE_night_movmean .- moving(Statistics.median, ifelse.(day, NEE_obs, NaN))
    df.TA_night_movmean = moving(Statistics.median, ifelse.(night, TA, NaN))
    lowlight = ifelse.(SW_IN .< 300, SW_IN, NaN)
    df.NEE_SWin_lowlight_lm_slope = moving(Statistics.median, moving_slope(lowlight, NEE_obs; min_range))
    h = 49 ÷ 2
    rue = map(eachindex(SW_IN)) do i
        slope_at_max_x(
            view(SW_IN, max(1, i - h):min(length(SW_IN), i + h)),
            view(NEE_obs, max(1, i - h):min(length(NEE_obs), i + h));
            min_x,
            min_range,
        )
    end
    df.NEE_RUE_at_max_SW = moving(Statistics.median, rue; window = 95)

    precip, LE = fillgaps(nan(raw.P_F)), fillgaps(nan(raw.LE_F_MDS))
    for cap in (1, 5, 25, 125, 625)
        df[!, "SWC_capa_$cap"] = soil_bucket(precip, LE, cap)
    end

    hours = Dates.hour.(df.DateTime) .+ Dates.minute.(df.DateTime) ./ 60
    df.cosHR = cos.(2π .* hours ./ 24)
    df.sinHR = sin.(2π .* hours ./ 24)
    decyear = Dates.year.(df.DateTime) .+ Dates.dayofyear.(df.DateTime) ./ 366
    elapsed = decyear .- minimum(decyear)
    period = maximum(elapsed)
    period = period > 0 ? period : one(period)
    for n in 1.0:5.0
        angle = 2π * n .* elapsed ./ period
        df[!, "sin_$(n)_year"] = sin.(angle)
        df[!, "cos_$(n)_year"] = cos.(angle)
    end

    for col in names(df, DataFrames.Not([:DateTime, :NEE]))
        df[!, col] = Float32.(fillgaps(df[!, col]))
    end
    df.NEE = Float32.(df.NEE)
    for col in (:RECO_NT_VUT_USTAR50, :GPP_NT_VUT_USTAR50, :RECO_DT_VUT_USTAR50, :GPP_DT_VUT_USTAR50)
        df[!, col] = Float32.(nan(raw[!, col]))
    end
    return df
end
