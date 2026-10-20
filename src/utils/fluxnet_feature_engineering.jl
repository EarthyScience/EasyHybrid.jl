export nan, fillgaps, moving, moving_slope, soil_bucket, esat, flux_features

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
    fillgaps(x) -> Vector

Fill non-finite values of `x` by linear interpolation between the neighbouring finite
values; leading and trailing gaps take the first and last finite value. Returns a copy.

# Example
```julia
fillgaps([NaN, 1.0, NaN, 3.0, NaN])  # [1.0, 1.0, 2.0, 3.0, 3.0]
```
"""
function fillgaps(x)
    y = copy(x)
    ok = findall(isfinite, y)
    isempty(ok) && return y
    y[1:(ok[1] - 1)] .= y[ok[1]]
    y[(ok[end] + 1):end] .= y[ok[end]]
    for (a, b) in zip(ok[1:(end - 1)], ok[2:end]), i in (a + 1):(b - 1)
        y[i] = y[a] + (y[b] - y[a]) * (i - a) / (b - a)
    end
    return y
end

"""
    moving(f, x; window = 49) -> Vector

Apply `f` to a centered moving window of `window` values of `x`, ignoring non-finite
values; the window is truncated at the edges. Returns `NaN` where a window holds no
finite value. The default of 49 half-hours covers about one day.

# Example
```julia
moving(median, NEE_night)
```
"""
function moving(f, x; window = 49)
    h, n = window ÷ 2, length(x)
    return map(1:n) do i
        v = filter(isfinite, view(x, max(1, i - h):min(n, i + h)))
        isempty(v) ? NaN : f(v)
    end
end

"""
    moving_slope(x, y; window = 49) -> Vector

Least-squares slope of `y` against `x` in a centered moving window of `window` values,
using only pairs where both are finite. Returns `NaN` where a window holds fewer than
four pairs or `x` does not vary.

# Example
```julia
moving_slope(SW_IN_lowlight, NEE)  # local light response of NEE
```
"""
function moving_slope(x, y; window = 49)
    h, n = window ÷ 2, length(x)
    return map(1:n) do i
        ok = filter(j -> isfinite(x[j]) && isfinite(y[j]), max(1, i - h):min(n, i + h))
        length(ok) > 3 || return NaN
        v = var(x[ok])
        v > 0 ? Statistics.cov(x[ok], y[ok]) / v : NaN
    end
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
    flux_features(raw::DataFrame) -> DataFrame

Engineer the predictors of the NEE flux-partitioning example from half-hourly FLUXNET
data, e.g. the output of [`load_fluxnet_shuttle`](@ref).

`raw` needs the columns `DateTime`, `TA_F`, `SW_IN_F`, `SW_IN_POT`, `VPD_F`, `P_F`,
`LE_F_MDS`, `NEE_VUT_USTAR50`, `NEE_VUT_USTAR50_QC` and the NT/DT partitioning results
`RECO_NT_VUT_USTAR50`, `GPP_NT_VUT_USTAR50`, `RECO_DT_VUT_USTAR50`, `GPP_DT_VUT_USTAR50`.

The returned `Float32` columns are
- `NEE`: measured NEE (`NEE_VUT_USTAR50_QC == 0`), `NaN` otherwise;
- `TA`, `SW_IN`, `SW_IN_POT`, `SW_IN_POT_diff`, `RH`: meteorology, with `RH` derived from
  `VPD_F` [hPa] and [`esat`](@ref);
- `NEE_night_movmean`, `TA_night_movmean`, `NEE_light_night_diff`: daily
  [`moving`](@ref) medians of night-time NEE and temperature and the night–day NEE
  difference;
- `NEE_SWin_lowlight_slope`: daily median of the local light response of NEE below
  300 W/m² ([`moving_slope`](@ref));
- `SWC_capa_1` … `SWC_capa_625`: [`soil_bucket`](@ref) soil-moisture proxies with
  capacities of 1, 5, 25, 125 and 625 mm;
- `sinHR`, `cosHR`, `sin_doy_k`, `cos_doy_k` (`k = 1, 2`): diurnal and seasonal harmonics;
- the NT/DT reference fluxes, unchanged.

All columns except `NEE` and the references are gap-filled with [`fillgaps`](@ref).

# Example
```julia
raw = load_fluxnet_shuttle("DE-Hai"; years = 2018:2018)
df = flux_features(raw)
```
"""
function flux_features(raw::DataFrame)
    TA, SW_IN, SW_IN_POT = fillgaps(nan(raw.TA_F)), fillgaps(nan(raw.SW_IN_F)), nan(raw.SW_IN_POT)
    NEE_obs = ifelse.(coalesce.(raw.NEE_VUT_USTAR50_QC .== 0, false), nan(raw.NEE_VUT_USTAR50), NaN)
    night, day = SW_IN .<= 0, SW_IN .>= 20

    df = DataFrame(DateTime = raw.DateTime, NEE = NEE_obs, TA = TA, SW_IN = SW_IN, SW_IN_POT = SW_IN_POT)
    df.SW_IN_POT_diff = [0.0; diff(SW_IN_POT)]
    df.RH = clamp.(1 .- fillgaps(nan(raw.VPD_F)) ./ (10 .* esat.(TA)), 0, 1)

    # daily context from the NEE observations themselves
    df.NEE_night_movmean = moving(Statistics.median, ifelse.(night, NEE_obs, NaN))
    df.NEE_light_night_diff = df.NEE_night_movmean .- moving(Statistics.median, ifelse.(day, NEE_obs, NaN))
    df.TA_night_movmean = moving(Statistics.median, ifelse.(night, TA, NaN))
    lowlight = ifelse.(SW_IN .< 300, SW_IN, NaN)
    df.NEE_SWin_lowlight_slope = moving(Statistics.median, moving_slope(lowlight, NEE_obs))

    precip, LE = fillgaps(nan(raw.P_F)), fillgaps(nan(raw.LE_F_MDS))
    for cap in (1, 5, 25, 125, 625)
        df[!, "SWC_capa_$cap"] = soil_bucket(precip, LE, cap)
    end

    hours = Dates.hour.(df.DateTime) .+ Dates.minute.(df.DateTime) ./ 60
    df.sinHR, df.cosHR = sin.(2π .* hours ./ 24), cos.(2π .* hours ./ 24)
    doy = Dates.dayofyear.(df.DateTime)
    for k in 1:2
        df[!, "sin_doy_$k"] = sin.(2π * k .* doy ./ 365.25)
        df[!, "cos_doy_$k"] = cos.(2π * k .* doy ./ 365.25)
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
