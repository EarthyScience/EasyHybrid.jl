using Dates: Dates, DateTime

@testset "FLUXNET feature engineering" begin
    @test isequal(nan([1, missing, 3]), [1.0, NaN, 3.0])

    @test fillgaps([NaN, 1.0, NaN, 3.0, NaN]) == [1.0, 1.0, 2.0, 3.0, 3.0]
    @test all(isnan, fillgaps([NaN, NaN]))
    short = [1.0; fill(NaN, 2); 4.0]
    @test fillgaps(short; maxgap = 2) ≈ [1.0, 2.0, 3.0, 4.0]
    long = [1.0; fill(NaN, 4); 2.0]
    @test isnan(fillgaps(long; maxgap = 2)[3]) && fillgaps(long; maxgap = 2)[1] == 1.0
    @test all(isnan, fillgaps([NaN, NaN, NaN, 1.0]; maxgap = 2)[1:2])
    x = [1.0, NaN, 3.0]
    fillgaps(x)
    @test isnan(x[2])

    @test moving(sum, [1.0, 2.0, 3.0, 4.0, 5.0]; window = 3) == [3.0, 6.0, 9.0, 12.0, 9.0]
    @test moving(sum, [1.0, NaN, 3.0]; window = 3) == [1.0, 4.0, 3.0]
    @test isequal(moving(sum, [NaN, NaN, NaN, 1.0]; window = 3), [NaN, NaN, 1.0, 1.0])
    @test isnan(moving(sum, [1.0, NaN, 3.0]; window = 3, min_count = 2)[1])
    @test moving(sum, [1.0, NaN, 3.0]; window = 3, min_count = 2)[2] == 4.0

    x = collect(1.0:10.0)
    slope = moving_slope(x, 2 .* x .+ 1; window = 5, min_range = 0)
    @test all(slope[2:9] .≈ 2)
    @test isnan(slope[1]) && isnan(slope[10])
    @test all(isnan, moving_slope(ones(10), x; window = 5))
    @test isnan(moving_slope([1.0, 2.0, NaN, NaN, NaN], x[1:5]; window = 5)[3])
    @test isnan(moving_slope([0.0, 1.7, 0.0, 0.5, 1.0], [1.0, 2.0, 3.0, 4.0, 5.0]; window = 5)[3])
    wide = collect(0.0:20.0:80.0)
    @test moving_slope(wide, 2 .* wide; window = 5)[3] ≈ 2

    @test isnan(slope_at_max_x([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]))
    x = [1.0, 2.0, 3.0, 4.0, 10.0]
    y = [0.0, 1.0, 2.0, 3.0, -5.0]
    @test slope_at_max_x(x, y; min_x = 0, min_range = 0) ≈ (y[5] - quantile(y, 0.9)) / x[5]
    @test slope_at_max_x([1.0, NaN, 3.0, 4.0, 10.0], [0.0, 100.0, 2.0, 3.0, -5.0]; min_x = 0, min_range = 0) ≈
        slope_at_max_x([1.0, 3.0, 4.0, 10.0], [0.0, 2.0, 3.0, -5.0]; min_x = 0, min_range = 0)
    @test isnan(slope_at_max_x([0.0, 0.0, 0.0, 1.7], [3.56, 2.0, 1.5, 1.30]))
    xhi = [0.0, 100.0, 400.0, 800.0]
    yhi = [2.0, 1.0, 0.0, -4.0]
    @test slope_at_max_x(xhi, yhi) ≈ (yhi[4] - quantile(yhi, 0.9)) / xhi[4]

    @test soil_bucket([0.0, 0.0, 5.0], [0.0, 2.45e6 / 1800, 0.0], 2.0) == [2.0, 1.0, 2.0]
    @test esat(0.0) ≈ 0.6112
    @test esat(20.0) ≈ 2.3326 atol = 1.0e-3
end

@testset "flux_features" begin
    n = 96
    t = DateTime(2018, 6, 1) .+ Dates.Minute.(30 .* (0:(n - 1)))
    sw = max.(0.0, 800 .* sin.(2π .* ((0:(n - 1)) .% 48 .- 12) ./ 48))
    raw = DataFrame(
        DateTime = t,
        TA_F = [missing; fill(15.0, n - 1)],
        SW_IN_F = sw,
        SW_IN_POT = sw,
        VPD_F = fill(5.0, n),
        P_F = zeros(n),
        LE_F_MDS = fill(50.0, n),
        NEE_VUT_USTAR50 = 2.0 .- 0.01 .* sw,
        NEE_VUT_USTAR50_QC = [isodd(i) ? 0 : 1 for i in 1:n],
        RECO_NT_VUT_USTAR50 = fill(2.0, n),
        GPP_NT_VUT_USTAR50 = 0.01 .* sw,
        RECO_DT_VUT_USTAR50 = fill(2.0, n),
        GPP_DT_VUT_USTAR50 = 0.01 .* sw,
    )
    df = flux_features(raw)

    @test nrow(df) == n
    @test all(c -> eltype(df[!, c]) == Float32, names(df, Not(:DateTime)))
    @test all(isnan, df.NEE[2:2:end]) && all(isfinite, df.NEE[1:2:end])
    predictors = names(df, Not([:DateTime, :NEE]))
    @test all(c -> all(isfinite, df[!, c]), predictors)
    @test df.TA[1] == 15.0f0
    @test all(df.NEE_night_movmean .≈ 2.0f0)
    @test all(df.SWC_capa_1 .< df.SWC_capa_625)
    @test issubset(
        ["SWC_capa_25", "sinHR", "cosHR", "sin_1.0_year", "cos_5.0_year", "NEE_SWin_lowlight_lm_slope", "NEE_RUE_at_max_SW", "RH_F"],
        names(df),
    )
    @test df[1, "sin_1.0_year"] ≈ 0 atol = 1.0e-5
    @test df[1, "cos_1.0_year"] ≈ 1 atol = 1.0e-5
    @test df[end, "cos_5.0_year"] ≈ 1 atol = 1.0e-5
end

@testset "long NEE outage does not invent light-response slopes" begin
    n_day, n_good, n_gap = 48, 4 * 48, 6 * 48
    n = 2 * n_good + n_gap
    t = DateTime(2018, 9, 1) .+ Dates.Minute.(30 .* (0:(n - 1)))
    sw = max.(0.0, 800 .* sin.(2π .* ((0:(n - 1)) .% n_day .- 12) ./ n_day))
    measured = trues(n)
    measured[(n_good + 1):(n_good + n_gap)] .= false
    raw = DataFrame(
        DateTime = t,
        TA_F = fill(15.0, n),
        SW_IN_F = sw,
        SW_IN_POT = sw,
        VPD_F = fill(5.0, n),
        P_F = zeros(n),
        LE_F_MDS = fill(50.0, n),
        NEE_VUT_USTAR50 = 2.0 .- 0.01 .* sw,
        NEE_VUT_USTAR50_QC = [measured[i] ? 0 : 1 for i in 1:n],
        RECO_NT_VUT_USTAR50 = fill(2.0, n),
        GPP_NT_VUT_USTAR50 = 0.01 .* sw,
        RECO_DT_VUT_USTAR50 = fill(2.0, n),
        GPP_DT_VUT_USTAR50 = 0.01 .* sw,
    )
    df = flux_features(raw)
    for col in ("NEE_night_movmean", "NEE_light_night_diff", "NEE_SWin_lowlight_lm_slope", "NEE_RUE_at_max_SW")
        @test all(isfinite, df[!, col])
    end
    for col in ("NEE_SWin_lowlight_lm_slope", "NEE_RUE_at_max_SW")
        @test all(abs.(df[!, col]) .< 0.2)
    end
    @test all(df.NEE_night_movmean .≈ 2.0f0)
    @test all(isfinite, df.SW_IN)
end

@testset "flux_features light-response thresholds" begin
    n = 48
    t = DateTime(2018, 6, 1) .+ Dates.Minute.(30 .* (0:(n - 1)))
    sw = 5 .* sin.(2π .* (0:(n - 1)) ./ n) .^ 2
    raw = DataFrame(
        DateTime = t,
        TA_F = fill(15.0, n),
        SW_IN_F = sw,
        SW_IN_POT = sw,
        VPD_F = fill(5.0, n),
        P_F = zeros(n),
        LE_F_MDS = fill(50.0, n),
        NEE_VUT_USTAR50 = 2.0 .- 0.1 .* sw,
        NEE_VUT_USTAR50_QC = fill(0, n),
        RECO_NT_VUT_USTAR50 = fill(2.0, n),
        GPP_NT_VUT_USTAR50 = 0.1 .* sw,
        RECO_DT_VUT_USTAR50 = fill(2.0, n),
        GPP_DT_VUT_USTAR50 = 0.1 .* sw,
    )
    guarded = flux_features(raw)
    open = flux_features(raw; min_x = 0, min_range = 0)
    @test all(isnan, guarded.NEE_RUE_at_max_SW)
    @test all(isnan, guarded.NEE_SWin_lowlight_lm_slope)
    @test all(isfinite, open.NEE_RUE_at_max_SW)
    @test all(isfinite, open.NEE_SWin_lowlight_lm_slope)
end
