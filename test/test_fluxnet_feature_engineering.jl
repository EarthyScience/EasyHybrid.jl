using Dates: Dates, DateTime

@testset "FLUXNET feature engineering" begin
    @test isequal(nan([1, missing, 3]), [1.0, NaN, 3.0])

    @test fillgaps([NaN, 1.0, NaN, 3.0, NaN]) == [1.0, 1.0, 2.0, 3.0, 3.0]
    @test all(isnan, fillgaps([NaN, NaN]))
    x = [1.0, NaN, 3.0]
    fillgaps(x)
    @test isnan(x[2])

    @test moving(sum, [1.0, 2.0, 3.0, 4.0, 5.0]; window = 3) == [3.0, 6.0, 9.0, 12.0, 9.0]
    @test moving(sum, [1.0, NaN, 3.0]; window = 3) == [1.0, 4.0, 3.0]
    @test isequal(moving(sum, [NaN, NaN, NaN, 1.0]; window = 3), [NaN, NaN, 1.0, 1.0])

    x = collect(1.0:10.0)
    slope = moving_slope(x, 2 .* x .+ 1; window = 5)
    @test all(slope[2:9] .≈ 2)
    @test isnan(slope[1]) && isnan(slope[10])
    @test all(isnan, moving_slope(ones(10), x; window = 5))
    @test isnan(moving_slope([1.0, 2.0, NaN, NaN, NaN], x[1:5]; window = 5)[3])

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
    @test issubset(["SWC_capa_25", "sinHR", "cos_doy_2", "NEE_SWin_lowlight_slope", "RH"], names(df))
end
