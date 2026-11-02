using Dates: Date, DateTime, Hour, Minute

@testset "combine_flux_partition" begin
    time = [
        DateTime(2018, 6, 1, 0, 0),
        DateTime(2018, 6, 1, 0, 30),
        DateTime(2018, 6, 1, 1, 0),
        DateTime(2018, 6, 1, 1, 30),
        DateTime(2018, 6, 2, 0, 0),
        DateTime(2018, 6, 2, 0, 30),
        DateTime(2018, 6, 2, 1, 0),
        DateTime(2018, 6, 2, 1, 30),
    ]
    raw = DataFrame(
        DateTime = time,
        TA_F = fill(10.0, 8),
        P_F = [2.0, missing, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0],
        LE_F_MDS = zeros(8),
        NEE_VUT_USTAR50 = [1.0, 100.0, 3.0, 100.0, 5.0, 100.0, 7.0, 100.0],
        NEE_VUT_USTAR50_QC = [0, 1, 0, 1, 0, 1, 0, 1],
        RECO_NT_VUT_USTAR50 = fill(2.0, 8),
        GPP_NT_VUT_USTAR50 = fill(0.5, 8),
        RECO_DT_VUT_USTAR50 = fill(3.0, 8),
        GPP_DT_VUT_USTAR50 = fill(1.0, 8),
    )
    pred = DataFrame(
        DateTime = reverse(time),
        NEE = fill(999.0, 8),
        TA = fill(99.0, 8),
        NEE_pred = collect(1.0:8.0),
        RECO_pred = fill(5.0, 8),
        GPP_pred = fill(1.0, 8),
        RUE_pred = fill(0.2, 8),
    )
    other = DataFrame(NEE_pred = fill(8.0, 8), RECO_pred = fill(6.0, 8), GPP_pred = fill(2.0, 8))

    half = combine_flux_partition(raw, pred, other; labels = ["hybrid", "refit"], scale = :hh)
    @test nrow(half) == 8
    @test half.time == time
    @test half.NEE[1] == 1.0 && isnan(half.NEE[2])
    @test half.TA == fill(10.0, 8)
    @test half.P == [2.0, 3.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0]
    @test half.NEE_DT == fill(2.0, 8)
    @test half.NEE_pred == collect(8.0:-1.0:1.0)
    @test half.NEE_pred_2 == fill(8.0, 8)
    @test half.RUE_pred == fill(0.2, 8)
    @test half.SWC_capa_25 == fill(25.0, 8)
    @test half.SWC_capa_625 == fill(625.0, 8)

    hourly = combine_flux_partition(raw, pred; scale = "hourly")
    @test hourly.time == [
        DateTime(2018, 6, 1), DateTime(2018, 6, 1, 1),
        DateTime(2018, 6, 2), DateTime(2018, 6, 2, 1),
    ]
    @test hourly.NEE == [1.0, 3.0, 5.0, 7.0]
    @test hourly.P == [5.0, 8.0, 0.0, 0.0]
    @test EasyHybrid._precip_unit(:hourly) == "mm/h"
    @test EasyHybrid._bar_width(:hourly) == 3_600_000.0

    daily = combine_flux_partition(raw, pred; scale = :day)
    @test daily.time == [Date(2018, 6, 1), Date(2018, 6, 2)]
    @test daily.NEE == [2.0, 6.0]
    @test daily.P == [13.0, 0.0]
    @test daily.NEE_pred == [6.5, 2.5]
    @test daily.RECO_pred == fill(5.0, 2)
    @test daily.GPP_NT_VUT_USTAR50 == fill(0.5, 2)
    @test EasyHybrid._precip_unit(:daily) == "mm/day"
    @test EasyHybrid._bar_width(:daily) == 86_400_000.0
    @test EasyHybrid._bar_width(:halfhourly) == 1_800_000.0

    kept = DataFrame(DateTime = time, SWC_capa_25 = fill(7.0, 8), NEE_pred = zeros(8))
    @test combine_flux_partition(raw, kept; swc_caps = (25,), scale = :halfhourly).SWC_capa_25 == fill(7.0, 8)

    @test EasyHybrid._swc_caps(:all) == [1, 5, 25, 125, 625]
    @test EasyHybrid._swc_caps("ALL") == [1, 5, 25, 125, 625]
    @test EasyHybrid._swc_caps(25) == [25]
    @test EasyHybrid._swc_caps((5, 125)) == [5, 125]
    @test_throws ArgumentError EasyHybrid._swc_caps(:shallow)
    @test_throws ArgumentError EasyHybrid._swc_caps(-1)

    every = combine_flux_partition(raw, pred; swc_caps = :all, scale = :halfhourly)
    @test every.SWC_capa_1 == fill(1.0, 8)
    @test every.SWC_capa_5 == fill(5.0, 8)
    @test every.SWC_capa_125 == fill(125.0, 8)
    @test every.SWC_capa_625 == fill(625.0, 8)
    one = combine_flux_partition(raw, pred; swc_caps = 5, scale = :halfhourly)
    @test one.SWC_capa_5 == fill(5.0, 8)
    @test !("SWC_capa_25" in names(one))
    @test !("SWC_capa_1" in names(one))

    hourly_obs = DataFrame(
        DateTime = DateTime(2018, 6, 1) .+ Hour.(0:3),
        P_F = zeros(4),
        LE_F_MDS = zeros(4),
        TA = fill(3.0, 4),
    )
    hourly_only = combine_flux_partition(hourly_obs; scale = :hourly)
    @test hourly_only.TA == fill(3.0, 4)
    @test !("SWC_capa_25" in names(hourly_only))

    @test_throws ArgumentError combine_flux_partition(raw; scale = :weekly)
    @test_throws ArgumentError combine_flux_partition(raw, pred, other; labels = ["only one"])
    @test_throws ArgumentError combine_flux_partition(raw[8:-1:1, :], pred)
    @test_throws ArgumentError combine_flux_partition(raw, DataFrame(NEE_pred = [1.0]))
    duplicated = DataFrame(DateTime = [time[1], time[1]], NEE_pred = [1.0, 2.0])
    @test_throws ArgumentError combine_flux_partition(raw, duplicated)
end
