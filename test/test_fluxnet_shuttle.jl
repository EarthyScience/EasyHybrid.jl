using Dates: DateTime
using ZipArchives: ZipWriter, zip_writefile

@testset "FLUXNET Shuttle loader" begin
    data_dir = mktempdir()
    csv = """
    TIMESTAMP_START,TIMESTAMP_END,TA_F,NEE_VUT_USTAR50
    201512312330,201601010000,1.5,-9999
    201601010000,201601010030,2.0,0.5
    201701010000,201701010030,3.0,-9999
    """
    ZipWriter(joinpath(data_dir, "ICOS_XX-Tst_FLUXNET_2015-2017_v1.3_r1.zip")) do w
        zip_writefile(w, "README.txt", codeunits("test archive"))
        zip_writefile(w, "ICOS_XX-Tst_FLUXNET_FLUXMET_HH_2015-2017_v1.3_r1.csv", codeunits(csv))
    end

    @test fluxnet_shuttle_zip("XX-Tst"; data_dir) == joinpath(data_dir, "ICOS_XX-Tst_FLUXNET_2015-2017_v1.3_r1.zip")

    df = load_fluxnet_shuttle("XX-Tst"; data_dir)
    @test names(df) == ["DateTime", "TIMESTAMP_START", "TIMESTAMP_END", "TA_F", "NEE_VUT_USTAR50"]
    @test df.DateTime == [DateTime(2015, 12, 31, 23, 30), DateTime(2016, 1, 1), DateTime(2017, 1, 1)]
    @test isequal(df.NEE_VUT_USTAR50, [missing, 0.5, missing])
    @test isfile(joinpath(data_dir, "ICOS_XX-Tst_FLUXNET_FLUXMET_HH_2015-2017_v1.3_r1.csv"))

    df_2016 = load_fluxnet_shuttle("XX-Tst"; variables = [:TA_F], years = 2016:2016, data_dir)
    @test names(df_2016) == ["DateTime", "TIMESTAMP_START", "TA_F"]
    @test df_2016.TA_F == [2.0]
end
