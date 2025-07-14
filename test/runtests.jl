using EasyHybrid
using Test
import Flux

dk = gen_linear_data()
dk_twos = gen_linear_data_2outputs()
#lhm_twos = LinearHybridModel_2outputs([:x2, :x3], [:x1, :x2], 1, 5, DenseNN; b=[4.0f0])
#out_twos = lhm_twos(dk_twos, :infer)

@testset "LinearHM" begin
    # test model instantiation
    NN = Lux.Chain(Lux.Dense(2, 5), Lux.Dense(5, 1))
    lhm = LinearHM(NN, (:x2, :x3), (:x1,), 1.5f0)
    @test lhm.forcing == [:x1]
    @test lhm.β == [1.5f0]
    @test lhm.predictors == [:x2, :x3]
end

@testset "EasyHybrid.jl" begin
    # test model instantiation
    lhm = LinearHybridModel([:x2, :x3], [:x1], 1, 5; b=[2.0f0])
    @test lhm.forcing == [:x1]
    @test lhm.b == [2.0f0]
    @test lhm.predictors == [:x2, :x3]
    @test typeof(lhm.DenseLayers) <: Flux.Chain

    # test model coupling
    out_lhm = lhm(dk, :infer)
    @test size(out_lhm.a) == (1, 1000)
    @test size(out_lhm.pred) == (1000, 1000)

    # test two_outputs
    lhm_twos = LinearHybridModel_2outputs([:x2, :x3], [:x1, :x2], 1, 5, DenseNN; b=[4.0f0])
    @test lhm_twos.a == [1.0f0]
    @test lhm_twos.b == [4.0f0]
    out_twos = lhm_twos(dk_twos, :infer)
    @test length(out_twos) == 3
    #test model output

    # test model instantiation
    fpmod = FluxPartModel_NEE_ET2([:TA_F, :SW_IN_F], [:doy], [:VPD_F, :hour], [:doy])
    @test typeof(fpmod) <: FluxPartModel_NEE_ET2
    # test model instantiation
    Q10_m = FluxPartModel_Q10([:TA, :VPD], [:SW_POT_sm_diff, :SW_POT_sm]; Q10=[2.0f0])
    @test Q10_m.Q10 == [2.0f0]

    # test model instantiation
    sinus_m = SinusHybridModel([:x1], [:x1], 1; b=[0.0f0])
    @test sinus_m.b == [0.0f0]

end
