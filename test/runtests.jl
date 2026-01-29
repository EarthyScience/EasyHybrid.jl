using EasyHybrid
using Test

# Include GenericHybridModel tests
include("test_generic_hybrid_model.jl")
# Include SplitData tests
include("test_split_data_train.jl")
include("test_autodiff_backend.jl")
include("test_loss_types.jl")
include("test_show_loss_types.jl")
include("test_compute_loss.jl")
include("test_loss_fn.jl")
include("test_show_train.jl")
include("test_show_generic_hybrid.jl")
include("test_wrap_tuples.jl")

@testset "LinearHM" begin
    # test model instantiation
    NN = Lux.Chain(Lux.Dense(2, 5), Lux.Dense(5, 1))
    lhm = LinearHM(NN, (:x2, :x3), (:x1,), (:obs,), 1.5f0)
    @test lhm.forcing == [:x1]
    @test lhm.β == [1.5f0]
    @test lhm.predictors == [:x2, :x3]
end
