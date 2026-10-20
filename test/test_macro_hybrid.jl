using Test
using EasyHybrid
using Lux
using Random
using Zygote

# Test mechanistic models
test_mech_linear(; x1, a, b, c = 0.0f0) = (; y_pred = a .* x1 .+ b .+ c)

function test_flux_mech(; SW_IN, TA, RUE, Rb, Q10)
    GPP = SW_IN .* RUE
    RECO = Rb .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
    NEE = RECO .- GPP
    return (; NEE, GPP, RECO)
end

@testset "@hybrid Macro & Revised Argument Order" begin
    params_linear = (
        a = (1.0f0, 0.0f0, 5.0f0),
        b = (2.0f0, 0.0f0, 10.0f0),
        c = (0.5f0, 0.0f0, 2.0f0),
    )

    @testset "constructHybridModel - Physics-First Ordering" begin
        # 1. Single-NN with Pair syntax and auto-inferred global parameters
        m1 = constructHybridModel(
            test_mech_linear,
            :y_pred,
            :x1,
            [:feat1, :feat2] => :a,
            params_linear;
            hidden_layers = [8, 8]
        )
        @test m1 isa HybridModel
        @test m1.targets == [:y_pred]
        @test m1.forcing == [:x1]
        @test m1.neural_param_names == [:a]
        @test m1.predictors == [:feat1, :feat2]
        @test sort(m1.global_param_names) == [:b, :c] # Auto-inferred
        @test isempty(m1.fixed_param_names)

        # 2. Multi-NN with NamedTuple syntax
        params_flux = (
            RUE = (1.0f0, 0.0f0, 5.0f0),
            Rb = (1.0f0, 0.0f0, 5.0f0),
            Q10 = (1.5f0, 1.0f0, 3.0f0, :log),
        )
        m2 = constructHybridModel(
            test_flux_mech,
            [:NEE, :GPP],
            [:SW_IN, :TA],
            (RUE = [:SWC, :TA, :SW_IN], Rb = [:SWC, :TA]),
            params_flux;
            scale_nn_outputs = true
        )
        @test m2 isa HybridModel
        @test m2.targets == [:NEE, :GPP]
        @test m2.forcing == [:SW_IN, :TA]
        @test m2.neural_param_names == [:RUE, :Rb]
        @test m2.global_param_names == [:Q10]
        @test m2.scale_nn_outputs == true

        # 3. Pure mechanistic (zero NNs)
        m3 = constructHybridModel(
            test_mech_linear,
            :y_pred,
            :x1,
            nothing,
            params_linear
        )
        @test m3 isa HybridModel
        @test isempty(m3.neural_param_names)
        @test isempty(m3.predictors)
        @test sort(m3.global_param_names) == [:a, :b, :c]
    end

    @testset "@hybrid Block DSL - Single NN" begin
        m_block = @hybrid begin
            mechanistic = test_mech_linear
            targets = :y_pred
            forcing = [:x1]
            neural = [:feat1, :feat2] => [:a]
            parameters = params_linear
            hidden_layers = [16, 16]
            activation = tanh
            scale_nn_outputs = true
        end

        @test m_block isa HybridModel
        @test m_block.targets == [:y_pred]
        @test m_block.forcing == [:x1]
        @test m_block.neural_param_names == [:a]
        @test m_block.predictors == [:feat1, :feat2]
        @test sort(m_block.global_param_names) == [:b, :c]
        @test m_block.scale_nn_outputs == true

        # Test forward pass & gradient
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, m_block)
        x_nn = rand(Float32, 2, 5)
        x_frc = (; x1 = rand(Float32, 5))
        out, st_new = m_block((x_nn, x_frc), ps, st)
        @test haskey(out, :y_pred)
        @test length(out.y_pred) == 5

        # Test AD gradient compatibility
        gs = Zygote.gradient(ps) do p
            o, _ = m_block((x_nn, x_frc), p, st)
            sum(o.y_pred)
        end
        @test gs[1] !== nothing
    end

    @testset "@hybrid Block DSL - Multi NN" begin
        params_flux = (
            RUE = (1.0f0, 0.0f0, 5.0f0),
            Rb = (1.0f0, 0.0f0, 5.0f0),
            Q10 = (1.5f0, 1.0f0, 3.0f0, :log),
        )

        m_multi = @hybrid begin
            model = test_flux_mech
            targets = [:NEE]
            forcing = [:SW_IN, :TA]
            predictors = (
                RUE = [:SWC, :TA, :SW_IN],
                Rb = [:SWC, :TA],
            )
            params = params_flux
            hidden_layers = (RUE = [8, 8], Rb = [4, 4])
            scale_nn_outputs = true
        end

        @test m_multi isa HybridModel
        @test m_multi.targets == [:NEE]
        @test m_multi.neural_param_names == [:RUE, :Rb]
        @test m_multi.global_param_names == [:Q10]

        # Test forward pass
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, m_multi)
        x_nn = (RUE = rand(Float32, 3, 4), Rb = rand(Float32, 2, 4))
        x_frc = (; SW_IN = rand(Float32, 4), TA = fill(20.0f0, 4))
        out, _ = m_multi((x_nn, x_frc), ps, st)
        @test haskey(out, :NEE)
        @test length(out.NEE) == 4
    end

    @testset "@hybrid Block DSL - Zero NN & Fixed Params" begin
        m_zero = @hybrid begin
            mechanistic = test_mech_linear
            targets = :y_pred
            forcing = :x1
            parameters = params_linear
            fixed_params = [:c]
        end

        @test m_zero isa HybridModel
        @test isempty(m_zero.neural_param_names)
        @test m_zero.global_param_names == [:a, :b]
        @test m_zero.fixed_param_names == [:c]
    end

    @testset "@hybrid Inline Syntax" begin
        m_inline = @hybrid test_mech_linear targets = :y_pred forcing = [:x1] neural = ([:feat1] => :a) parameters = params_linear hidden_layers = [4, 4]

        @test m_inline isa HybridModel
        @test m_inline.targets == [:y_pred]
        @test m_inline.forcing == [:x1]
        @test m_inline.neural_param_names == [:a]
        @test m_inline.predictors == [:feat1]
    end

    @testset "@hybrid Syntax Error Handling" begin
        # Missing mechanistic model
        @test_throws ErrorException @macroexpand @hybrid begin
            targets = :y_pred
            parameters = params_linear
        end

        # Missing targets
        @test_throws ErrorException @macroexpand @hybrid begin
            mechanistic = test_mech_linear
            parameters = params_linear
        end

        # Missing parameters
        @test_throws ErrorException @macroexpand @hybrid begin
            mechanistic = test_mech_linear
            targets = :y_pred
        end
    end
end
