using EasyHybrid
using Test
using Lux
using Random
using AxisKeys
using ComponentArrays

# Test data generation
dk = gen_linear_data()

# Simple mechanistic model for testing, c and d as dummy parameters
function test_mechanistic_model(; x1, a, b, c = nothing, d = nothing)
    return (; y_pred = a .* x1 .+ b)
end

# Test parameters
test_parameters = (
    a = (1.0f0, 0.0f0, 5.0f0),
    b = (2.0f0, 0.0f0, 10.0f0),
    c = (0.5f0, 0.0f0, 2.0f0),
    d = (0.5f0, 0.0f0, 2.0f0),
)

@testset "GenericHybridModel - Basic Functions" begin
    @testset "hard_sigmoid function" begin
        x = [0.0f0, 1.0f0, 2.0f0, -1.0f0, 5.0f0]
        result = hard_sigmoid(x)

        @test length(result) == length(x)
        @test all(result .>= 0.0f0)
        @test all(result .<= 1.0f0)
        @test result[1] ≈ 0.5f0  # hard_sigmoid(0) = 0.5
        @test result[2] ≈ 0.7f0  # hard_sigmoid(1) = 0.7
        @test result[3] ≈ 0.9f0  # hard_sigmoid(2) = 0.9
        @test result[4] ≈ 0.3f0  # hard_sigmoid(-1) = 0.3
        @test result[5] ≈ 1.0f0  # hard_sigmoid(5) = 1.0 (clamped)
    end

    @testset "AbstractHybridModel types" begin
        @test AbstractHybridModel <: Any
        @test ParameterContainer <: AbstractHybridModel
        @test HybridParams <: AbstractHybridModel
    end

    @testset "ParameterContainer construction" begin
        params = (a = (1.0f0, 0.0f0, 2.0f0), b = (2.0f0, 1.0f0, 3.0f0))
        pc = ParameterContainer(params)

        @test pc isa ParameterContainer
        @test hasproperty(pc, :values)
        @test hasproperty(pc, :table)
        @test pc.values == params
    end

    @testset "HybridParams construction" begin
        params = (a = (1.0f0, 0.0f0, 2.0f0),)
        pc = ParameterContainer(params)
        hp = HybridParams{typeof(test_mechanistic_model)}(pc)

        @test hp isa HybridParams
        @test hp.hybrid == pc
    end
end

@testset "GenericHybridModel - Parameter Functions" begin
    params = (a = (1.0f0, 0.0f0, 2.0f0), b = (2.0f0, 1.0f0, 3.0f0))
    hp = HybridParams{typeof(test_mechanistic_model)}(ParameterContainer(params))

    @testset "default function" begin
        defaults = default(hp)
        @test defaults.a == 1.0f0
        @test defaults.b == 2.0f0
    end

    @testset "lower function" begin
        lowers = lower(hp)
        @test lowers.a == 0.0f0
        @test lowers.b == 1.0f0
    end

    @testset "upper function" begin
        uppers = upper(hp)
        @test uppers.a == 2.0f0
        @test uppers.b == 3.0f0
    end

    @testset "pnames function" begin
        names = EasyHybrid.pnames(hp)
        @test collect(names) == [:a, :b]
    end

    @testset "scale_single_param function" begin
        # Test sigmoid scaling
        scaled_a = scale_single_param(:a, [0.0f0], hp)
        @test length(scaled_a) == 1
        @test scaled_a[1] ≈ 1.0f0  # sigmoid(0) = 0.5, so 0 + 0.5*(2-0) = 1.0

        scaled_b = scale_single_param(:b, [0.0f0], hp)
        @test scaled_b[1] ≈ 2.0f0  # sigmoid(0) = 0.5, so 1 + 0.5*(3-1) = 2.0
    end

    @testset "scale_single_param_minmax function" begin
        # Test inverse sigmoid scaling
        inv_scaled_a = EasyHybrid.scale_single_param_minmax(:a, hp)
        @test inv_scaled_a ≈ 0.0f0  # inverse sigmoid of 0.5 is 0.0

        inv_scaled_b = EasyHybrid.scale_single_param_minmax(:b, hp)
        @test inv_scaled_b ≈ 0.0f0  # inverse sigmoid of 0.5 is 0.0
    end
end

@testset "GenericHybridModel - SingleNNHybridModel" begin
    @testset "constructHybridModel with Vector predictors" begin
        predictors = [:x2, :x3]
        forcing = [:x1]
        targets = [:obs]
        neural_param_names = [:a]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            neural_param_names,
            global_param_names
        )

        @test model isa SingleNNHybridModel
        @test model.predictors == predictors
        @test model.forcing == forcing
        @test model.targets == targets
        @test model.neural_param_names == neural_param_names
        @test model.global_param_names == global_param_names
        @test model.fixed_param_names == [:c, :d]
        @test model.scale_nn_outputs == false
        @test model.start_from_default == true
        @test model.NN isa Chain
    end

    @testset "constructHybridModel with empty predictors" begin
        predictors = Symbol[]
        forcing = [:x1]
        targets = [:obs]
        neural_param_names = Symbol[]
        global_param_names = [:a, :b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            neural_param_names,
            global_param_names
        )

        @test model isa SingleNNHybridModel
        @test model.predictors == predictors
        @test model.NN isa Chain
        @test length(model.NN.layers) == 0  # Empty chain
    end

    @testset "SingleNNHybridModel initialparameters" begin
        predictors = [:x2, :x3]
        forcing = [:x1]
        targets = [:obs]
        neural_param_names = [:a]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            neural_param_names,
            global_param_names
        )

        rng = Random.default_rng()
        ps = LuxCore.initialparameters(rng, model)

        @test haskey(ps, :ps)  # Neural network parameters
        @test haskey(ps, :b)   # Global parameter
        @test length(ps.b) == 1
        @test ps.b[1] isa Float32
    end

    @testset "SingleNNHybridModel initialstates" begin
        predictors = [:x2, :x3]
        forcing = [:x1]
        targets = [:obs]
        neural_param_names = [:a]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            neural_param_names,
            global_param_names
        )

        rng = Random.default_rng()
        st = LuxCore.initialstates(rng, model)

        @test haskey(st, :st_nn)    # Neural network states
        @test haskey(st, :fixed) # Fixed parameters
        @test haskey(st.fixed, :c)
        @test length(st.fixed.c) == 1
        @test st.fixed.c[1] isa Float32
        @test haskey(st.fixed, :d)
        @test length(st.fixed.d) == 1
        @test st.fixed.d[1] isa Float32
    end

    @testset "SingleNNHybridModel forward pass" begin
        predictors = [:x2, :x3]
        forcing = [:x1]
        targets = [:obs]
        neural_param_names = [:a]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            neural_param_names,
            global_param_names
        )

        rng = Random.default_rng()
        ps = LuxCore.initialparameters(rng, model)
        st = LuxCore.initialstates(rng, model)

        # Test forward pass
        output, new_st = model(dk, ps, st)

        @test haskey(output, :y_pred)
        @test haskey(output, :parameters)
        @test haskey(output.parameters, :a)
        @test haskey(output.parameters, :b)
        @test haskey(output.parameters, :c)
        @test haskey(new_st, :st_nn)
        @test haskey(new_st, :fixed)
    end

    @testset "SingleNNHybridModel with scale_nn_outputs=true" begin
        predictors = [:x2, :x3]
        forcing = [:x1]
        targets = [:obs]
        neural_param_names = [:a]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            neural_param_names,
            global_param_names;
            scale_nn_outputs = true
        )

        @test model.scale_nn_outputs == true

        rng = Random.default_rng()
        ps = LuxCore.initialparameters(rng, model)
        st = LuxCore.initialstates(rng, model)

        output, new_st = model(dk, ps, st)

        @test haskey(output, :y_pred)
        @test haskey(output, :parameters)
    end
end

@testset "GenericHybridModel - MultiNNHybridModel" begin
    @testset "constructHybridModel with NamedTuple predictors" begin
        predictors = (a = [:x2, :x3], d = [:x1])
        forcing = [:x1]
        targets = [:obs]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            global_param_names
        )

        @test model isa MultiNNHybridModel
        @test model.predictors == predictors
        @test model.forcing == forcing
        @test model.targets == targets
        @test model.neural_param_names == [:a, :d]
        @test model.global_param_names == global_param_names
        @test model.fixed_param_names == [:c]
        @test model.scale_nn_outputs == false
        @test model.start_from_default == true
        @test haskey(model.NNs, :a)
        @test haskey(model.NNs, :d)
    end

    @testset "MultiNNHybridModel with NamedTuple hidden_layers and activation" begin
        predictors = (a = [:x2, :x3], d = [:x1])
        forcing = [:x1]
        targets = [:obs]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            global_param_names;
            hidden_layers = (a = [16, 8], d = [8]),
            activation = (a = tanh, d = sigmoid)
        )

        @test model isa MultiNNHybridModel
        @test haskey(model.NNs, :a)
        @test haskey(model.NNs, :d)
    end

    @testset "MultiNNHybridModel initialparameters" begin
        predictors = (a = [:x2, :x3], d = [:x1])
        forcing = [:x1]
        targets = [:obs]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            global_param_names
        )

        rng = Random.default_rng()
        ps = LuxCore.initialparameters(rng, model)

        @test haskey(ps, :a)  # First neural network parameters
        @test haskey(ps, :d)  # Second neural network parameters
        @test haskey(ps, :b)    # Global parameter
        @test length(ps.b) == 1
        @test ps.b[1] isa Float32
    end

    @testset "MultiNNHybridModel initialstates" begin
        predictors = (a = [:x2, :x3], d = [:x1])
        forcing = [:x1]
        targets = [:obs]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            global_param_names
        )

        rng = Random.default_rng()
        st = LuxCore.initialstates(rng, model)

        @test haskey(st, :a)   # First neural network states
        @test haskey(st, :d)   # Second neural network states
        @test haskey(st, :fixed) # Fixed parameters
        @test haskey(st.fixed, :c)
        @test length(st.fixed.c) == 1
        @test st.fixed.c[1] isa Float32
    end

    @testset "MultiNNHybridModel forward pass" begin
        predictors = (a = [:x2, :x3], d = [:x1])
        forcing = [:x1]
        targets = [:obs]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            global_param_names
        )

        rng = Random.default_rng()
        ps = LuxCore.initialparameters(rng, model)
        st = LuxCore.initialstates(rng, model)

        # Test forward pass
        output, new_st = model(dk, ps, st)

        @test haskey(output, :y_pred)
        @test haskey(output, :parameters)
        @test haskey(output, :nn_outputs)
        @test haskey(output.parameters, :a)
        @test haskey(output.parameters, :b)
        @test haskey(output.parameters, :c)
        @test haskey(output.parameters, :d)
        @test haskey(new_st, :a)
        @test haskey(new_st, :d)
        @test haskey(new_st, :fixed)
    end

    @testset "MultiNNHybridModel with scale_nn_outputs=true" begin
        predictors = (a = [:x2, :x3], d = [:x1])
        forcing = [:x1]
        targets = [:obs]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            global_param_names;
            scale_nn_outputs = true
        )

        @test model.scale_nn_outputs == true

        rng = Random.default_rng()
        ps = LuxCore.initialparameters(rng, model)
        st = LuxCore.initialstates(rng, model)

        # ! TODO: Fix this test; currently fails due to nn_outputs not being returned
        # ERROR: type NamedTuple has no field nn1
        output, new_st = model(dk, ps, st)

        @test haskey(output, :y_pred)
        @test haskey(output, :parameters)
        @test haskey(output, :nn_outputs)
        @test haskey(output.nn_outputs, :a)
    end
end

@testset "GenericHybridModel - Edge Cases" begin
    @testset "Empty neural and global parameters" begin
        predictors = Symbol[]
        forcing = [:x1]
        targets = [:obs]
        neural_param_names = Symbol[]
        global_param_names = Symbol[]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            neural_param_names,
            global_param_names
        )

        @test model isa SingleNNHybridModel
        @test isempty(model.neural_param_names)
        @test isempty(model.global_param_names)
        @test model.fixed_param_names == [:a, :b, :c, :d]

        rng = Random.default_rng()
        ps = LuxCore.initialparameters(rng, model)
        st = LuxCore.initialstates(rng, model)

        @test haskey(ps, :ps)  # Even with empty NN, ps key exists (may be empty)
        @test isempty(ps.ps)

        output, new_st = model(dk, ps, st)
        @test haskey(output, :y_pred)
        @test haskey(output, :parameters)
    end

    @testset "start_from_default=false" begin
        predictors = [:x2, :x3]
        forcing = [:x1]
        targets = [:obs]
        neural_param_names = [:a]
        global_param_names = [:b]

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            neural_param_names,
            global_param_names;
            start_from_default = false
        )

        @test model.start_from_default == false

        rng = Random.default_rng()
        ps = LuxCore.initialparameters(rng, model)

        # Parameters should be random, not default values
        @test haskey(ps, :b)
        @test ps.b[1] isa Float32
        # Note: We can't easily test if it's random vs default without setting a specific seed
    end

    @testset "Custom Chain as hidden_layers" begin
        predictors = [:x2, :x3]
        forcing = [:x1]
        targets = [:obs]
        neural_param_names = [:a]
        global_param_names = [:b]

        custom_chain = Chain(Dense(2, 16, tanh), Dense(16, 8, tanh))

        model = constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            neural_param_names,
            global_param_names;
            hidden_layers = custom_chain
        )

        @test model isa SingleNNHybridModel
        @test model.NN isa Chain
        # The chain should have the custom layers plus input and output layers
        @test length(model.NN.layers) > length(custom_chain.layers)
    end
end

@testset "GenericHybridModel - Error Handling" begin
    @testset "Invalid neural_param_names" begin
        predictors = [:x2, :x3]
        forcing = [:x1]
        targets = [:obs]
        neural_param_names = [:invalid_param]  # Not in test_parameters
        global_param_names = [:b]

        @test_throws AssertionError constructHybridModel(
            predictors,
            forcing,
            targets,
            test_mechanistic_model,
            test_parameters,
            neural_param_names,
            global_param_names
        )
    end

    # No assertion on invalid global_param_names in current implementation; skip such test.
end
