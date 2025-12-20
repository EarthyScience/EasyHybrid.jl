using EasyHybrid: HybridParams, ParameterContainer, SingleNNHybridModel, MultiNNHybridModel
using EasyHybrid: _print_field, _print_header, IndentedIO

@testset "show_generic.jl" begin

    @testset "_print_field and _print_header" begin
        result = sprint(
            io -> begin
                _print_header(io, "Test Header", color = :blue)
                _print_field(io, "scalar", 42)
                _print_field(io, "bool_true", true)
                _print_field(io, "bool_false", false)
                _print_field(io, "namedtuple", (a = 1, b = 2.0))
                _print_field(io, "function", sum)
            end, context = :color => false
        )

        @test occursin("Test Header", result)
        @test occursin("scalar", result) && occursin("42", result)
        @test occursin("bool_true", result) && occursin("bool_false", result)
        @test occursin("namedtuple", result) && occursin("a =", result) && occursin("b =", result)
        @test occursin("function", result) && occursin("sum", result)
    end

    @testset "HybridParams show" begin
        params = (a = (1.0, 0.0, 2.0), b = (2.0, 1.0, 3.0))
        pc = ParameterContainer(params)
        hp = HybridParams{typeof(sum)}(pc)

        # Compact show
        result_compact = sprint(show, hp, context = :color => false)
        @test occursin("HybridParams", result_compact) && occursin("ParameterContainer", result_compact)

        # Text/plain show
        result_full = sprint(show, MIME"text/plain"(), hp, context = :color => false)
        @test occursin("Hybrid Parameters", result_full) # only check for the header
    end

    @testset "ParameterContainer compact show" begin
        params = (a = (1.0, 0.0, 2.0), b = (2.0, 1.0, 3.0))
        pc = ParameterContainer(params)

        result = sprint(show, pc, context = :color => false)
        @test occursin("ParameterContainer(a, b)", result)
    end

    @testset "SingleNNHybridModel show" begin
        function test_model(; x1, a, b)
            return (; y_pred = a .* x1 .+ b)
        end

        model = constructHybridModel(
            [:x1, :x2], [:x3], [:y], test_model,
            (a = (1.0, 0.0, 2.0), b = (2.0, 1.0, 3.0)),
            [:a], [:b];
            hidden_layers = [4, 4], activation = tanh
        )

        result = sprint(show, MIME"text/plain"(), model, context = :color => false)

        @test occursin("Hybrid Model (Single NN)", result)
        @test occursin("Neural Network:", result) && occursin("Configuration:", result)
        @test all(
            occursin.(
                [
                    "predictors", "forcing", "targets", "mechanistic_model",
                    "neural_param_names", "global_param_names", "scale_nn_outputs",
                    "start_from_default", "config", "Parameters:",
                ], Ref(result)
            )
        )
    end

    @testset "MultiNNHybridModel show" begin
        function test_model(; x1, x2, x3, a, b, c, d)
            return (; obs = a .* x2 .+ d .* x1 .+ b)
        end

        model = constructHybridModel(
            (a = [:x2, :x3], d = [:x1]), [:x1], [:obs], test_model,
            (a = (1.0, 0.0, 5.0), b = (2.0, 0.0, 10.0), c = (0.5, 0.0, 2.0), d = (0.5, 0.0, 2.0)),
            [:b];  # Only global_param_names, neural_param_names derived from predictors keys
            hidden_layers = [4, 4], activation = tanh
        )

        result = sprint(show, MIME"text/plain"(), model, context = :color => false)

        @test occursin("Hybrid Model (Multi NN)", result)
        @test occursin("Neural Networks:", result) && occursin("Configuration:", result)
        @test all(occursin.(["predictors", "a", "d", "forcing", "targets", "config", "Parameters:"], Ref(result)))
    end

    @testset "IndentedIO" begin
        result = sprint(
            io -> begin
                ido = IndentedIO(io; indent = "  ")
                println(ido, "line1")
                println(ido, "line2")
            end, context = :color => false
        )

        @test occursin("  line1", result) && occursin("  line2", result)
    end

end
