using Test
using EasyHybrid
using EasyHybrid: SymbolicLoss, FunctionLoss, ParameterizedLoss, ExtraLoss, _format_loss_spec, LoggingLoss, PerTarget

identity_fn(x) = x
simple_fn(x, a) = x + a
simple_fn(x, y; scale = 1.0) = scale * (x + y)


@testset "Loss spec formatting and LoggingLoss printing" begin

    @testset "SymbolicLoss" begin
        spec = SymbolicLoss(:mse)
        result = sprint(io -> _format_loss_spec(io, spec), context = :color => false)
        @test result == ":mse"
    end

    @testset "FunctionLoss" begin
        spec = FunctionLoss(identity_fn)
        result = sprint(io -> _format_loss_spec(io, spec), context = :color => false)
        @test result == "identity_fn"
    end

    @testset "ParameterizedLoss" begin
        spec_args = ParameterizedLoss(simple_fn, (1, 2), NamedTuple())
        result = sprint(io -> _format_loss_spec(io, spec_args), context = :color => false)
        @test result == "simple_fn(1, 2)"

        spec_kwargs = ParameterizedLoss(simple_fn, (), (scale = 2.0,))
        result = sprint(io -> _format_loss_spec(io, spec_kwargs), context = :color => false)
        @test result == "simple_fn(; scale=2.0)"

        spec_both = ParameterizedLoss(simple_fn, (1,), (scale = 2.0,))
        result = sprint(io -> _format_loss_spec(io, spec_both), context = :color => false)
        @test result == "simple_fn(1; scale=2.0)"
    end

    @testset "PerTarget" begin
        pt = PerTarget(
            (
                SymbolicLoss(:mse),
                FunctionLoss(identity_fn),
            )
        )
        result = sprint(io -> _format_loss_spec(io, pt), context = :color => false)
        @test result == "PerTarget(:mse, identity_fn)"
    end

    @testset "ExtraLoss" begin
        spec_none = ExtraLoss(nothing)
        result = sprint(io -> _format_loss_spec(io, spec_none), context = :color => false)
        @test result == "nothing"

        result = sprint(io -> _format_loss_spec(io, ExtraLoss(simple_fn)), context = :color => false)
        @test result == "simple_fn"
    end

    @testset "LoggingLoss compact printing" begin
        ll = LoggingLoss(
            loss_types = [:mse],
            training_loss = :mse,
            agg = sum
        )

        result = sprint(show, MIME"text/plain"(), ll; context = (:compact => true, :color => false))

        @test result == "LoggingLoss(:mse)"
    end

    @testset "LoggingLoss full printing" begin
        ll = LoggingLoss(
            loss_types = [:mse],
            training_loss = :mse,
            extra_loss = nothing,
            agg = sum,
            train_mode = true
        )

        expected = "LoggingLoss(\n  loss_types = [:mse],\n  training_loss = :mse,\n  extra_loss = nothing,\n  agg = sum,\n  train_mode = true\n)"
        result = sprint(show, MIME"text/plain"(), ll; context = :color => false)

        @test result == expected
    end

    @testset "LoggingLoss with Tuple training_loss" begin
        @testset "Tuple training_loss - compact printing" begin
            ll = LoggingLoss(training_loss = (:mse, :mae))

            expected = "LoggingLoss(PerTarget(:mse, :mae))"
            result = sprint(show, MIME"text/plain"(), ll; context = (:compact => true, :color => false))

            @test result == expected
        end

        @testset "Tuple training_loss - full printing" begin
            ll = LoggingLoss(training_loss = (:mse, :mae), train_mode = true)

            expected = "LoggingLoss(\n  loss_types = [:mse],\n  training_loss = PerTarget(:mse, :mae),\n  extra_loss = nothing,\n  agg = sum,\n  train_mode = true\n)"
            result = sprint(show, MIME"text/plain"(), ll; context = :color => false)

            @test result == expected
        end

        @testset "Tuple training_loss with mixed types - compact printing" begin
            ll = LoggingLoss(
                loss_types = [:mse, identity_fn],
                training_loss = (:mse, FunctionLoss(identity_fn)),
                agg = sum
            )

            expected = "LoggingLoss(PerTarget(:mse, identity_fn))"
            result = sprint(show, MIME"text/plain"(), ll; context = (:compact => true, :color => false))

            @test result == expected
        end
    end

end
