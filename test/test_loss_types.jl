using Test
using EasyHybrid
using EasyHybrid: SymbolicLoss, FunctionLoss, ParameterizedLoss, ExtraLoss, _format_loss_spec, LoggingLoss, PerTarget
using EasyHybrid: loss_name, loss_spec, _to_extra_loss_spec

identity_fn(x) = x
simple_fn(x, a) = x + a
simple_fn(x, y; scale = 1.0) = scale * (x + y)

@testset "LoggingLoss" begin
    @testset "Constructor defaults" begin
        logging = LoggingLoss()
        @test loss_types(logging) == [:mse]
        @test training_loss(logging) == :mse
        @test extra_loss(logging) === nothing
        @test logging.agg == sum
        @test logging.train_mode == true
    end

    @testset "Custom constructor" begin
        # Simple custom loss function
        custom_loss(ŷ, y) = mean(abs2, ŷ .- y)

        # Loss function with args
        weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)

        # Loss function with kwargs
        scaled_loss(ŷ, y; scale = 1.0) = scale * mean(abs2, ŷ .- y)
        # extra loss
        extra(ŷ) = sum(abs, ŷ)

        @testset "Basic custom constructor" begin
            logging = LoggingLoss(
                loss_types = [:mse, :mae],
                training_loss = :mae,
                agg = mean,
                extra_loss = extra,
                train_mode = false
            )
            @test loss_types(logging) == [:mse, :mae]
            @test training_loss(logging) == :mae
            @test extra_loss(logging) === extra
            @test logging.agg == mean
            @test logging.train_mode == false
        end

        @testset "Mixed loss_types" begin
            logging = LoggingLoss(
                loss_types = [:mse, custom_loss, (weighted_loss, (0.5,)), (scaled_loss, (scale = 2.0,))],
                training_loss = :mse,
                agg = sum
            )
            @test length(loss_types(logging)) == 4
            @test loss_types(logging)[1] == :mse
            @test loss_types(logging)[2] == custom_loss
            @test loss_types(logging)[3] == (weighted_loss, (0.5,))
            @test loss_types(logging)[4] == (scaled_loss, (scale = 2.0,))
        end

        @testset "PerTarget Mixed loss_types" begin
            logging = LoggingLoss(
                loss_types = [:mse],
                training_loss = (
                    :mse,
                    custom_loss,
                    (weighted_loss, (0.5,)),
                    (scaled_loss, (scale = 2.0,)),
                ),
                agg = sum
            )

            @test length(training_loss(logging)) == 4
            @test first(training_loss(logging)) == :mse
            @test training_loss(logging)[2] == custom_loss
            @test training_loss(logging)[3] == (weighted_loss, (0.5,))
            @test last(training_loss(logging)) == (scaled_loss, (scale = 2.0,))
        end

        @testset "Custom training_loss variations" begin
            # Function as training_loss
            logging = LoggingLoss(
                loss_types = [:mse],
                training_loss = custom_loss
            )
            @test training_loss(logging) == custom_loss

            # Tuple with args as training_loss
            logging = LoggingLoss(
                loss_types = [:mse],
                training_loss = (weighted_loss, (0.5,))
            )
            @test training_loss(logging) == (weighted_loss, (0.5,))

            # Tuple with kwargs as training_loss
            logging = LoggingLoss(
                loss_types = [:mse],
                training_loss = (scaled_loss, (scale = 2.0,))
            )
            @test training_loss(logging) == (scaled_loss, (scale = 2.0,))

            # Tuple with both args and kwargs
            complex_loss(x, y, w; scale = 1.0) = scale * w * mean(abs2, x .- y)
            logging = LoggingLoss(
                loss_types = [:mse],
                training_loss = (complex_loss, (0.5,), (scale = 2.0,))
            )
            @test training_loss(logging) == (complex_loss, (0.5,), (scale = 2.0,))
        end
    end
end


@testset "ParameterizedLoss constructors" begin
    @testset "Basic constructor" begin
        pl = ParameterizedLoss(simple_fn)
        @test pl.f === simple_fn
        @test pl.args == ()
        @test pl.kwargs == NamedTuple()
    end

    @testset "Constructor with args" begin
        pl = ParameterizedLoss(simple_fn, (1, 2))
        @test pl.f === simple_fn
        @test pl.args == (1, 2)
        @test pl.kwargs == NamedTuple()
    end

    @testset "Constructor with kwargs" begin
        pl = ParameterizedLoss(simple_fn, (scale = 2.0,))
        @test pl.f === simple_fn
        @test pl.args == ()
        @test pl.kwargs == (scale = 2.0,)
    end
end

@testset "_to_extra_loss_spec edge cases" begin
    @testset "Nothing returns ExtraLoss(nothing)" begin
        el = _to_extra_loss_spec(nothing)
        @test el isa ExtraLoss
        @test el.f === nothing
    end
end

@testset "loss_name edge cases" begin
    @testset "SymbolicLoss returns name" begin
        @test loss_name(SymbolicLoss(:mse)) === :mse
    end

    @testset "Other LossSpecs return nothing" begin
        @test loss_name(FunctionLoss(simple_fn)) === nothing
        @test loss_name(ParameterizedLoss(simple_fn)) === nothing
        @test loss_name(ExtraLoss(simple_fn)) === nothing
    end
end


@testset "loss_spec edge cases" begin
    @testset "SymbolicLoss" begin
        ls = SymbolicLoss(:mse)
        @test loss_spec(ls) == :mse
    end

    @testset "FunctionLoss" begin
        fl = FunctionLoss(simple_fn)
        @test loss_spec(fl) === simple_fn
    end

    @testset "ParameterizedLoss" begin
        pl = ParameterizedLoss(simple_fn, (1,), (scale = 2.0,))
        @test loss_spec(pl) == (simple_fn, (1,), (scale = 2.0,))
    end

    @testset "ExtraLoss" begin
        el = ExtraLoss(simple_fn)
        @test loss_spec(el) === simple_fn
    end

    @testset "PerTarget" begin
        pt = PerTarget((SymbolicLoss(:mse), SymbolicLoss(:mae)))
        result = loss_spec(pt)
        @test result isa PerTarget
        @test result.losses == (:mse, :mae)
    end
end

@testset "PerTarget edge cases" begin
    @testset "Empty PerTarget" begin
        pt_empty = PerTarget(())
        @test length(pt_empty) == 0
        @test iterate(pt_empty) === nothing
        @test_throws ArgumentError first(pt_empty)
        @test_throws BoundsError last(pt_empty)
    end

    @testset "Single-element PerTarget" begin
        pt_single = PerTarget((SymbolicLoss(:mse),))
        @test length(pt_single) == 1
        @test first(pt_single) == SymbolicLoss(:mse)
        @test last(pt_single) == SymbolicLoss(:mse)
    end

    @testset "Iteration" begin
        pt = PerTarget((SymbolicLoss(:mse), SymbolicLoss(:mae)))
        vals = []
        for l in pt
            push!(vals, l)
        end
        @test vals == [SymbolicLoss(:mse), SymbolicLoss(:mae)]
    end
end

@testset "PerTarget Base methods" begin
    @testset "Base.length" begin
        pt = PerTarget((SymbolicLoss(:mse), SymbolicLoss(:mae), FunctionLoss(identity_fn)))
        @test length(pt) == 3

        pt_single = PerTarget((SymbolicLoss(:mse),))
        @test length(pt_single) == 1
    end

    @testset "Base.getindex" begin
        pt = PerTarget((SymbolicLoss(:mse), SymbolicLoss(:mae), FunctionLoss(identity_fn)))
        @test pt[1] == SymbolicLoss(:mse)
        @test pt[2] == SymbolicLoss(:mae)
        @test pt[3] == FunctionLoss(identity_fn)
    end

    @testset "Base.first" begin
        pt = PerTarget((SymbolicLoss(:mse), SymbolicLoss(:mae)))
        @test first(pt) == SymbolicLoss(:mse)
    end

    @testset "Base.last" begin
        pt = PerTarget((SymbolicLoss(:mse), SymbolicLoss(:mae)))
        @test last(pt) == SymbolicLoss(:mae)
    end

    @testset "Base.keys" begin
        pt = PerTarget((SymbolicLoss(:mse), SymbolicLoss(:mae)))
        @test keys(pt) == keys(pt.losses)
        @test collect(keys(pt)) == [1, 2]
    end

    @testset "Base.eltype" begin
        pt = PerTarget((SymbolicLoss(:mse), SymbolicLoss(:mae)))
        @test eltype(pt) == eltype(pt.losses)
    end
end
