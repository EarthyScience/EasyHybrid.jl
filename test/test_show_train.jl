using Test
using EasyHybrid
using EasyHybrid: TrainResults
using DataFrames

@testset "show_train.jl" begin

    @testset "_print_nested_keys" begin
        # Test scalars, arrays, nested NamedTuples, empty and non-empty tuples
        nt = (
            scalar = 1.0,
            array = [1, 2, 3],
            nested = (x = 1.0, y = 2.0),
            empty_tuple = (),
            non_empty_tuple = (1.0, 2.0, 3.0),  # Non-empty tuple to cover tuple length printing
        )
        result = sprint(io -> EasyHybrid._print_nested_keys(io, nt; indent = 4), context = :color => false)

        @test occursin("scalar", result) && occursin("array", result) && occursin("nested", result)
        @test occursin("(x, y)", result)  # Nested property names
        @test occursin("(3,)", result)  # Array size
        @test occursin("non_empty_tuple", result) && occursin("(3,)", result)  # Non-empty tuple length
        @test occursin("empty_tuple", result) && occursin("()", result)  # Empty tuple
    end

    @testset "Base.show for TrainResults" begin
        # Comprehensive test covering all field types
        train_history = [(mse = (reco = 1.0, sum = 0.5), r2 = (reco = 0.9, sum = 0.45))]
        val_history = [(mse = (reco = 1.1, sum = 0.55), r2 = (reco = 0.88, sum = 0.44))]
        ps_history = [(ϕ = (), monitor = (train = (), val = ()))]
        train_obs_pred = DataFrame(reco = [1.0, 2.0], index = [1, 2], reco_pred = [1.1, 2.1])
        val_obs_pred = DataFrame(reco = [3.0], index = [3], reco_pred = [3.1])
        train_diffs = (Q10 = [2.0], rb = [1.0, 2.0], parameters = (rb = [1.0], Q10 = [2.0]))
        val_diffs = (Q10 = [2.0], rb = [3.0], parameters = (rb = [3.0], Q10 = [2.0]))
        ps = ([1.0, 2.0],)  # Tuple, (not really the full expected type, but it's a tuple)
        st = (st_nn = (layer_1 = (), layer_2 = ()), fixed = ())
        best_epoch = 42
        best_loss = 0.123

        tr = TrainResults(
            train_history, val_history, ps_history, train_obs_pred, val_obs_pred,
            train_diffs, val_diffs, ps, st, best_epoch, best_loss
        )

        result = sprint(show, MIME"text/plain"(), tr; context = :color => false)

        # All fields present
        @test all(
            occursin.(
                [
                    "train_history:", "val_history:", "ps_history:", "train_obs_pred:",
                    "val_obs_pred:", "train_diffs:", "val_diffs:", "ps:", "st:",
                    "best_epoch:", "best_loss:",
                ], Ref(result)
            )
        )

        # Array sizes and nested structures
        @test occursin("(1,)", result)
        @test occursin("(reco, sum)", result) && occursin("(train, val)", result)
        @test occursin("(rb, Q10)", result)

        # DataFrame format
        @test occursin("DataFrame", result) && occursin("reco", result) && occursin("index", result)

        # Scalar values printed
        @test occursin("42", result) && occursin("0.123", result)

        # Empty arrays and tuples handled
        tr_empty = TrainResults([], [], [], DataFrame(), DataFrame(), nothing, nothing, (), (), 0, 0.0)
        result_empty = sprint(show, MIME"text/plain"(), tr_empty; context = :color => false)
        @test occursin("(0,)", result_empty) && occursin("best_epoch:", result_empty)
    end

end
