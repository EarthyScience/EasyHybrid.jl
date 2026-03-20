# test/book_chapter_example_tests.jl
using Random
using Lux
using DataFrames
using Statistics
using DimensionalData
using ChainRulesCore
using GPUArraysCore

GPUArraysCore.allowscalar(false)

# ------------------------------------------------------------------------------
# Synthetic data similar to the example's columns (no network calls)
# ------------------------------------------------------------------------------
function make_synth_df(n::Int = 512; seed::Int = 42)
    rng = MersenneTwister(seed)
    ta = 10 .+ 10 .* randn(rng, n)                # air temperature [°C]
    sw_pot = abs.(50 .+ 20 .* randn(rng, n))      # solar radiation-ish
    dsw_pot = vcat(0.0, diff(sw_pot))             # simple derivative
    true_Q10 = 2.0
    true_rb = 3.0 .+ 0.02 .* (sw_pot .- mean(sw_pot))
    tref = 15.0
    reco = true_rb .* (true_Q10 .^ (0.1 .* (ta .- tref))) .+ 0.1 .* randn(rng, n)
    return DataFrame(;
        ta = Float32.(ta),
        sw_pot = Float32.(sw_pot),
        dsw_pot = Float32.(dsw_pot),
        reco = Float32.(reco),
        id = 1:n
    )
end

# ------------------------------------------------------------------------------
# RbQ10 physical model (from example)
# ------------------------------------------------------------------------------
function RbQ10(; ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end

# Parameter spec analogous to the example
const RbQ10_PARAMS = (
    rb = (3.0f0, 0.0f0, 13.0f0),
    Q10 = (2.0f0, 1.0f0, 4.0f0),
)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
@testset "Book Chapter Example - RbQ10 Hybrid" begin
    df = make_synth_df(32)  # keep it small/fast

    forcing = [:ta]
    predictors = [:sw_pot, :dsw_pot]
    target = [:reco]
    global_param_names = [:Q10]
    neural_param_names = [:rb]

    @testset "test DataFrame and thereby KeyedArray" begin
        model = constructHybridModel(
            predictors, forcing, target, RbQ10,
            RbQ10_PARAMS, neural_param_names, global_param_names
        )
        @test model isa SingleNNHybridModel
        # prepare_data should produce something consumable by split_data
        ka = prepare_data(model, df)
        @test !isnothing(ka)

        trainshort(ka; kwargs...) = train(
            model, ka, ();
            nepochs = 1,
            batchsize = 12,
            plotting = false,
            show_progress = false,
            kwargs...
        )

        out = trainshort(ka; model_name = "test_1")
        @test !isnothing(out)

        out = trainshort(ka; shuffleobs = true, model_name = "test_2")
        @test !isnothing(out)

        out = trainshort(ka; split_data_at = 0.8, model_name = "test_3")
        @test !isnothing(out)

        out = trainshort(ka; shuffleobs = true, split_data_at = 0.8, model_name = "test_4")
        @test !isnothing(out)

        #only doable on df not ka, since that row gets deleted at the moment
        out = trainshort(df; split_by_id = :id, model_name = "test_5")
        @test !isnothing(out)

        out = trainshort(ka; split_by_id = df.id, model_name = "test_6")
        @test !isnothing(out)

        out = trainshort(ka; split_by_id = df.id, shuffleobs = true, model_name = "test_7")
        @test !isnothing(out)

        out = trainshort(ka; split_by_id = df.id, shuffleobs = false, model_name = "test_8")
        @test !isnothing(out)

        folds = make_folds(df, k = 3, shuffle = true)
        @test !isnothing(folds)

        df.folds = folds

        out = trainshort(ka; folds = folds, val_fold = 1, model_name = "test_9")
        @test !isnothing(out)

        out = trainshort(df; folds = :folds, val_fold = 1, model_name = "test_10")
        @test !isnothing(out)

        out = trainshort(df; folds = :folds, val_fold = 1, shuffleobs = true, model_name = "test_11")
        @test !isnothing(out)

        @test_throws ArgumentError trainshort(df; folds = :folds, val_fold = 1, shuffleobs = true, split_by_id = :id)

        sdata = split_data(df, model, split_by_id = :id)
        @test !isnothing(sdata)

        out = trainshort(sdata; model_name = "test_12")
        @test !isnothing(out)

        mat = vcat(ka[1], ka[2])
        da = DimArray(mat, (Dim{:variable}(mat.keys[1]), Dim{:batch_size}(1:size(mat, 2))))'
        ka = prepare_data(model, da)
        @test !isnothing(ka)

        # TODO: this is not working, transpose da columns to rows?
        #dtuple_tuple = split_data(da, model)
        #@test !isnothing(dtuple_tuple)
        # TODO: this is not working, need to fix GenericHybrid Model for DimensionalData
        # out = trainshort(dtuple_tuple)
    end

    @testset "test keep_history" begin
        model = constructHybridModel(
            predictors, forcing, target, RbQ10,
            RbQ10_PARAMS, neural_param_names, global_param_names
        )
        @test model isa SingleNNHybridModel
        # prepare_data should produce something consumable by split_data
        ka = prepare_data(model, df)
        @test !isnothing(ka)

        out_1 = train(
            model, ka, ();
            nepochs = 5,
            batchsize = 12,
            plotting = false,
            show_progress = false,
            keep_history = true,
            hybrid_name = "keep_history_1"
        )
        out_2 = train(
            model, ka, ();
            nepochs = 5,
            batchsize = 12,
            plotting = false,
            show_progress = false,
            keep_history = false,
            hybrid_name = "keep_history_2"
        )
        @test length(out_1.epoch_history) == 6 # 5 epochs + initial values
        @test length(out_2.epoch_history) == 1

    end
end
