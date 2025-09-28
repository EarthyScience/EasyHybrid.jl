# test/book_chapter_example_tests.jl
using Test
using Random
using EasyHybrid
using Lux
using MLUtils
using DataFrames
using DimensionalData
using ChainRulesCore
using Statistics

# ------------------------------------------------------------------------------
# Synthetic data similar to the example's columns (no network calls)
# ------------------------------------------------------------------------------
function make_synth_df(n::Int=512; seed::Int=42)
    rng = MersenneTwister(seed)
    ta = 10 .+ 10 .* randn(rng, n)                # air temperature [°C]
    sw_pot = abs.(50 .+ 20 .* randn(rng, n))      # solar radiation-ish
    dsw_pot = vcat(0.0, diff(sw_pot))             # simple derivative
    true_Q10 = 2.0
    true_rb  = 3.0 .+ 0.02 .* (sw_pot .- mean(sw_pot))
    tref = 15.0
    reco = true_rb .* (true_Q10 .^ (0.1 .* (ta .- tref))) .+ 0.1 .* randn(rng, n)
    DataFrame(; ta = Float32.(ta),
                 sw_pot = Float32.(sw_pot),
                 dsw_pot = Float32.(dsw_pot),
                 reco = Float32.(reco),
                 id = 1:n)
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
    rb  = (3.0f0, 0.0f0, 13.0f0),
    Q10 = (2.0f0, 1.0f0, 4.0f0),
)

# ------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------
@testset "Book Chapter Example - RbQ10 Hybrid" begin
    df = make_synth_df(32)  # keep it small/fast

    forcing   = [:ta]
    predictors = [:sw_pot, :dsw_pot]
    target    = [:reco]
    global_param_names = [:Q10]
    neural_param_names = [:rb]

    @testset "constructHybridModel test" begin
        model = constructHybridModel(
            predictors, forcing, target, RbQ10,
            RbQ10_PARAMS, neural_param_names, global_param_names
        )
        @test model isa SingleNNHybridModel
    end

    @testset "test DataFrame and thereby KeyedArray" begin
        # prepare_data should produce something consumable by split_data
        ka = prepare_data(model, df)
        @test !isnothing(ka)

        trainshort(ka; kwargs...) = train(model, ka, ();
            nepochs = 1,
            batchsize = 12,
            plotting = false,
            show_progress = false,
            kwargs...
        )

        out = trainshort(ka)
        @test !isnothing(out)

        out = trainshort(ka, shuffleobs = true)
        @test !isnothing(out)

        out = trainshort(ka, split_data_at = 0.8)
        @test !isnothing(out)

        out = trainshort(ka, shuffleobs = true, split_data_at = 0.8)
        @test !isnothing(out)

        #only doable on df not ka, since that row gets deleted at the moment
        out = trainshort(df, split_by_id = :id)
        @test !isnothing(out)

        out = trainshort(ka, split_by_id = df.id)
        @test !isnothing(out)

        out = trainshort(ka, split_by_id = df.id, shuffleobs = true)
        @test !isnothing(out)

        out = trainshort(ka, split_by_id = df.id, shuffleobs = false)
        @test !isnothing(out)

        folds = make_folds(df, k=3, shuffle=true)
        @test !isnothing(folds)

        df.folds = folds

        out = trainshort(ka, folds = folds, val_fold = 1)
        @test !isnothing(out)

        out = trainshort(ka, folds = :folds, val_fold = 1)
        @test !isnothing(out)

        out = trainshort(df, folds = :folds, val_fold = 1, shuffleobs = true)
        @test !isnothing(out)

        out = trainshort(df, folds = :folds, val_fold = 1, shuffleobs = true, split_by_id = :id)

        @test_throws ArgumentError trainshort(df; folds = :folds, val_fold = 1, shuffleobs = true, split_by_id = :id)
    end

    @testset "test split_data" begin
        # prepare_data should produce something consumable by split_data

        sdata = split_data(df, model, split_by_id = :id)
        @test !isnothing(sdata)

        out = trainshort(sdata)
        @test !isnothing(out)
    end
end
