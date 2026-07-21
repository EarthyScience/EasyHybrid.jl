using Random
using DataFrames
using Statistics

function make_synth_df(n::Int = 64; seed::Int = 42)
    rng = MersenneTwister(seed)
    ta = 10 .+ 10 .* randn(rng, n)
    sw_pot = abs.(50 .+ 20 .* randn(rng, n))
    dsw_pot = vcat(0.0, diff(sw_pot))
    true_Q10 = 2.0
    true_rb = 3.0 .+ 0.02 .* (sw_pot .- mean(sw_pot))
    reco = true_rb .* (true_Q10 .^ (0.1 .* (ta .- 15.0))) .+ 0.1 .* randn(rng, n)
    return DataFrame(; ta = Float32.(ta), sw_pot = Float32.(sw_pot), dsw_pot = Float32.(dsw_pot), reco = Float32.(reco))
end

RbQ10(; ta, Q10, rb, tref = 15.0f0) = (; reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref)), Q10, rb)

const RbQ10_PARAMS = (rb = (3.0f0, 0.0f0, 13.0f0), Q10 = (2.0f0, 1.0f0, 4.0f0))

@testset "cv_test" begin
    df = make_synth_df(64)
    model = constructHybridModel(
        [:sw_pot, :dsw_pot], [:ta], [:reco], RbQ10, RbQ10_PARAMS, [:rb], [:Q10],
        hidden_layers = [8], activation = sigmoid, scale_nn_outputs = true, input_batchnorm = true,
    )
    kw = (; nepochs = 1, batchsize = 16, patience = 10, plotting = false, show_progress = false, save_training = false)

    @testset "cv only" begin
        r = cv_test(model, df; k = 2, opt = RMSProp(0.001), kw...)
        @test r isa CVTestResults
        @test r.mode === :cv
        @test length(r.cv_fold_results) == 2
        @test length(cv_fold_losses(r)) == 2
        @test r.mean_cv_loss > 0
        @test r.test_loss === nothing
        @test r.ho === nothing
    end

    @testset "folds from column name" begin
        df2 = copy(df)
        df2.fold = repeat(1:2, outer = cld(nrow(df2), 2))[1:nrow(df2)]
        r = cv_test(model, df2; k = 2, folds = :fold, opt = RMSProp(0.001), kw...)
        @test length(cv_fold_losses(r)) == 2
    end

    @testset "non-contiguous folds rejected" begin
        bad = fill(1, nrow(df)); bad[end] = 3  # labels {1,3}, missing 2
        @test_throws ArgumentError cv_test(model, df; k = 3, folds = bad, opt = RMSProp(0.001), kw...)
    end

    @testset "maximize metric selects best (not worst)" begin
        # With an :nse-first metric, higher is better; the chosen model must be the argmax.
        Random.seed!(11)
        r = cv_test(model, df;
            k = 2, nhyper = 3,
            loss_types = [:nse], training_loss = :nseLoss, agg = mean,
            hyper = (; opt = [RMSProp(0.001), AdamW(0.01)]),
            kw...,
        )
        @test r.ho !== nothing
        scores = Float64.(r.ho.results)
        @test isapprox(r.mean_cv_loss, maximum(scores); atol = 1e-8)  # :nse ⇒ maximize
    end

    @testset "hyper under the hood" begin
        r = cv_test(model, df;
            k = 2,
            nhyper = 2,
            hyper = (; opt = [RMSProp(0.001), AdamW(0.01)], input_batchnorm = [true, false]),
            kw...,
        )
        @test r.ho !== nothing
        @test haskey(r.best_hyperparams, :opt)
        @test r.mean_cv_loss > 0
    end

    @testset "random test fold" begin
        Random.seed!(123)
        r = cv_test(model, df; k = 3, test_fold = :random, opt = RMSProp(0.001), kw...)
        @test 1 ≤ r.test_fold ≤ 3
        @test r.test_loss > 0
        @test r.final_train !== nothing
    end

    @testset "hyper + random test fold" begin
        Random.seed!(7)
        r = cv_test(model, df;
            k = 3,
            test_fold = :random,
            nhyper = 2,
            hyper = (; opt = [RMSProp(0.001), AdamW(0.01)], input_batchnorm = [true, false]),
            kw...,
        )
        @test r.test_fold isa Int
        @test r.ho !== nothing
        @test r.test_loss > 0
    end

    @testset "nested test folds (:all)" begin
        r = cv_test(model, df; k = 3, test_fold = :all, opt = RMSProp(0.001), kw...)
        @test r.mode === :nested
        @test length(r.held_outs) == 3
        @test [x.test_fold for x in r.held_outs] == 1:3
        @test r.mean_test_loss > 0
        @test r.pooled_test_loss !== nothing && isfinite(r.pooled_test_loss)
        @test r.pooled_val_loss !== nothing && isfinite(r.pooled_val_loss)
        @test all(x -> x.test_obs_pred isa DataFrame, r.held_outs)
        s = sprint(show, MIME"text/plain"(), r)
        @test occursin("pooled test", s)
        @test !occursin("mean_cv", s)
        @test !occursin("rotate", s)
    end

    @testset "LHSampler + parallel hyper" begin
        r = cv_test(model, df;
            k = 2,
            nhyper = 2,
            sampler = LHSampler(),
            parallel = :hyper,
            hyper = (;
                opt = [RMSProp(0.001), AdamW(0.01)],
                input_batchnorm = [true, false],
            ),
            kw...,
        )
        @test r.ho !== nothing
        @test r.mean_cv_loss > 0
    end

    @testset "parallel folds" begin
        r = cv_test(model, df; k = 2, parallel = :folds, opt = RMSProp(0.001), kw...)
        @test length(r.cv_fold_results) == 2
        @test r.mean_cv_loss > 0
    end

    @testset "reject invalid parallel" begin
        @test_throws ArgumentError cv_test(model, df; k = 2, parallel = :both, opt = RMSProp(0.001), kw...)
    end

    @testset "parallel auto chooses hyper when nhyper larger" begin
        r = cv_test(model, df;
            k = 2,
            nhyper = 4,
            parallel = :auto,
            hyper = (; opt = [RMSProp(0.001), AdamW(0.01)], input_batchnorm = [true, false]),
            kw...,
        )
        @test r.ho !== nothing
        @test r.mean_cv_loss > 0
    end

    @testset "compact show" begin
        r = cv_test(model, df; k = 2, opt = RMSProp(0.001), kw...)
        s = sprint(show, MIME"text/plain"(), r)
        @test occursin("CVTestResults", s)
        @test occursin("best val", s)
        @test occursin("mode=:cv", s)
        @test !occursin("ŷ_train", s)
        @test !occursin("DataFrame", s)
        @test !occursin("trial", s)
    end
end
