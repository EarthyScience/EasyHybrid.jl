using ForwardDiff
# using Mooncake
using Enzyme
using Lux
using Random
using DataFrames
using Statistics

Enzyme.API.strictAliasing!(false)

@isdefined(make_synth_df) || function make_synth_df(n::Int = 64; seed::Int = 42)
    rng = MersenneTwister(seed)
    ta = 10 .+ 10 .* randn(rng, n)
    sw_pot = abs.(50 .+ 20 .* randn(rng, n))
    dsw_pot = vcat(0.0, diff(sw_pot))
    true_Q10 = 2.0
    true_rb = 3.0 .+ 0.02 .* (sw_pot .- mean(sw_pot))
    reco = true_rb .* (true_Q10 .^ (0.1 .* (ta .- 15.0))) .+ 0.1 .* randn(rng, n)
    return DataFrame(; ta = Float32.(ta), sw_pot = Float32.(sw_pot), dsw_pot = Float32.(dsw_pot), reco = Float32.(reco))
end

@isdefined(RbQ10) || (RbQ10(; ta, Q10, rb, tref = 15.0f0) = (; reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref)), Q10, rb))
@isdefined(RbQ10_PARAMS) || (const RbQ10_PARAMS = (rb = (3.0f0, 0.0f0, 13.0f0), Q10 = (2.0f0, 1.0f0, 4.0f0)))

@testset "autodiff backends" verbose = true begin
    df = make_synth_df(32)  # keep it small/fast

    forcing = [:ta]
    predictors = [:sw_pot, :dsw_pot]
    target = [:reco]
    global_param_names = [:Q10]
    neural_param_names = [:rb]

    model = constructHybridModel(
        predictors, forcing, target, RbQ10,
        RbQ10_PARAMS, neural_param_names, global_param_names
    )

    ka = prepare_data(model, df)

    _BACKENDS_SPEC = (
        ("EnzymeConst", AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Reverse), function_annotation = Enzyme.Const)),
        ("ForwardDiff", AutoForwardDiff()),
        # ("Mooncake", AutoMooncake(; config = nothing)), # ? it needs special rrules
        ("Zygote", AutoZygote()),
    )
    for (backend_name, backend_fn) in _BACKENDS_SPEC
        @testset "backend: $backend_name" begin
            out = train(
                model, ka, ();
                nepochs = 1,
                batchsize = 12,
                plotting = false,
                show_progress = false,
                model_name = "test_$(backend_name)",
                autodiff_backend = backend_fn,
            )
            @test !isnothing(out)
        end
    end
end
