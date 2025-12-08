using ForwardDiff
# using Mooncake
# using Enzyme

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
        # ("EnzymeConst",  AutoEnzyme(; function_annotation = Enzyme.Const)),
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
                hybrid_name = "test_$(backend_name)",
                autodiff_backend = backend_fn,
            )
            @test !isnothing(out)
        end
    end
end
