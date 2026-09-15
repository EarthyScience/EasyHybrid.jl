using EasyHybrid
using Test
using Random
using ComponentArrays

# Custom mechanistic model for testing
function custom_flux_mechanistic_model(; PAR, Tair, a, b, c)
    flux = @. a * PAR + b * Tair + c
    return (; flux = flux)
end

# Synthetic mechanistic model
function simple_mechanistic_model(; Rg, VPD, alpha, beta, gamma)
    y = @. alpha * Rg + beta * VPD + gamma
    return (; target = y)
end

@testset "LuxZarr HybridModel I/O Suite" begin
    rng = Random.Xoshiro(42)

    params_dict = (
        alpha = (0.5, 0.0, 1.0, :linear),
        beta = (1.0, 0.1, 5.0, :log),
        gamma = (0.0, -2.0, 2.0, :linear),
    )

    @testset "Single-NN HybridModel Serialization & Roundtrip" begin
        hm = constructHybridModel(
            [:Rg, :VPD],
            [:Rg, :VPD],
            [:target],
            simple_mechanistic_model,
            params_dict,
            [:alpha],
            [:beta];
            hidden_layers = [8, 8],
            activation = tanh,
            scale_nn_outputs = true,
        )

        ps, st = LuxCore.setup(rng, hm)
        N = 10
        Rg_val = rand(rng, Float32, 1, N)
        VPD_val = rand(rng, Float32, 1, N)
        inputs = (
            vcat(Rg_val, VPD_val),
            (; Rg = vec(Rg_val), VPD = vec(VPD_val)),
        )

        out_orig, st_orig = hm(inputs, ps, st)

        mktempdir() do tmpdir
            save_path = joinpath(tmpdir, "single_nn_hm.zarr")

            # 1. Save model
            save_model(save_path, ps, st; model = hm, force = true)

            # 2. Standalone Lazy Load
            lazy_hm = load_model(save_path; lazy = true)
            @test lazy_hm isa LazyLuxModel
            out_lazy, _ = lazy_hm(inputs)
            @test isapprox(out_lazy.target, out_orig.target; rtol = 1.0e-5)
            @test isapprox(out_lazy.parameters.alpha, out_orig.parameters.alpha; rtol = 1.0e-5)
            @test isapprox(out_lazy.parameters.beta, out_orig.parameters.beta; rtol = 1.0e-5)

            # 3. Standalone Eager Load
            ps_eager, st_eager, hm_eager = load_model(save_path; lazy = false)
            @test ps_eager isa NamedTuple
            @test hm_eager isa HybridModel
            out_eager, _ = hm_eager(inputs, ps_eager, st_eager)
            @test isapprox(out_eager.target, out_orig.target; rtol = 1.0e-5)

            # 4. Guided Load with skeleton
            ps_skel, st_skel = LuxCore.setup(rng, hm)
            ps_guided, st_guided = load_model(save_path, ps_skel, st_skel; lazy = false)
            out_guided, _ = hm(inputs, ps_guided, st_guided)
            @test isapprox(out_guided.target, out_orig.target; rtol = 1.0e-5)
        end
    end

    @testset "Multi-NN HybridModel Serialization & Roundtrip" begin
        multi_predictors = (;
            alpha = [:Rg],
            beta = [:VPD],
        )

        hm_multi = constructHybridModel(
            multi_predictors,
            [:Rg, :VPD],
            [:target],
            simple_mechanistic_model,
            params_dict,
            Symbol[];
            hidden_layers = [8],
            activation = tanh,
            scale_nn_outputs = true,
        )

        ps_m, st_m = LuxCore.setup(rng, hm_multi)
        N = 8
        Rg_val = rand(rng, Float32, 1, N)
        VPD_val = rand(rng, Float32, 1, N)
        inputs_m = (
            (; alpha = Rg_val, beta = VPD_val),
            (; Rg = vec(Rg_val), VPD = vec(VPD_val)),
        )

        out_orig_m, _ = hm_multi(inputs_m, ps_m, st_m)

        mktempdir() do tmpdir
            save_path = joinpath(tmpdir, "multi_nn_hm.zarr")
            save_model(save_path, ps_m, st_m; model = hm_multi, force = true)

            # Standalone lazy load
            lazy_hm_m = load_model(save_path; lazy = true)
            @test lazy_hm_m isa LazyLuxModel
            out_lazy_m, _ = lazy_hm_m(inputs_m)
            @test isapprox(out_lazy_m.target, out_orig_m.target; rtol = 1.0e-5)
        end
    end

    @testset "ComponentArray Weights Serialization" begin
        hm = constructHybridModel(
            [:Rg, :VPD],
            [:Rg, :VPD],
            [:target],
            simple_mechanistic_model,
            params_dict,
            [:alpha],
            [:beta];
            hidden_layers = [4],
            activation = tanh,
        )

        ps, st = LuxCore.setup(rng, hm)
        ps_ca = ComponentArray(ps)

        mktempdir() do tmpdir
            save_path = joinpath(tmpdir, "ca_hm.zarr")
            save_model(save_path, ps_ca, st; model = hm, force = true)

            # Guided load into ComponentArray skeleton
            ps_loaded, st_loaded = load_model(save_path, ps_ca, st; lazy = false)
            @test isapprox(Array(ps_loaded.ps.layer_2.weight), Array(ps_ca.ps.layer_2.weight))
            @test isapprox(Array(ps_loaded.beta), Array(ps_ca.beta))
        end
    end

    @testset "Custom User Function Resolution & Override" begin
        custom_params = (
            a = (0.2, 0.0, 1.0, :linear),
            b = (0.5, 0.0, 2.0, :linear),
            c = (0.0, -1.0, 1.0, :linear),
        )

        hm_custom = constructHybridModel(
            [:PAR, :Tair],
            [:PAR, :Tair],
            [:flux],
            custom_flux_mechanistic_model,
            custom_params,
            [:a],
            [:b];
            hidden_layers = [8],
            activation = tanh,
        )

        ps_c, st_c = LuxCore.setup(rng, hm_custom)
        N = 6
        inputs_c = (
            rand(rng, Float32, 2, N),
            (; PAR = rand(rng, Float32, N), Tair = rand(rng, Float32, N)),
        )

        out_orig_c, _ = hm_custom(inputs_c, ps_c, st_c)

        mktempdir() do tmpdir
            save_path = joinpath(tmpdir, "custom_fn_hm.zarr")
            save_model(save_path, ps_c, st_c; model = hm_custom, force = true)

            # 1. Automatic source / name resolution
            lazy_c = load_model(save_path; lazy = true)
            out_c, _ = lazy_c(inputs_c)
            @test isapprox(out_c.flux, out_orig_c.flux; rtol = 1.0e-5)

            # 2. Explicit mechanistic_model override
            ps_e, st_e, hm_e = load_model(save_path; mechanistic_model = custom_flux_mechanistic_model, lazy = false)
            @test hm_e.mechanistic_model === custom_flux_mechanistic_model
            out_override, _ = hm_e(inputs_c, ps_e, st_e)
            @test isapprox(out_override.flux, out_orig_c.flux; rtol = 1.0e-5)
        end
    end
end
