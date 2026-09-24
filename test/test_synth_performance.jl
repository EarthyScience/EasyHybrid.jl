using Random
using Statistics
using Zygote
using Lux
using Logging
using DataFrames
using EasyHybrid
using EasyHybrid: compute_loss, scale_single_param, LoggingLoss

# Synthetic RbQ10 setup matching docs/literate/research/synthetic_respiration.jl
# (same columns, hidden layers, activation, scaling, and batch size; data generated locally).

const SYNTH_BATCH = 512
# Paired median of fused `compute_loss` Zygote.gradient vs a Lux same-math gradient
# on this example. New code may not exceed this by more than `SYNTH_SLOWDOWN_TOLERANCE`.
const SYNTH_COMPUTE_LOSS_VS_LUX = 1.18
const SYNTH_SLOWDOWN_TOLERANCE = 0.1

function _synth_rbq10(; n = 2048, batch = SYNTH_BATCH, seed = 42)
    rng = MersenneTwister(seed)
    ta = 10 .+ 10 .* randn(rng, n)
    sw_pot = abs.(50 .+ 20 .* randn(rng, n))
    dsw_pot = vcat(0.0, diff(sw_pot))
    true_Q10 = 2.0
    true_rb = 3.0 .+ 0.02 .* (sw_pot .- mean(sw_pot))
    reco = true_rb .* (true_Q10 .^ (0.1 .* (ta .- 15.0))) .+ 0.1 .* randn(rng, n)
    df = DataFrame(;
        ta = Float32.(ta),
        sw_pot = Float32.(sw_pot),
        dsw_pot = Float32.(dsw_pot),
        reco = Float32.(reco),
    )

    function RbQ10(; ta, Q10, rb, tref = 15.0f0)
        reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
        return (; reco, Q10, rb)
    end
    parameters = (
        rb = (3.0f0, 0.0f0, 13.0f0),
        Q10 = (2.0f0, 1.0f0, 4.0f0),
    )
    hm = constructHybridModel(
        [:sw_pot, :dsw_pot], [:ta], [:reco], RbQ10, parameters, [:rb], [:Q10];
        hidden_layers = [16, 16],
        activation = sigmoid,
        scale_nn_outputs = true,
        input_batchnorm = true,
    )
    (x_all, forcings_all), y_all = prepare_data(hm, df)
    x = Array(x_all)[:, 1:batch]
    ta_b = Array(forcings_all.ta)[1:batch]
    yv = Array(y_all.reco)[1:batch]
    mask = .!isnan.(yv)
    ds = ((x, (; ta = ta_b)), ((; reco = yv), (; reco = mask)))
    ps, st = LuxCore.setup(MersenneTwister(seed), hm)
    logging = LoggingLoss(; training_loss = :mse, loss_types = [:mse], train_mode = true, agg = sum)
    return (; hm, ps, st, ds, logging, df, x, ta_b, yv, mask)
end

function _median_s(f; warmup = 4, samples = 21)
    for _ in 1:warmup
        f()
    end
    GC.gc()
    times = Vector{Float64}(undef, samples)
    for i in 1:samples
        times[i] = @elapsed f()
    end
    return median(times)
end

function _paired_median_ratio(f_a, f_b; warmup = 8, samples = 41)
    for _ in 1:warmup
        f_a()
        f_b()
    end
    GC.gc()
    ratios = Vector{Float64}(undef, samples)
    for i in 1:samples
        if isodd(i)
            ta = @elapsed f_a()
            tb = @elapsed f_b()
        else
            tb = @elapsed f_b()
            ta = @elapsed f_a()
        end
        ratios[i] = ta / tb
    end
    return median(ratios)
end

function _lux_same_math(s)
    hm, st, x, ta, yv, mask = s.hm, s.st, s.x, s.ta_b, s.yv, s.mask
    bounds = hm.parameters
    return function (p)
        o, _ = LuxCore.apply(hm.NNs, x, p.ps, st.st_nn)
        rb = scale_single_param(:rb, eachslice(o, dims = 1)[1], bounds)
        Q10 = scale_single_param(:Q10, p.Q10, bounds)
        reco = rb .* Q10 .^ (0.1f0 .* (ta .- 15.0f0))
        return mean(abs2, reco[mask] .- yv[mask])
    end
end

@testset "Synthetic RbQ10 performance" begin
    with_logger(NullLogger()) do
        s = _synth_rbq10()
        hm, ps, st, ds, logging = s.hm, s.ps, s.st, s.ds, s.logging
        lux_loss = _lux_same_math(s)
        loss_fn = (model, p, st_, data) -> compute_loss(model, p, st_, data, logging)

        f_fwd = () -> hm((s.x, (; ta = s.ta_b)), ps, st)
        f_loss = () -> compute_loss(hm, ps, st, ds, logging)
        f_lux = () -> Zygote.gradient(lux_loss, ps)
        f_cl = () -> Zygote.gradient(p -> compute_loss(hm, p, st, ds, logging)[1], ps)
        f_hm_mse = () -> Zygote.gradient(
            p -> begin
                out, _ = hm((s.x, (; ta = s.ta_b)), p, st)
                mean(abs2, out.reco[s.mask] .- s.yv[s.mask])
            end, ps
        )
        f_step = () -> begin
            ts = Lux.Training.TrainState(hm, ps, st, AdamW(0.1))
            Lux.Training.single_train_step!(AutoZygote(), loss_fn, ds, ts)
        end

        f_lux(); f_fwd(); f_cl(); f_hm_mse(); f_step()

        t_fwd = _median_s(f_fwd)
        t_loss = _median_s(f_loss)
        t_lux = _median_s(f_lux)
        t_cl = _median_s(f_cl)
        t_hm_mse = _median_s(f_hm_mse)
        t_step = _median_s(f_step)
        ratio = _paired_median_ratio(f_cl, f_lux)
        cap = SYNTH_COMPUTE_LOSS_VS_LUX * (1 + SYNTH_SLOWDOWN_TOLERANCE)

        println("synthetic RbQ10 steps (median µs):")
        println("  HybridModel forward           ", round(t_fwd * 1.0e6; digits = 1))
        println("  compute_loss value            ", round(t_loss * 1.0e6; digits = 1))
        println("  Lux same-math gradient        ", round(t_lux * 1.0e6; digits = 1))
        println("  compute_loss gradient         ", round(t_cl * 1.0e6; digits = 1))
        println("  HybridModel + MSE gradient    ", round(t_hm_mse * 1.0e6; digits = 1))
        println("  single_train_step!            ", round(t_step * 1.0e6; digits = 1))
        println("  compute_loss / Lux (paired)   ", round(ratio; digits = 3), "  (cap ", round(cap; digits = 3), ")")

        @testset "step: HybridModel forward" begin
            @test t_fwd > 0
        end
        @testset "step: compute_loss value" begin
            @test t_loss > 0
            cl = compute_loss(hm, ps, st, ds, logging)[1]
            out, _ = hm((s.x, (; ta = s.ta_b)), ps, st)
            @test cl ≈ mean(abs2, out.reco[s.mask] .- s.yv[s.mask]) rtol = 1.0e-4
            @test cl ≈ lux_loss(ps) rtol = 1.0e-4
        end
        @testset "step: compute_loss gradient vs Lux" begin
            @test ratio ≤ SYNTH_COMPUTE_LOSS_VS_LUX * (1 + SYNTH_SLOWDOWN_TOLERANCE)
        end
        @testset "step: compute_loss not slower than public HybridModel MSE" begin
            @test t_cl ≤ t_hm_mse * (1 + SYNTH_SLOWDOWN_TOLERANCE)
        end
        @testset "step: single_train_step!" begin
            @test t_step > 0
        end
    end
end
