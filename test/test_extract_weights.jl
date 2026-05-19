using EasyHybrid
using Lux
using Random
using Test
using Zygote

@testset "extract_weights / weight_l2" begin
    rng = Random.default_rng()
    chain = Chain(
        BatchNorm(2),
        Dense(2 => 16, tanh),
        Dense(16 => 1),
    )
    ps, _ = Lux.setup(rng, chain)

    ws = extract_weights(ps)
    @test !isempty(ws)
    @test all(w -> w isa AbstractMatrix, ws)

    manual_l2 = sum(sum(abs2, w) for w in ws)
    @test weight_l2(ps) ≈ manual_l2

    λ = 1.0f-3
    loss(ps) = λ * weight_l2(ps)
    g = Zygote.gradient(loss, ps)[1]
    @test g !== nothing
    g_ws = extract_weights(g)
    @test !isempty(g_ws)
    @test all(gw -> all(!iszero, gw), g_ws)

    # biases are not regularized by default
    @test weight_l2(ps; key = :bias) ≈ sum(sum(abs2, b) for b in extract_weights(ps; key = :bias))
end
