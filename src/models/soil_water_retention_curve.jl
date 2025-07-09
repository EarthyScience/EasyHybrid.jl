export WaterRetentionHybrid
export Cf_FXW
export Gamma_FXW
export Se_FXW
export mFXW_theta


"""
    Cf_FXW(h, h_r, h_0)

Fredlund & Xing (1994) correction factor C_f(h).

Arguments:
- h: suction (pressure head)
- h_r: reference suction (often 1)
- h_0: large suction value (e.g., 1e6)

Returns:
- C_f(h) as an array or scalar
"""
Cf_FXW(h, h_r, h_0) = @. 1.f0 - log(1.f0 + h / h_r) / log(1.f0 + h_0 / h_r)

"""
    Gamma_FXW(h, α, n, m)

Fredlund & Xing (1994) Gamma function Γ(h).

Arguments:
- h: suction (pressure head)
- α, n, m: curve parameters

Returns:
- Γ(h) as an array or scalar
"""
Gamma_FXW(h, α, n, m) = @. (log(exp(1.f0) + abs(α * h)^n))^(-m)

"""
    Se_FXW(h, h_r, h_0, α, n, m)

Effective saturation S_e(h) using Fredlund & Xing (1994).

Arguments:
- h: suction (pressure head)
- h_r: reference suction
- h_0: large suction value
- α, n, m: curve parameters

Returns:
- S_e(h) as an array or scalar
"""
Se_FXW(h, h_r, h_0, α, n, m) = @. Cf_FXW(h, h_r, h_0) * Gamma_FXW(h, α, n, m)

"""
    mFXW_theta(h, θ_s, h_r, h_0, α, n, m)

Compute volumetric water content θ(h) using Fredlund & Xing (1994).

Arguments:
- h: suction (pressure head)
- θ_s: saturated water content
- h_r: reference suction
- h_0: large suction value
- α, n, m: curve parameters

Returns:
- θ(h) as an array or scalar
"""
function mFXW_theta(h, θ_s, h_r, h_0, α, n, m)
    S_e = Se_FXW(h, h_r, h_0, α, n, m)
    θ = θ_s .* S_e
    return θ
end


"""
    WaterRetentionHybrid(NN, predictors, forcing, targets, h_r, h_0)

A hybrid model with a neural network `NN`, `predictors`, `targets`, and `forcing` terms for the water retention curve. The neural network outputs the FXW parameters (θ_s, α, n, m), and h_r and h_0 are global parameters to be estimated. The forcing is the pressure head h.
"""
struct WaterRetentionHybrid{D, T1, T2, T3, T4} <: LuxCore.AbstractLuxContainerLayer{(:NN, :predictors, :forcing, :targets, :h_r, :h_0)}
    NN
    predictors
    forcing
    targets
    h_r
    h_0
    function WaterRetentionHybrid(NN::D, predictors::T1, forcing::T2, targets::T3, h_r::T4, h_0::T4) where {D, T1, T2, T3, T4}
        new{D, T1, T2, T3, T4}(NN, collect(predictors), collect(forcing), collect(targets), [h_r], [h_0])
    end
end

function LuxCore.initialparameters(::AbstractRNG, layer::WaterRetentionHybrid)
    ps, _ = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; ps, h_r = layer.h_r, h_0 = layer.h_0)
end

function LuxCore.initialstates(::AbstractRNG, layer::WaterRetentionHybrid)
    _, st = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; st)
end

"""
    WaterRetentionHybrid(NN, predictors, forcing, targets, h_r, h_0)(ds_k)

Model definition: ŷ = θ(h) = θ_s * Sₑ(h), where θ_s, α, n, m are the outputs of the neural network, h_r and h_0 are global parameters, and h is the forcing.
"""
function (hm::WaterRetentionHybrid)(ds_k, ps, st::NamedTuple)
    p = ds_k(hm.predictors)
    h = Array(ds_k(hm.forcing))[1,:]

    # NN output: θ_s, α, n, m
    nn_out, st = LuxCore.apply(hm.NN, p, ps.ps, st.st)
    θ_s, α, n, m = eachrow(nn_out)

    θ = mFXW_theta(h, θ_s, ps.h_r, ps.h_0, α, n, m)
    
    return (; θ, θ_s, α, n, m, h), (; st)

end

