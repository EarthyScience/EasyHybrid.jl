

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
    # Correction factor C_f(h)
    C_f = @. 1.f0 - log(1.f0 + h / h_r) / log(1.f0 + h_0 / h_r)
    
    # Gamma function Γ(h)
    Γ = @. (log(exp(1.f0) + abs(α * h)^n))^(-m)
    
    # Effective saturation S_e(h)
    S_e = @. C_f * Γ
    
    # Volumetric water content θ(h)
    θ = θ_s .* S_e

    return θ
end

