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

