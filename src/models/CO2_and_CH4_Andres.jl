using DifferentialEquations
using Plots

# --------------------------------------------------------
# Shared parameters
# --------------------------------------------------------

depth = 0.1   # m (active layer depth)

# Seasonal drivers
Tmean, Tamp = 8.0, 10.0
T_C = t -> Tmean + Tamp * sin(2π * (t - 200.0) / 365)
θmean, θamp = 0.55, 0.10
θ = t -> clamp(θmean + θamp * sin(2π * (t - 170.0) / 365), 0.05, 0.9)

# Soil/air parameters
BD, PD = 0.8, 2.52
porosity = 1 - BD/PD
a = t -> max(porosity - θ(t), 1e-6)
Dliq, Dgas = 3.17, 1.67
Dliq_t = t -> Dliq * θ(t)^3
Dgas_t = t -> Dgas * a(t)^(4/3)
O2airfraction = 0.209

# DOC fraction
psx = 4.14e-4

# Universal constants
Rgas = 8.314
Tref = 293.15

# --------------------------------------------------------
# Oxic respiration parameters (CO₂)
# --------------------------------------------------------

αS_ref = 1.0       # g C m^-3 d^-1
Ea     = 72000.0   # J mol^-1
ckM    = 5.0*0.012 # g C m^-3
mkM    = 0.1*0.012 # g C m^-3 °C^-1
kM_O2  = 0.121     # dimensionless

# --------------------------------------------------------
# CH₄ production parameters
# --------------------------------------------------------

αCH4_prod = 0.5
Ea_CH4_prod = 72000.0
kM_C_CH4 = 0.1
kI_O2_CH4 = 0.05   # O₂ inhibition

# CH₄ oxidation
αCH4_ox = 0.8
Ea_CH4_ox = 65000.0
kM_CH4 = 0.05
kM_O2_CH4 = 0.05

# --------------------------------------------------------
# Initial states
# --------------------------------------------------------

C0    = 500.0    # g C m^-2 SOC
CH4_0 = 0.01     # g C m^-3 CH₄
CO2_0 = 0.0      # g C m^-2 cumulative CO₂
CH4_cum0 = 0.0   # g C m^-2 cumulative CH₄

u0 = [C0, CH4_0, CO2_0, CH4_cum0]
tspan = (0.0, 365.0)

# --------------------------------------------------------
# Linked ODE system
# --------------------------------------------------------

function soc_redox!(du, u, p_, t)
    SOC  = u[1]    # g C m^-2
    CH4  = u[2]    # g C m^-3
    FCO2 = u[3]    # g C m^-2 cumulative CO₂
    FCH4 = u[4]    # g C m^-2 cumulative CH₄

    # Substrate (DOC pool proxy)
    Ssol = psx * SOC
    Sx = Ssol * Dliq_t(t) / depth  # g C m^-3

    # O₂ availability
    O2 = Dgas_t(t) * O2airfraction

    # Temperature
    TK = T_C(t) + 273.15

    # --- Oxic respiration ---
    Vmax_CO2 = αS_ref * exp(-Ea/Rgas * (1/TK - 1/Tref))
    kM_Sx = ckM + mkM * T_C(t)
    Rco2 = Vmax_CO2 * (Sx / (kM_Sx + Sx)) * (O2 / (kM_O2 + O2))  # g C m^-3 d^-1

    # --- Methanogenesis ---
    Vmax_prod = αCH4_prod * exp(-Ea_CH4_prod/Rgas * (1/TK - 1/Tref))
    Rprod = Vmax_prod * (Sx / (kM_C_CH4 + Sx)) * (1 / (1 + O2/kI_O2_CH4))  # g C m^-3 d^-1

    # --- Methane oxidation ---
    Vmax_ox = αCH4_ox * exp(-Ea_CH4_ox/Rgas * (1/TK - 1/Tref))
    Rox = Vmax_ox * (CH4 / (kM_CH4 + CH4)) * (O2 / (kM_O2_CH4 + O2))  # g C m^-3 d^-1

    # --- State updates ---
    du[1] = -(Rco2*depth + Rprod*depth)    # SOC loss to respiration & methanogenesis
    du[2] = Rprod - Rox                    # CH₄ conc (g C m^-3)
    du[3] = Rco2*depth                     # cumulative CO₂ flux (g C m^-2)
    du[4] = (Rprod - Rox)*depth            # cumulative CH₄ flux (g C m^-2)
end

# --------------------------------------------------------
# Solve
# --------------------------------------------------------

prob = ODEProblem(soc_redox!, u0, tspan)
sol = solve(prob, Tsit5(); saveat=1.0)

t     = sol.t
SOC   = sol[1, :]
CH4   = sol[2, :]
CO2_cum = sol[3, :]
CH4_cum = sol[4, :]

CO2_flux = [CO2_cum[1]; diff(CO2_cum)]
CH4_flux = [CH4_cum[1]; diff(CH4_cum)]

# --------------------------------------------------------
# Plots
# --------------------------------------------------------

p1 = plot(t, SOC, lw=2, xlabel="Day", ylabel="SOC (g C m⁻²)", label="SOC", title="SOC stock")
p2 = plot(t, CO2_flux, lw=2, xlabel="Day", ylabel="CO₂ flux (g C m⁻² d⁻¹)", label="CO₂ flux")
plot!(p2, t, CO2_cum, lw=2, ls=:dash, label="Cumulative CO₂")

p3 = plot(t, CH4_flux, lw=2, xlabel="Day", ylabel="CH₄ flux (g C m⁻² d⁻¹)", label="CH₄ flux")
plot!(p3, t, CH4_cum, lw=2, ls=:dash, label="Cumulative CH₄")
