# --------------------------------------------------------
# DAMM CH4 module (Methanogenesis + Oxidation)
# Based on Sihi et al. 2020
# --------------------------------------------------------

using DifferentialEquations
using Plots

# --------------------------------------------------------
# Environmental drivers
# --------------------------------------------------------

Tmean, Tamp = 8.0, 10.0                     # °C mean & amplitude
T_C = t -> Tmean + Tamp * sin(2π * (t - 200.0) / 365)   # seasonal soil T
Tref = 293.15                               # K reference temperature

θmean, θamp = 0.55, 0.10
θ = t -> clamp(θmean + θamp * sin(2π * (t - 170.0) / 365), 0.05, 0.9)

# Soil properties   
BD = 0.8                              # bulk density (g cm^-3)    
PD = 2.52                             # particle density (g cm^-3)   
porosity = 1 - BD/PD
a = t -> max(porosity - θ(t), 1e-6)   # air-filled porosity: max and exp just avoids zero or negative values given by large θ(t)
depth = 0.1                           # m (active layer depth, e.g., 10 cm). This should be water table depth. 

# Diffusion parameters (does it hold for CH4?)  
Dliq    = 3.17                     # Dimensionless. Diffusion coefficient of substrate in liquid phase (Papendick & Campbell, 1981)
Dliq_t  = t -> Dliq * θ(t)^3       # Dliq changes with θ, cubic relationship 
Dgas    = 1.67                     # Dimensionless. Diffusion coefficient of O2 in air  (Davidson 2012)
Dgas_t  = t -> Dgas * a(t)^(4/3)   # Dgas changes with θ, sharp drop as saturation approaches. Too dry/wet = little diffusion

# Constants
psx     = 4.14e-4                  # "p" in Davidson 2012 (from Harvard forest) this is site-dependent -> proportion of C that is DOC 
O2airfraction = 0.209              # Dimensionless. O2 fraction in air 
Rgas = 8.314                       # Universal gas constant (J mol^-1 K^-1)

# --------------------------------------------------------
# CH4 production parameters
# --------------------------------------------------------

αCH4_prod = 0.5         # g C m^-3 d^-1   (Vmax_ref, per volume basis)
Ea_CH4_prod = 72000.0   # J mol^-1        (activation energy for methanogenesis)
kM_C_CH4 = 0.1          # g C m^-3        (DOC half-saturation constant)
kI_O2_CH4 = 0.05        # dimensionless   (O2 inhibition constant, relative O2)

# --------------------------------------------------------
# CH4 oxidation parameters
# --------------------------------------------------------

αCH4_ox = 0.8           # g C m^-3 d^-1   (Vmax_ref for methanotrophy, per volume)
Ea_CH4_ox = 65000.0     # J mol^-1        (activation energy for oxidation)
kM_CH4 = 0.05           # g C m^-3        (CH4 half-saturation constant)
kM_O2_CH4 = 0.05        # dimensionless   (O2 half-saturation for oxidation)

# --------------------------------------------------------
# State variables & initial conditions
# --------------------------------------------------------

C0    = 500.0     # g C m^-2 SOC
CH4_0 = 0.01      # g C m^-3 CH₄ conc
F0    = 0.0       # g C m^-2 cumulative CH₄ flux
u0 = [C0, CH4_0, F0]  # [SOC, CH₄, cumFlux]

tspan = (0.0, 365.0)

# --------------------------------------------------------
# CH4 module (methanogenesis + oxidation)
# ODE definition
# --------------------------------------------------------

function ch4_module!(du, u, p, t)
    SOC  = u[1]      # g C m^-2
    CH4  = u[2]      # g C m^-3
    Fcum = u[3]      # g C m^-2

    # Substrate availability
    Ssol = psx * SOC                 # g C m^-2
    Sx   = Ssol * Dliq_t(t) / depth  # g C m^-3

    # O₂ availability (relative, dimensionless)
    O2 = Dgas_t(t) * O2airfraction

    # Temperature (K)
    TK = T_C(t) + 273.15

    # Arrhenius scalings
    Vmax_prod = αCH4_prod * exp(-Ea_CH4_prod/Rgas * (1/TK - 1/Tref))
    Vmax_ox   = αCH4_ox   * exp(-Ea_CH4_ox  /Rgas * (1/TK - 1/Tref))

    # Rates
    Rprod = Vmax_prod * (Sx / (kM_C_CH4 + Sx)) * (1 / (1 + O2/kI_O2_CH4))   # g C m^-3 d^-1
    Rox   = Vmax_ox   * (CH4 / (kM_CH4 + CH4)) * (O2 / (kM_O2_CH4 + O2))    # g C m^-3 d^-1

    # State derivatives
    du[1] = 0.0                        # SOC not yet linked
    du[2] = Rprod - Rox                # CH₄ conc dynamics (g C m^-3)
    du[3] = (Rprod - Rox) * depth      # cumulative flux (g C m^-2)
end

# --------------------------------------------------------
# Solve ODE
# --------------------------------------------------------

prob = ODEProblem(ch4_module!, u0, tspan)
sol = solve(prob, Tsit5(); saveat=1.0)

t   = sol.t
SOC = sol[1, :]
CH4 = sol[2, :]
CH4_flux = [sol[3,1]; diff(sol[3,:])]   # instantaneous flux g C m^-2 d^-1
CH4_cum  = sol[3, :]

# --------------------------------------------------------
# Plots
# --------------------------------------------------------

p1 = plot(t, CH4, lw=2, xlabel="Day", ylabel="CH₄ (g C m⁻³)", label="CH₄ conc", title="CH₄ dynamics")
p2 = plot(t, CH4_flux, lw=2, xlabel="Day", ylabel="CH₄ flux (g C m⁻² d⁻¹)", label="Instantaneous flux", title="Methane flux")
plot!(p2, t, CH4_cum, lw=2, ls=:dash, label="Cumulative CH₄")