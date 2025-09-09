# V2 DAMM (Davidson 2012) but improved  by including pools from Abramoff 2018) - Millenial Pools version of CO2 model (tailored for peatlands
# Millenial was not tested: we will through easyhybrid
# Agreggate C pool: not included since the structure in peatlands is dominated by fibrous material
# MAOM pool: set to 0   
# LMWC ≈ DOC 

#import Pkg; Pkg.add("DifferentialEquations")
#import Pkg; Pkg.add("Plots")

using DifferentialEquations
using Plots
#gr() # Backend for using Plots

# --------------------------------------------------------         
# Parameters (as mean or seasonal values)
# --------------------------------------------------------   

# 1 µmol C L⁻¹ = 0.012 g C m⁻³

# Productivity (NPP) g C m^-2 d^-1 (Input to soil)
NPP = 1.5   # constant value
NPP_t = t -> NPP

# If seasonal NPP 
amp     = 0.8                    # amplitude around mean NPP 
phase   = 120.0                  # day of year with peak NPP
NPP_t   = t -> max(0.0, NPP + amp * sin(2π * (t - phase) / 365.0))

# Temperature (T) °C (it should be Soil temp)
Tmean, Tamp = 8.0, 10.0        # °C mean & amplitude
T_C = t -> Tmean + Tamp * sin(2π * (t - 200.0) / 365)

# Volumetric water content (θ) m3/m3 
θmean, θamp = 0.55, 0.10      # peat is wet; keep ≤ porosity
θ = t -> clamp(θmean + θamp * sin(2π * (t - 170.0) / 365), 0.05, 0.9)  # constrain between 0.05 and 0.9 since here is always film water in dry conditions or air in wet conditions. Also extremes avoided prevent numerical stability. Achtung! measure in the field. 

# Soil properties
BD = 0.8                    # bulk density (g cm^-3)
PD = 2.52                   # particle density (g cm^-3)
porosity = 1 - BD/PD
a = t -> max(porosity - θ(t), 1e-6)   # air-filled porosity: max and exp just avoids zero or negative values given by large θ(t)
depth = 0.1                 # m (active layer depth, e.g., 10 cm). This should be water table depth.

# Diffusion parameters
Dliq    = 3.17                     # Dimensionless. Diffusion coefficient of substrate in liquid phase (Papendick & Campbell, 1981)
Dliq_t  = t -> Dliq * θ(t)^3       # Dliq changes with θ, cubic relationship 
Dgas    = 1.67                     # Dimensionless. Diffusion coefficient of O2 in air  (Davidson 2012)
Dgas_t  = t -> Dgas * a(t)^(4/3)   # Dgas changes with θ, sharp drop as saturation approaches. Too dry/wet = little diffusion

# psx     = 4.14e-4                  # "p" in Davidson 2012 (from Harvard forest) this is site-dependent -> proportion of C that is DOC. Now replaced by pool LMWC 
O2airfraction = 0.209              # Dimensionless. O2 fraction in air 
Rgas = 8.314                       # Universal gas constant (J mol^-1 K^-1)

# Temperature dependence (Arrhenius with reference) 
αS_ref = 1.5                         # g C m^-2 d^-1 (pre-exponential Vmax factor) at T_ref (20C). This is a placeholder substrate-specific. It should be set as "measured max CO₂ release rates at high temperature and abundant O₂." "should be calibrated" 
MB_ref = 10.0                      # g C m^-2 (reference microbial biomass; here set equal to MB0)
# Convert αS_ref into a per-biomass specific uptake rate (d^-1). Hence dropped because now its dependant on MB 
Vmax_ref_spec = αS_ref / MB_ref    # d^-1. Reference uptake rate per g C MB at Tref.

Ea   = 72000                       # J mol^-1 (activation energy of the enzymatic reaction).  "typical value for enzymatic decomposition of labile organic matter " "should be calibrated" . 
Tref = 293.15                      # K

ckM = 5.0 * 0.012                  # micromol L^-1 * 0.012  = g C m^-3 . Intercept, the baseline when T=0. Used for substrate T dependence        
mkM = 0.1 * 0.012                  # micromol L^-1 Celsius ^-1 * 0.012 = g C m^-3. Slope, rate of change. Used for substrate T dependence  
kM_O2 = 0.121                      # Dimensionless because its a fractional limitation. Michaelis constant for O2. No Arrhernius function for O2, because it is not T dependent

# Decomposition parameters for pools
k_PL = 0.05                 # d⁻¹ - First-order decay rate from POM to LMWC.
k_MB = 0.01                 # d⁻¹ - Microbial turnover rate constant. Fraction of the MB that dies and is recycled back into the LMWC 
CUE = 0.4                   # Dimensionless. Carbon use efficiency. Microbial growth efficiency (ratio of carbon assimilated to carbon taken up).

# Numerical safeguard for division by zero
eps = 1e-12

# --------------------------------------------------------   
# Initial states and time definition
# -------------------------------------------------------- 

POM0      = 400.0   # g C m^-2, Particulate OM (POM)
LMWC0     = 50.0    # g C m^-2, LMWC (DOC pool) - Low molecular weight carbon
MB0     = 10.0    # g C m^-2, microbial biomass
MAOM0   = 0.0     # g C m^-2, optional mineral-associated OM (set to zero in peatlands)
R0      = 0.0     # g C m^-2, cumulative CO₂ flux

u0 = [POM0, LMWC0, MB0, MAOM0, R0]
tspan = (0.0, 365.0)      # days (simulation period)

function soc_oxic_damm!(du, u, p_, t)
    POM = u[1]                      # Particulate Organic Matter (POM) stock g C m^-2
    LMWC = u[2]                      # Low Molecular Weight Carbon (LMWC) stock g C m^-2
    MB = u[3]                     # Microbial Biomass stock g C m^-2
    MAOM = u[4]                   # Mineral-Associated Organic Matter stock g C m^-2
    R = u[5]                      # Cumulative CO2 flux g C m^-2

# --- Environmental drivers ---
TK = T_C(t) + 273.15                        # Soil temperature (K)
temp_factor = exp(-Ea / Rgas * (1/TK - 1/Tref))   # Arrhenius scaling factor

# Diffusion-limited substrate and oxygen availability
Sx = (LMWC / depth) * Dliq_t(t)          # [Sx] g C m-3 (soluble fraction of SOC) substrate concentration over time
O2 = Dgas_t(t) * O2airfraction              # Dimensionless effective O2 at microbial sites.

# Michaelis constants
kM_Sx = max(ckM + mkM * T_C(t), eps)        # g C m^-3. Michaelis constant for substrate with linear T dependence. Max and eps are numerical safeguards
lim_Sx = Sx / (kM_Sx + Sx + eps)            # dimensionless Substrate limitation (0–1)
lim_O  = O2 / (kM_O2 + O2 + eps)            # dimensionless O2 limitation (0–1)

# --- Temperature-corrected rate constants ---
Vmax = Vmax_ref_spec * temp_factor          # d^-1, per-biomass uptake capacity. Arrhenius function for temperature dependence of reaction rates. MB scaled. 
k_PL_T = k_PL * temp_factor             # d^-1, POM → LMWC decomposition
k_MB_T = k_MB * temp_factor             # d^-1, microbial turnover

# --- Fluxes ---
    # 1. Microbial uptake of LMWC → MB growth + respiration
    F_LWMC_MB = Vmax * MB * lim_Sx * lim_O          # g C m^-2 d^-1. It depends on MB because without microbes, no uptake happens
    Growth_MB = CUE * F_LWMC_MB                     # g C m^-2 d^-1
    Resp_Uptake = (1 - CUE) * F_LWMC_MB             # g C m^-2 d^-1

    # 2. POM → LMWC (enzyme-mediated, microbial dependent)
    # Equation: F_POM_LMWC = k_PL_T * (POM/(POM+ε)) * (MB/(MB+ε)) * θ(t)
    F_POM_LMWC = k_PL_T * (POM / (POM + eps)) * (MB / (MB + eps)) * θ(t)

    # 3. Microbial turnover → LMWC
    # Equation: F_MB_LMWC = k_MB_T * MB
    F_MB_LMWC = k_MB_T * MB

    # 4. Optional processes (disabled for peat at this stage)
    Fl_leach = 0.0        # leaching loss from LMWC
    Fl_adsorb = 0.0       # adsorption into MAOM (placeholder)

    # --- Cumulative respiration flux ---
    Rco2_total = Resp_Uptake    # g C m^-2 d^-1. (Add maintenance respiration here if included later.)

    # --- Differential equations ---
    du[1] = NPP_t(t) - F_POM_LMWC                        # ΔPOM: NPP input - decomposition
    du[2] = F_POM_LMWC + F_MB_LMWC - F_LWMC_MB - Fl_leach - Fl_adsorb  # ΔLMWC
    du[3] = Growth_MB - F_MB_LMWC                        # ΔMB
    du[4] = 0.0                                          # ΔMAOM (inactive here)
    du[5] = Rco2_total                                   # ΔR (cumulative respiration)
end

# --------------------------------------------------------
# Solve ODE
# --------------------------------------------------------

prob = ODEProblem(soc_oxic_damm!, u0, tspan)
sol = solve(prob, Tsit5(); saveat=1.0)

t      = sol.t
POM    = sol[1, :]
LMWC   = sol[2, :]
MB     = sol[3, :]
MAOM   = sol[4, :]
Rco2   = sol[5, :]

# Instantaneous flux: from cumulative R, daily steps
flux   = [Rco2[1]; diff(Rco2)]          # g C m^-2 d^-1

# Calculate total SOC stock
SOC = POM + LMWC + MB + MAOM

# --------------------------------------------------------
# Plots
# --------------------------------------------------------

# Panel 1: SOC stock
p1 = plot(t, SOC, lw=2, xlabel="Day", ylabel="Total SOC (g C m⁻²)", label="Total SOC", title="SOC and pools over time")
plot!(p1, t, POM, lw=2, ls=:dash, label="POM Pool")
plot!(p1, t, LMWC, lw=2, ls=:dot, label="LMWC Pool")
plot!(p1, t, MB, lw=2, ls=:dashdot, label="MB Pool")

# Panel 2: CO₂ flux vs cumulative
p2 = plot(t, flux, lw=2, xlabel="Day", ylabel="CO₂ flux (g C m⁻² d⁻¹)", label="Instantaneous flux", title="Soil respiration")
plot!(p2, t, Rco2, lw=2, ls=:dash, label="Cumulative CO₂")

# Panel 3: Drivers
p3 = plot(t, T_C.(t), lw=2, xlabel="Day", ylabel="°C", label="Temperature")
p4 = plot(t, θ.(t), lw=2, xlabel="Day", ylabel="θ (m³ m⁻³)", label="Soil moisture")
p5 = plot(t, NPP_t.(t), lw=2, xlabel="Day", ylabel="g C m⁻² d⁻¹", label="NPP")

plot(p1, p2, p3, p4, p5, layout=(3,2))