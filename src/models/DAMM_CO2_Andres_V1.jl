# V1: DAMM Model for C accumulation and CO2 fluxes over time 

# TO improve compared to Davidson: electron acceptor limitation and its associated seasonality; substrate-enxyme complexes limitation as a function of 
# microbial activity and trophic interactions instead of only diffusivity or solubility (DOC factors), possibly evaluation of other C pools (mineral-associated).
# effects of wetting events could be included by optimizing CO2 fluxes over seasonal precipitation or SWC.   

# import Pkg; Pkg.add("DifferentialEquations")
# import Pkg; Pkg.add("Plots")

using DifferentialEquations
using Plots

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
θ = t -> clamp(θmean + θamp * sin(2π * (t - 170.0) / 365), 0.05, 0.9)

# Soil properties   
BD = 0.8                              # bulk density (g cm^-3)    
PD = 2.52                             # particle density (g cm^-3)   
porosity = 1 - BD/PD
a = t -> max(porosity - θ(t), 1e-6)   # air-filled porosity: max and exp just avoids zero or negative values given by large θ(t)
depth = 0.1                           # m (active layer depth, e.g., 10 cm). This should be water table depth. 
C = 500                               # g C m^-2 (initial SOC stock in active layer)

# Diffusion parameters
Dliq    = 3.17                     # Dimensionless. Diffusion coefficient of substrate in liquid phase (Papendick & Campbell, 1981)
Dliq_t  = t -> Dliq * θ(t)^3       # Dliq changes with θ, cubic relationship 
Dgas    = 1.67                     # Dimensionless. Diffusion coefficient of O2 in air  (Davidson 2012)
Dgas_t  = t -> Dgas * a(t)^(4/3)   # Dgas changes with θ, sharp drop as saturation approaches. Too dry/wet = little diffusion

psx     = 4.14e-4                  # "p" in Davidson 2012 (from Harvard forest) this is site-dependent -> proportion of C that is DOC 
O2airfraction = 0.209              # Dimensionless. O2 fraction in air 
Rgas = 8.314                       # Universal gas constant (J mol^-1 K^-1)

# Temperature dependence (Arrhenius with reference) 
αS_ref = 1                         # g C m^-2 d^-1???? (pre-exponential Vmax factor) at T_ref (20C). This is a placeholder substrate-specific. It should be set as "measured max CO₂ release rates at high temperature and abundant O₂." "should be calibrated" 
Ea   = 72000                       # J mol^-1 (activation energy of the enzymatic reaction).  "typical value for enzymatic decomposition of labile organic matter " "should be calibrated" . 
Tref = 293.15                      # K

ckM = 5.0 * 0.012                  # micromol L^-1 * 0.012  = g C m^-3 . Intercept, the baseline when T=0. Used for substrate T dependence        
mkM = 0.1 * 0.012                  # micromol L^-1 Celsius ^-1 * 0.012 = g C m^-3. Slope, rate of change. Used for substrate T dependence  
kM_O2 = 0.121                      # Here it is dimensionless because its a fractional limitation. Michaelis constant for O2. No Arrhernius function for O2, because it is not T dependent

# --------------------------------------------------------   
# Initial states and time definition
# -------------------------------------------------------- 

C0       = 500.0                # g C m^-2 (initial SOC stock in active layer)
R0       = 0.0                   # initial CO₂ flux
u0       = [C0, R0]
tspan    = (0.0, 365.0)         # days (simulation period)
# --------------------------------------------------------   
# Model definition
# --------------------------------------------------------   

function soc_oxic_damm!(du, u, p_, t)
    C = u[1]                                           # SOC stock g C m^-2        

    # Substrate and O2 at reaction site
    Ssol = psx * C                                     # [Ssol] g C m-2 (soluble fraction of SOC) substrate concentration over time
    Sx = Ssol * Dliq_t(t) / depth                      # [Sx] g C m-3 (soluble fraction of SOC) substrate concentration over time 
    O2 = Dgas_t(t) * O2airfraction                     # Dgas_t(t) includes already "a(t)^(4/3)"   # How much O2 diffuses and 4/3 is how much is at microbial sites

    # Temperature effects
    TK = T_C(t) + 273.15                               # Kelvin conversion
   # Vmax = αS_ref  * exp(-Ea / (Rgas * TK))           # g C m⁻² d⁻¹ Arrhenius function for temperature dependence of respiration/reaction rates as long as substrate is not limiting 
    Vmax = αS_ref  * exp(-Ea/Rgas * (1/TK - 1/Tref))   # 
    kM_Sx = ckM + mkM * T_C(t)                         # g C m^-3. Michaelis constant (microbial kinetics) for substrate with linear T dependence. Measured experimentally or estimated/calibrated. 

 # DAMM rate on both substrates
    Rco2 = Vmax * (Sx / (kM_Sx + Sx)) * (O2 / (kM_O2 + O2))
    du[1] = NPP_t(t) - Rco2                         # SOC balance
    du[2] = Rco2                                    # Cumulative CO2 flux output

end

# --------------------------------------------------------   
# Solve ODE
# --------------------------------------------------------   

prob = ODEProblem(soc_oxic_damm!, u0, tspan)
sol = solve(prob, Tsit5(); saveat=1.0)

t    = sol.t
SOC  = sol[1, :]
Rco2 = sol[2, :]
flux = [Rco2[1]; diff(Rco2)]          # g C m^-2 d^-1 when saveat=1.0

# --------------------------------------------------------   
# Plots
# --------------------------------------------------------   

# Panel 1: SOC stock
p1 = plot(t, SOC, lw=2, xlabel="Day", ylabel="SOC (g C m⁻²)", label="SOC", title="SOC stock over time")
# Panel 2: CO₂ flux vs cumulative
p2 = plot(t, flux, lw=2, xlabel="Day", ylabel="CO₂ flux (g C m⁻² d⁻¹)", label="Instantaneous flux", title="Soil respiration")
p6 = plot(t, Rco2, lw=2, xlabel="Day", ylabel="Cumulative CO₂ (g C m⁻²)", label="Cumulative CO₂", title="Cumulative CO₂ flux")  
plot!(p2, t, Rco2, lw=2, ls=:dash, label="Cumulative CO₂")


# Panel 3: Drivers
p3 = plot(t, T_C.(t), lw=2, xlabel="Day", ylabel="°C", label="Temperature")
p4 = plot(t, θ.(t), lw=2, xlabel="Day", ylabel="θ (m³ m⁻³)", label="Soil moisture")
p5 = plot(t, NPP_t.(t), lw=2, xlabel="Day", ylabel="g C m⁻² d⁻¹", label="NPP")


