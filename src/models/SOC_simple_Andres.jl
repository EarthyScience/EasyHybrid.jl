
# Simple mechanistic SOC model for an oxic peat layer over 365 days.
# dC/dt = NPP - k_oxic * C

#import Pkg; Pkg.add("DifferentialEquations"); using DifferentialEquations
#import Pkg; Pkg.add("Plots"); using Plots

using DifferentialEquations
using Plots

# --------------------------------------------------------         
# Parameters (as mean or seasonal values)
# --------------------------------------------------------   

# Productivity (NPP) g C m^-2 d^-1 (Input to soil)
NPP      = 1.5  
# Using constant NPP:
# NPP_t = t -> NPP    

# If seasonal NPP 
amp = 0.8                      # amplitude around mean NPP (g C m^-2 d^-1)
phase = 120.0                  # day of year with peak NPP
NPP_t = t -> max(0.0, NPP + amp * sin(2π * (t - phase) / 365.0))

# --------------------------------------------------------   
# Initial states and time definition
# -------------------------------------------------------- 

k_oxic   = 0.010            # d^-1 (decomposition rate)
C0       = 500.0            # g C m^-2 (initial SOC stock in active layer)
tspan    = (0.0, 365.0)     # days (simulation period)

# --------------------------------------------------------   
# Model definition
# --------------------------------------------------------   

function soc_oxic!(dC, C, p, t)
    # State variables
    SOC = C[1]                    # C[1] = SOC stock
    Rco2 = k_oxic * SOC           # Oxic CO2 release using first-order kinetics
   
    dC[1] = NPP_t(t) - Rco2       # change in SOC
    dC[2] = k_oxic * SOC          # CO₂ flux as a dynamic output
end

# Initial state: SOC stock and initial CO2 flux
R0 = k_oxic * C0
u0 = [C0, R0]

# --------------------------------------------------------   
# Solve ODE
# --------------------------------------------------------   

prob = ODEProblem(soc_oxic!, u0, tspan)
sol  = solve(prob, Tsit5(); saveat=1.0)    # Store results every day. # Why Tsit5 sovler method?
t   = sol.t                                # Extracts time vector
SOC  = sol[1, :]                           # Extracts SOC stock for all time points
Rco2 = sol[2, :]                           # Calculates CO2 flux (g C m^-2 d^-1)

# --------------------------------------------------------   
# Plots
# --------------------------------------------------------   
Plots.plot(t, SOC, label="SOC stock", lw=2, xlabel="Day", ylabel="SOC stock (g C m⁻²)")
Plots.plot(t, Rco2, label="CO₂ flux", lw=2, xlabel="Day", ylabel="g C m⁻² d⁻¹")
Plots.plot(t, NPP_t, label="NPP", lw=2, xlabel="Day", ylabel="g C m⁻² d⁻¹")
