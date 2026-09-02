# # Building Models Examples
#
# `EasyHybrid.jl` allows constructing diverse modeling architectures using the unified `HybridModel` struct.
# Previously, users defined bespoke structs (like `LinearHM`, `RespirationRbQ10`) for different configurations.
# Here we demonstrate how those legacy model architectures can be trivially constructed via `HybridModel`.
#
# ## Setup
# First, let's load our required packages:
using EasyHybrid

# ## 1. Linear Hybrid Model
# This is a basic model with one neural network predicting a coefficient `α`, and an explicit global parameter `β`.
# The equation is: `ŷ = α * x + β`
#
# ### Process-Based Definition
linear_mechanistic(; x, α, β) = (; obs = α .* x .+ β)

# ### Parameter Setup
params_linear = (
    α = (1.0f0, 0.0f0, 2.0f0),
    β = (1.5f0, -1.0f0, 3.0f0),
)

# ### HybridModel Construction
# We use `x` as forcing data, predict `α` with a neural network based on some predictors `a` and `b`,
# and leave `β` as a globally optimized constant parameter.
lhm = constructHybridModel(
    [:a, :b],          # predictors for the NN (predicts α)
    [:x],              # forcing variable
    [:obs],            # targets
    linear_mechanistic, # our mechanistic model
    params_linear,     # parameter container
    [:α],              # parameters predicted by the NN
    [:β];              # globally optimized constant parameters
    hidden_layers = [4, 4],
    activation = tanh
)


# ## 2. Respiration Rb Q10
# A single NN predicting `Rb` for a Q10 temperature-sensitive respiration formulation.
# The equation is: `R_soil = Rb * Q10^(0.1 * (Temp - 15))`
#
# ### Process-Based Definition
function mRbQ10(; Temp, Rb, Q10)
    R_soil = @. Rb * Q10^(0.1f0 * (Temp - 15.0f0))
    return (; R_soil)
end

# ### Parameter Setup
params_rbq10 = (
    Rb = (1.0f0, 0.0f0, 5.0f0),
    Q10 = (1.5f0, 1.0f0, 3.0f0),
)

# ### HybridModel Construction
m_rbq10 = constructHybridModel(
    [:SWC, :TA], # predictors for Rb
    [:Temp],     # forcing variable
    [:R_soil],   # targets
    mRbQ10,      # mechanistic model
    params_rbq10,
    [:Rb],       # predicted by NN
    [:Q10];      # globally optimized
    hidden_layers = [8, 8]
)


# ## 3. Respiration Components
# A single NN outputting 3 distinct parameters (`Rb_het`, `Rb_root`, `Rb_myc`).
#
# ### Process-Based Definition
function rs_comp(; Temp, Rb_het, Rb_root, Rb_myc, Q10_het, Q10_root, Q10_myc)
    R_het = @. Rb_het * Q10_het^(0.1f0 * (Temp - 15.0f0))
    R_root = @. Rb_root * Q10_root^(0.1f0 * (Temp - 15.0f0))
    R_myc = @. Rb_myc * Q10_myc^(0.1f0 * (Temp - 15.0f0))
    R_soil = R_het .+ R_root .+ R_myc
    return (; R_soil, R_het, R_root, R_myc)
end

# ### Parameter Setup
params_rs_comp = (
    Rb_het = (1.0f0, 0.0f0, 5.0f0),
    Rb_root = (1.0f0, 0.0f0, 5.0f0),
    Rb_myc = (1.0f0, 0.0f0, 5.0f0),
    Q10_het = (1.5f0, 1.0f0, 3.0f0),
    Q10_root = (1.5f0, 1.0f0, 3.0f0),
    Q10_myc = (1.5f0, 1.0f0, 3.0f0),
)

# ### HybridModel Construction
m_rs_comp = constructHybridModel(
    [:SWC, :TA],  # predictors for all 3 Rb parameters
    [:Temp],
    [:R_soil],
    rs_comp,
    params_rs_comp,
    [:Rb_het, :Rb_root, :Rb_myc],
    [:Q10_het, :Q10_root, :Q10_myc];
    hidden_layers = [16, 16]
)


# ## 4. Flux Partitioning with Multiple NNs
# A multi-NN architecture predicting `RUE` (Radiation Use Efficiency) and `Rb` from different sets of predictors.
#
# ### Process-Based Definition
function flux_part(; SW_IN, TA, RUE, Rb, Q10)
    GPP = @. SW_IN * RUE / 12.011f0
    RECO = @. Rb * Q10^(0.1f0 * (TA - 15.0f0))
    NEE = RECO .- GPP
    return (; NEE, GPP, RECO)
end

# ### Parameter Setup
params_flux = (
    RUE = (1.0f0, 0.0f0, 5.0f0),
    Rb = (1.0f0, 0.0f0, 5.0f0),
    Q10 = (1.5f0, 1.0f0, 3.0f0),
)

# ### HybridModel Construction
# By passing a `NamedTuple` to `predictors`, `HybridModel` automatically provisions
# an independent Neural Network for each key.
predictors_multi = (
    RUE = [:SWC, :TA, :SW_IN],
    Rb = [:SWC, :TA],
)

m_flux = constructHybridModel(
    predictors_multi, # Triggers Multi-NN construction
    [:SW_IN, :TA],    # Forcing variables
    [:NEE],           # Targets
    flux_part,        # Mechanistic model
    params_flux,
    [:Q10];           # Global parameter
    hidden_layers = (RUE = [8, 8], Rb = [4, 4]), # Custom architectures per NN
    activation = (RUE = Lux.sigmoid, Rb = tanh)
)


# ## 5. Process-Based Model (Zero NNs)
# A purely process-based configuration where all parameters are optimized globally, and no Neural Networks are built.
#
# ### Process-Based Definition
function mRbQ10_0(; Temp, Rb, Q10)
    R_soil = @. Rb * Q10^(0.1f0 * (Temp - 0.0f0))
    return (; R_soil)
end

# ### HybridModel Construction
# Passing an empty `Symbol[]` array to `predictors` prevents any Neural Networks from being created.
m_pbm = constructHybridModel(
    Symbol[],      # No predictors -> No Neural Network
    [:Temp],       # Forcing
    [:R_soil],     # Target
    mRbQ10_0,
    params_rbq10,
    Symbol[],      # No neural params
    [:Rb, :Q10]    # Both are optimized as global parameters
)

# ## 6. Per-Parameter Scaling (`:linear`, `:log`, `:logit`)
# Every optimizable parameter is mapped from an unconstrained value into its
# bounds `[lower, upper]` via a monotone *warp*. By default this warp is
# `:linear` (uniform resolution in the value). You can select a different warp
# per parameter by appending a 4th element to its tuple:
#
# * `:linear` — default; good for narrow, well-behaved ranges.
# * `:log` — uniform resolution in `log(value)`; ideal for strictly-positive
#   quantities spanning several orders of magnitude (rates, turnover times,
#   observation-noise scales). Requires `lower > 0`.
# * `:logit` — uniform resolution in the log-odds; ideal for fractions that can
#   approach 0 and/or 1. Requires `0 < lower < upper < 1`.
#
# ### Process-Based Definition
# A minimal decomposition model: an NN predicts the carbon-use efficiency `CUE`
# (a fraction), while a basal rate `k` and observation-noise scale `σ` (both
# spanning orders of magnitude) are optimized globally.
decomp(; Corg, k, CUE, σ = nothing) = (; flux = k .* Corg .* (1.0f0 .- CUE))

# ### Parameter Setup
# Note the optional 4th tuple element selecting the warp. `CUE` keeps the
# default `:linear` (already logit-space for the optimizer over an interior
# range), `k` and `σ` use `:log`.
params_scaled = (
    k = (0.01f0, 1.0f-4, 1.0f0, :log),    # rate over ~4 orders of magnitude
    CUE = (0.5f0, 0.05f0, 0.65f0),          # interior fraction -> :linear
    σ = (1.0f0, 0.01f0, 100.0f0, :log),   # obs-noise scale, stays > 0
)

# ### HybridModel Construction
m_scaled = constructHybridModel(
    [:SWC, :TA],   # predictors for CUE
    [:Corg],       # forcing variable
    [:flux],       # target
    decomp,
    params_scaled,
    [:CUE],        # predicted by NN
    [:k, :σ];      # globally optimized (with :log warp)
    hidden_layers = [8, 8],
    scale_nn_outputs = true,
)

# The chosen warp is recorded per parameter and used for both initialization and
# the forward pass; nothing else in your training code needs to change.
m_scaled.parameters.scales

# ## Summary
#
# As demonstrated above, `HybridModel` provides a highly flexible, unified interface.
# By simply modifying the `predictors` argument and your mechanistic function, you can rapidly scale from a purely
# process-based model, to a single Neural Network hybrid model, all the way up to complex multi-Neural Network architectures!
# And with the optional per-parameter warp (`:linear`, `:log`, `:logit`), each parameter is optimized on the scale
# that best matches its physical range.
