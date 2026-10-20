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
# We use `x` as forcing data, predict `α` with a neural network based on predictors `a` and `b`,
# and leave `β` as a globally optimized constant parameter.
#
# We can construct this model either with the declarative `@hybrid` macro:
lhm = @hybrid begin
    mechanistic = linear_mechanistic
    targets = :obs
    forcing = :x
    neural = [:a, :b] => :α
    parameters = params_linear
    hidden_layers = [4, 4]
    activation = tanh
end

# Or with the functional `constructHybridModel` following the physics-first argument order:
lhm = constructHybridModel(
    linear_mechanistic, # mechanistic model
    :obs,               # targets
    :x,                 # forcing variable
    [:a, :b] => :α,     # neural mapping (predictors => predicted parameter)
    params_linear;      # parameter container (global parameter β is auto-inferred)
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
m_rbq10 = @hybrid begin
    mechanistic = mRbQ10
    targets = :R_soil
    forcing = [:Temp]
    neural = [:SWC, :TA] => :Rb
    parameters = params_rbq10
    hidden_layers = [8, 8]
end


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
m_rs_comp = @hybrid begin
    mechanistic = rs_comp
    targets = :R_soil
    forcing = [:Temp]
    neural = [:SWC, :TA] => [:Rb_het, :Rb_root, :Rb_myc]
    parameters = params_rs_comp
    hidden_layers = [16, 16]
end


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
# By passing a `NamedTuple` to `neural`/`predictors`, `HybridModel` automatically provisions
# an independent Neural Network for each key.
m_flux = @hybrid begin
    mechanistic = flux_part
    targets = [:NEE]
    forcing = [:SW_IN, :TA]
    neural = (
        RUE = [:SWC, :TA, :SW_IN],
        Rb = [:SWC, :TA],
    )
    parameters = params_flux
    hidden_layers = (RUE = [8, 8], Rb = [4, 4])
    activation = (RUE = Lux.sigmoid, Rb = tanh)
end


# ## 5. Process-Based Model (Zero NNs)
# A purely process-based configuration where all parameters are optimized globally, and no Neural Networks are built.
#
# ### Process-Based Definition
function mRbQ10_0(; Temp, Rb, Q10)
    R_soil = @. Rb * Q10^(0.1f0 * (Temp - 0.0f0))
    return (; R_soil)
end

# ### HybridModel Construction
# Setting `neural = nothing` (or passing `nothing` in `constructHybridModel`) prevents any Neural Networks from being created.
m_pbm = @hybrid begin
    mechanistic = mRbQ10_0
    targets = [:R_soil]
    forcing = [:Temp]
    neural = nothing
    parameters = params_rbq10
end

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
m_scaled = @hybrid begin
    mechanistic = decomp
    targets = :flux
    forcing = [:Corg]
    neural = [:SWC, :TA] => :CUE
    parameters = params_scaled
    hidden_layers = [8, 8]
    scale_nn_outputs = true
end

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
