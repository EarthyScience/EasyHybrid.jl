
export RespirationRbQ10
export mRbQ10

"""
    RespirationRbQ10(NN, predictors, forcing, targets, Q10)

A linear hybrid model with a neural network `NN`, `predictors`, `targets` and `forcing` terms.
"""
struct RespirationRbQ10{D, T1, T2, T3, T4} <: LuxCore.AbstractLuxContainerLayer{(:NN, :predictors, :forcing, :targets, :Q10)}
    NN
    predictors
    forcing #TODO order is messed up compared to new
    targets
    Q10
    function RespirationRbQ10(NN::D, predictors::T1, forcing::T2, targets::T3, Q10::T4) where {D, T1, T2, T3, T4}
        new{D, T1, T2, T3, T4}(NN, collect(predictors), collect(targets), collect(forcing), [Q10])
    end
end

# ? Q10 is a parameter, so expand the initialparameters!
function LuxCore.initialparameters(::AbstractRNG, layer::RespirationRbQ10)
    ps, _ = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; ps, Q10 = layer.Q10,)
end
# TODO: trainable vs non-trainable! set example!
# see: https://lux.csail.mit.edu/stable/manual/migrate_from_flux#Implementing-Custom-Layers
function LuxCore.initialstates(::AbstractRNG, layer::RespirationRbQ10)
    _, st = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; st)
end

"""
    mRbQ10(Rb, Q10, Temp, Tref)

    Rb base respiration, Q10 temperature sensitivity, Temp current temperature, Tref reference temperature
    
"""

function mRbQ10(Rb, Q10, Temp, Tref)
    @. Rb * Q10 ^(0.1f0 * (Temp - Tref))
end

"""
    RespirationRbQ10(NN, predictors, forcing, targets, Q10)(ds_k)

# Model definition `ŷ = Rb(αᵢ(t)) * Q10^((T(t) - T_ref)/10)`

ŷ (respiration rate) is computed as a function of the neural network output `Rb(αᵢ(t))` and the temperature `T(t)` adjusted by the reference temperature `T_ref` (default 15°C) using the Q10 temperature sensitivity factor.
````
"""
function (hm::RespirationRbQ10)(ds_k, ps, st::NamedTuple)
    p = ds_k(hm.predictors)
    x = Array(ds_k(hm.forcing)) # don't propagate names after this
    
    Rb, stQ10 = LuxCore.apply(hm.NN, p, ps.ps, st.st) #! NN(αᵢ(t)) ≡ Rb(T(t), M(t))

    #TODO output name flexible - could be R_soil, heterotrophic, autotrophic, etc.
    R_soil = mRbQ10(Rb, ps.Q10, x, 15.0f0) # ? should 15°C be the reference temperature also an input variable?

    return (; R_soil, Rb), (; st = (; st = stQ10))
end

