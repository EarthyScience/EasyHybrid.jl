export RbQ10_2p

"""
    RbQ10_2p(forcing, targets, Q10)
"""
struct RbQ10_2p{T2, T3, T4} <: LuxCore.AbstractLuxContainerLayer{(:forcing, :targets, :Q10, :Rb)}
    forcing
    targets
    Q10
    Rb
    function RbQ10_2p(forcing::T2, targets::T3, Q10::T4, Rb::T4) where {T2, T3, T4}
        new{T2, T3, T4}(collect(targets), collect(forcing), [Q10], [Rb])
    end
end

# ? Q10 is a parameter, so expand the initialparameters!
function LuxCore.initialparameters(::AbstractRNG, layer::RbQ10_2p)
    #ps, _ = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; Q10 = layer.Q10, Rb = layer.Rb, )
end
# TODO: trainable vs non-trainable! set example!
# see: https://lux.csail.mit.edu/stable/manual/migrate_from_flux#Implementing-Custom-Layers
function LuxCore.initialstates(::AbstractRNG, layer::RbQ10_2p)
    #_, st = LuxCore.setup(Random.default_rng(), layer.NN)
    return NamedTuple()
end


"""
    RbQ10_2p(NN, predictors, forcing, targets, Q10)(ds_k)

# Model definition `ŷ = Rb(αᵢ(t)) * Q10^((T(t) - T_ref)/10)`

ŷ (respiration rate) is computed as a function of the neural network output `Rb(αᵢ(t))` and the temperature `T(t)` adjusted by the reference temperature `T_ref` (default 15°C) using the Q10 temperature sensitivity factor.
````
"""
function (hm::RbQ10_2p)(ds_k, ps, st)

    x = Array(ds_k(hm.forcing)) # don't propagate names after this

    R_soil = mRbQ10(ps.Rb, ps.Q10, x, 0.0f0) # ? should 15°C be the reference temperature also an input variable?

    return (; R_soil), (; st)
end

