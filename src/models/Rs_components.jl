
export Rs_components

"""
    Rs_components(NN, predictors, forcing, targets, Q10_het, Q10_root, Q10_myc)

A linear hybrid model with a neural network `NN`, `predictors`, `targets` and `forcing` terms.
"""
struct Rs_components{D, T1, T2, T3, T4} <: LuxCore.AbstractLuxContainerLayer{(:NN, :predictors, :forcing, :targets, :Q10_het, :Q10_root, :Q10_myc)}
    NN
    predictors
    forcing
    targets
    Q10_het
    Q10_root
    Q10_myc
    function Rs_components(NN::D, predictors::T1, forcing::T2, targets::T3, Q10_het::T4, Q10_root::T4, Q10_myc::T4) where {D, T1, T2, T3, T4}
        new{D, T1, T2, T3, T4}(NN, collect(predictors), collect(targets), collect(forcing), [Q10_het], [Q10_root], [Q10_myc])
    end
end

# ? Q10 is a parameter, so expand the initialparameters!
function LuxCore.initialparameters(::AbstractRNG, layer::Rs_components)
    ps, _ = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; ps, Q10_het = layer.Q10_het, Q10_root = layer.Q10_root, Q10_myc = layer.Q10_myc, )
end

function LuxCore.initialstates(::AbstractRNG, layer::Rs_components)
    _, st = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; st)
end

"""
    Rs_components(NN, predictors, forcing, targets, Q10)(ds_k)

# Model definition `ŷ = Rb(αᵢ(t)) * Q10^((T(t) - T_ref)/10)`

ŷ (respiration rate) is computed as a function of the neural network output `Rb(αᵢ(t))` and the temperature `T(t)` adjusted by the reference temperature `T_ref` (default 15°C) using the Q10 temperature sensitivity factor.
````
"""
function (hm::Rs_components)(ds_k, ps, st::NamedTuple)
    p = ds_k(hm.predictors)
    x = Array(ds_k(hm.forcing)) # don't propagate names after this
    
    out, st = LuxCore.apply(hm.NN, p, ps.ps, st.st)
    
    Rb_het = out[1,:]
    Rb_root = out[2,:]
    Rb_myc = out[3,:]

    R_het = mRbQ10(Rb_het, ps.Q10_het, x[1,:], 15.f0)	
    R_root = mRbQ10(Rb_root, ps.Q10_root, x[1,:], 15.f0)
    R_myc = mRbQ10(Rb_myc, ps.Q10_myc, x[1,:], 15.f0)

    R_soil = R_het .+ R_root .+ R_myc

    return (; R_soil, R_het, R_root, R_myc), (; Rb_het, Rb_root, Rb_myc, st)
end