
export RespirationDAMM

"""
    RespirationDAMM(NN, predictor, forcing, targets, Q10, kmo, kms)

A DAMM equation type with `targets` and `forcing` terms.
"""
struct RespirationDAMM{D, T1,T2, T3, T4} <: LuxCore.AbstractLuxContainerLayer{(:NN, :predictors, :forcing, :targets, :Q10, :kmo, :kms)}
    NN
    predictors
    forcing
    targets
    Q10
    kmo # how efficient an enzyme of microbial activity is at utilizing oxygen. Low Kmo means high efficiency as for aerobic organisms.
    kms
    function RespirationDAMM(NN::D, predictors::T1, forcing::T2, targets::T3, Q10::T4, kmo::T4, kms::T4) where {D, T1, T2, T3, T4}
        new{D, T1, T2, T3, T4}(NN, collect(predictors), collect(targets), collect(forcing), [Q10], [kmo], [kms])
    end
end

# ? Q10 is a parameter, so expand the initialparameters!
function LuxCore.initialparameters(::AbstractRNG, layer::RespirationDAMM)
    ps, _ = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; ps, Q10 = layer.Q10, kmo = layer.kmo, kms=layer.kms,)
end
# TODO: trainable vs non-trainable! set example!
# see: https://lux.csail.mit.edu/stable/manual/migrate_from_flux#Implementing-Custom-Layers
function LuxCore.initialstates(::AbstractRNG, layer::RespirationDAMM)
   _, st = LuxCore.setup(Random.default_rng(), layer.NN)
   return (; st)
end

"""
    RespirationRbQ10(NN, predictors, forcing, targets, Q10)(ds_k)

# Model definition `ŷ = Rb(αᵢ(t)) * Q10^((T(t) - T_ref)/10)`

ŷ (respiration rate) is computed as a function of the neural network output `Rb(αᵢ(t))` and the temperature `T(t)` adjusted by the reference temperature `T_ref` (default 15°C) using the Q10 temperature sensitivity factor.
````
"""
function (hm::RespirationDAMM)(ds_k, ps, st::NamedTuple)
    p = ds_k(hm.predictors)
   x = ds_k(hm.forcing) # don't propagate names after this
   x.Moist
   
   Rb, st = LuxCore.apply(hm.NN, p, ps.ps, st.st) #! NN(αᵢ(t)) ≡ Rb(T(t), M(t))

   BD =0.8
   PD =2.52
   Dgas= 1.67
   O2airfraction = 0.2095
   porosity = 1 - BD/PD

   O2 =  Dgas * O2airfraction *(porosity .-x.Moist) .^(4.0 /3.0) # Oxygen concentration in the soil air (mol/m^3)
     # Michaelis-Menten limitation for Oxygen 
   Ox_limitation = O2 ./ (ps.kmo .+ O2)

   # Substrate definition
   Dliq = 3.17
   psx = 4.14f-4 # proportion of C that is DOC 
   Sxtot =0.048 
   Sx = Sxtot * psx * Dliq * x.Moist .^ 3 

   # Substrate limitation 
    S_limitation = Sx ./ (ps.kms .+ Sx)

    Rh = Rb .* ps.Q10 .^(0.1f0 * (x .- 15.0f0)) .* Ox_limitation .* S_limitation # ? should 15°C be the reference temperature also an input variable?

    return (; Rh), (; Rb, st)
end

