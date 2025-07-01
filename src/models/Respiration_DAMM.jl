
export RespirationDAMM

"""
    RespirationDAMM(NN, predictor, forcing, targets, Q10, kmo, kms) # pedictor is Rgpot, forcing is Temp and Moist   

A DAMM equation type with `targets` and `forcing` terms.
Rh: is the primary target. Respiration_heterotrophic.
Rb:	Output of the neural net, base respiration before modifiers.
Q10, kmo, kms:	Learned physical parameters like temp sensitivity and Michaelis-Menten constants.
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
    p = Array(ds_k(hm.predictors))
    x = ds_k(hm.forcing) # don't propagate names after this
   
    Moist = Array(x([:Moist]))
    Temp = Array(x([:Temp]))

   Rb, st = LuxCore.apply(hm.NN, p, ps.ps, st.st) #! NN(αᵢ(t)) ≡ Rb(T(t), M(t))

   # Michaelis-Menten limitation for Oxygen 
   BD =0.8
   PD =2.52
   Dgas= 1.67
   O2airfraction = 0.2095
   porosity = 1 - BD/PD

   kmo = sigmoid(ps.kmo) .* 1.f-3 .+ 1.f-4
   kms = sigmoid(ps.kms) .* 1.f-6 .+ 1.f-8
   Q10 = sigmoid(ps.Q10) .* 3.f0 .+ 1.f0

   O2 =  @. Dgas * O2airfraction *(porosity - Moist) ^(4.0 /3.0) # Oxygen concentration in the soil air (mol/m^3)
   Ox_limitation = O2 ./ (kmo .+ O2)


   # Michaelis-Menten limitation for substrate 
   Dliq = 3.17
   psx = 4.14f-4 # proportion of C that is DOC 
   Sxtot =0.048 
   
   Sx = @. Sxtot * psx * Dliq * Moist .^ 3 
   S_limitation = Sx ./ (kms .+ Sx)

   Q10term = Q10 .^(0.1f0 * (Temp .- 15.0f0))

    # Model for Respiration_heterotrophic   
    Rh = Rb .* Q10term .* Ox_limitation .* S_limitation # ? should 15°C be the reference temperature also an input variable?

    return (; Rh, O2, Ox_limitation, Moist, Sx, S_limitation, porosity, Rb, Q10term), ( ;st)
end

