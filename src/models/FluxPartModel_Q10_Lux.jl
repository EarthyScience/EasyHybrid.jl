export FluxPartModelQ10Lux

"""
    FluxPartModelQ10Lux(RUE_NN, Rb_NN, RUE_predictors, Rb_predictors, forcing, targets, Q10)

A flux partitioning model with separate neural networks for RUE (Radiation Use Efficiency) and Rb (basal respiration),
using Q10 temperature sensitivity for respiration calculations.
"""
struct FluxPartModelQ10Lux{D1, D2, T1, T2, T3, T4, T5} <: LuxCore.AbstractLuxContainerLayer{(:RUE_NN, :Rb_NN, :RUE_predictors, :Rb_predictors, :forcing, :targets, :Q10)}
    RUE_NN
    Rb_NN
    RUE_predictors
    Rb_predictors
    forcing
    targets
    Q10
    function FluxPartModelQ10Lux(RUE_NN::D1, Rb_NN::D2, RUE_predictors::T1, Rb_predictors::T2, forcing::T3, targets::T4, Q10::T5) where {D1, D2, T1, T2, T3, T4, T5}
        new{D1, D2, T1, T2, T3, T4, T5}(RUE_NN, Rb_NN, collect(RUE_predictors), collect(Rb_predictors), collect(forcing), collect(targets), [Q10])
    end
end

function LuxCore.initialparameters(::AbstractRNG, layer::FluxPartModelQ10Lux)
    ps_RUE, _ = LuxCore.setup(Random.default_rng(), layer.RUE_NN)
    ps_Rb, _ = LuxCore.setup(Random.default_rng(), layer.Rb_NN)
    return (; RUE = ps_RUE, Rb = ps_Rb, Q10 = layer.Q10)
end

function LuxCore.initialstates(::AbstractRNG, layer::FluxPartModelQ10Lux)
    _, st_RUE = LuxCore.setup(Random.default_rng(), layer.RUE_NN)
    _, st_Rb = LuxCore.setup(Random.default_rng(), layer.Rb_NN)
    return (; RUE = st_RUE, Rb = st_Rb)
end

"""
    FluxPartModelQ10Lux(RUE_NN, Rb_NN, RUE_predictors, Rb_predictors, forcing, targets, Q10)(ds_k, ps, st)

# Model definition
- GPP = SW_IN * RUE(αᵢ(t)) / 12.011  # µmol/m²/s = J/s/m² * g/MJ / g/mol
- Reco = Rb(βᵢ(t)) * Q10^((T(t) - T_ref)/10)
- NEE = Reco - GPP

where:
- RUE(αᵢ(t)) is the radiation use efficiency predicted by neural network
- Rb(βᵢ(t)) is the basal respiration predicted by neural network
- SW_IN is incoming shortwave radiation
- T(t) is air temperature
- T_ref is reference temperature (15°C)
- Q10 is temperature sensitivity factor
"""
function (hm::FluxPartModelQ10Lux)(ds_k, ps, st::NamedTuple)
    # Get inputs for RUE neural network
    RUE_input = ds_k(hm.RUE_predictors)
    
    # Get inputs for Rb neural network
    Rb_input = ds_k(hm.Rb_predictors)
    
    # Get forcing variables
    forcing_data = ds_k(hm.forcing) # don't propagate names after this
    sw_in = Array(forcing_data([:SW_IN]))  # SW_IN
    ta = Array(forcing_data([:TA]))     # TA
    
    # Apply neural networks
    RUE, st_RUE = LuxCore.apply(hm.RUE_NN, RUE_input, ps.RUE, st.RUE) # TODO could be simplified if we move diagnostics out of the tuple with st
    Rb, st_Rb = LuxCore.apply(hm.Rb_NN, Rb_input, ps.Rb, st.Rb)
    
    # Scale outputs
    RUE_scaled = 1.0f0 .* RUE
    Rb_scaled = 1.0f0 .* Rb
    
    # Calculate fluxes
    GPP = sw_in .* RUE_scaled ./ 12.011f0  # µmol/m²/s
    RECO = Rb_scaled .* ps.Q10 .^ (0.1f0 .* (ta .- 15.0f0))
    NEE = RECO .- GPP
    
    # Update states
    new_st = (; RUE = st_RUE, Rb = st_Rb)
    
    return (; NEE, RUE = RUE_scaled, Rb = Rb_scaled, GPP, RECO), (; st = new_st)
end