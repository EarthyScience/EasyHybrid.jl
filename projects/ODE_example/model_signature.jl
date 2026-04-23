function step(;u, SW_IN, TA, RUE, Rb, Q10, t)
    GPP  = SW_IN .* RUE ./ 12.011f0          # µmol/m²/s
    RECO = Rb .* u .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
    NEE  = RECO .- GPP
    return (; NEE, RECO, GPP, Q10, RUE, Rb)
end

function step(;C, SW_IN, TA, RUE, Rb, Q10, t)
    GPP  = SW_IN .* RUE ./ 12.011f0          # µmol/m²/s
    RECO = Rb .* C .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
    dC = RECO .- GPP
    return (; dC, RECO, GPP, Q10, RUE, Rb)
end



dCdt(;C, RECO, GPP) = RECO(;C, Rb, Q10, TA) .- GPP(;)

mGPP(;SW_IN, RUE) = SW_IN .* RUE ./ 12.011f0          # µmol/m²/s
mRECO(;C, Rb, Q10, TA) = Rb .* C .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))

function mOnePool(;C, SW_IN, TA, RUE, Rb, Q10, t)
    GPP  = SW_IN .* RUE ./ 12.011f0          # µmol/m²/s
    RECO = Rb .* C .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
    dC = RECO .- GPP
    return (; dC, RECO, GPP, Q10, RUE, Rb)
end

function mOnePool(; C, times, SW_IN, TA, RUE, Rb, Q10)
    # step is a closure function that “remembers” the outer parameters (SW_IN, TA, RUE, Rb, Q10);
    # no need to thread them as arguments
    # step only gets the evolving state C and time t
    function step(C, t)
        GPP  = SW_IN .* RUE ./ 12.011      # µmol m⁻² s⁻¹ → gC-ish (unit note as needed)
        RECO = Rb .* C .* Q10 .^ (0.1 .* (TA .- 15.0))
        dC   = RECO .- GPP
        return dC
    end

    return ode(C, times, step)
end