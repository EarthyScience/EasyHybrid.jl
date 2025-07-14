
export BulkDensitySOC
export compute_bulk_density

"""
    BulkDensitySOC(NN, predictors, targets, oBD)

A hybrid model with a neural network `NN`, `predictors` and one global parameter oBD.
"""
struct BulkDensitySOC{D, T1, T2, T3} <: LuxCore.AbstractLuxContainerLayer{(:NN, :predictors, :targets, :oBD)}
    NN
    predictors # names of predictors
    targets # names of targets
    oBD # organic matter bulk density
    function BulkDensitySOC(NN::D, predictors::T1, targets::T2, oBD::T3) where {D, T1, T2, T3}
        new{D, T1, T2, T3}(NN, collect(predictors), collect(targets), [oBD])
    end
end

# ? oBD is a parameter, so expand the initialparameters!
function LuxCore.initialparameters(::AbstractRNG, layer::BulkDensitySOC)
    ps, _ = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; ps, oBD = layer.oBD,)
end

function LuxCore.initialstates(::AbstractRNG, layer::BulkDensitySOC)
    _, st = LuxCore.setup(Random.default_rng(), layer.NN)
    return (; st)
end


"""
    compute_bulk_density(SOCconc, oBD, mBD)

# model for bulk density based on the Federer (1993) paper http://dx.doi.org/10.1139/x93-131 plus SOC concentrations, density and coarse fraction
"""
function compute_bulk_density(SOCconc, oBD, mBD)
    oF = SOCconc .* 1.724  # TODO: has to be a ratio
    BD = @. oBD * mBD / (oF * mBD + (1.0f0 - oF) * oBD)
    return BD
end


"""
    BulkDensitySOC(NN, predictors, oBD)(ds_k)

# Hybrid model for bulk density based on the Federer (1993) paper http://dx.doi.org/10.1139/x93-131 plus SOC concentrations, density and coarse fraction
"""
function (hm::BulkDensitySOC)(ds_p, ps, st::NamedTuple)
    p = ds_p(hm.predictors)
    
    out, st = LuxCore.apply(hm.NN, p, ps.ps, st.st)

    SOCconc = out[1, :] .* 0.6f0 #TODO has to be a ratio 
    CF = out[2, :]
    mBD = out[3, :] .* (1.5f0 - 0.75f0) .+ 0.75f0 # mineral bulk density

    oBD = sigmoid(ps.oBD) .* (0.4f0 - 0.05f0) .+ 0.05f0

    BD = compute_bulk_density(SOCconc, oBD, mBD)
    
    SOCdensity = @. SOCconc * BD * (1 - CF)

    return (; SOCconc, CF, BD, SOCdensity, mBD), (; mBD, st) # removed oBD from here since its logged via ps.oBD
end