export FluxPartModel_Q10

struct FluxPartModel_Q10 <: EasyHybridModels
    RUE_chain::Flux.Chain
    RUE_predictors::AbstractArray{Symbol}
    Rb_chain::Flux.Chain
    Rb_predictors::AbstractArray{Symbol}

    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}

    Q10
end

"""
FluxPartModel_Q10(RUE_predictors, Rb_predictors; Q10=[1.5f0])
"""
function FluxPartModel_Q10(RUE_predictors, Rb_predictors; forcing = [:SW_IN, :TA], Q10 = [1.5f0], neurons = 15)
    RUE_ch = Dense_RUE_Rb(length(RUE_predictors); neurons)
    Rb_ch = Dense_RUE_Rb(length(Rb_predictors); neurons)
    return FluxPartModel_Q10(
        RUE_ch,
        RUE_predictors,
        Rb_ch,
        Rb_predictors,
        union(RUE_predictors, Rb_predictors),
        forcing,
        Q10
    )
end

function (m::FluxPartModel_Q10)(dk, ::Val{:infer})
    RUE_input4Chain = select_predictors(dk, m.RUE_predictors)
    Rb_input4Chain = select_predictors(dk, m.Rb_predictors)
    Rb = 100.0f0 * m.Rb_chain(Rb_input4Chain)
    RUE = 1.0f0 * m.RUE_chain(RUE_input4Chain)
    #SW_IN = Matrix(x([:SW_IN]))
    #TA = Matrix(x([:TA]))
    #µmol/m2/s1 =  J/s/m2 * g/MJ / g/mol
    sw_in = select_variable(dk, m.forcing[1])
    ta = select_variable(dk, m.forcing[2])

    GPP = sw_in .* RUE ./ 12.011f0
    Reco = Rb .* m.Q10[1] .^ (0.1f0(ta .- 15.0f0))
    return (; RUE, Rb, GPP = GPP, RECO = Reco, NEE = Reco - GPP)
end

function (m::FluxPartModel_Q10)(dk)
    res = m(dk, :infer)
    return res.NEE
end

"""
(m::`FluxPartModel_Q10`)(dk, infer::Symbol)
"""
function (m::FluxPartModel_Q10)(dk, infer::Symbol)
    return m(dk, Val(infer))
end

# Call @layer to allow for training the custom model
Flux.@layer FluxPartModel_Q10
