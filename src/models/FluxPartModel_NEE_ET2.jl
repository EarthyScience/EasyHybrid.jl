export FluxPartModel_NEE_ET2

### Model definition for a FP of carbon and water....
struct FluxPartModel_NEE_ET2 <: EasyHybridModels
    RUE_chain::Flux.Chain
    RUE_predictors::AbstractArray{Symbol}
    Rb_chain::Flux.Chain
    Rb_predictors::AbstractArray{Symbol}

    WUE_chain::Flux.Chain
    WUE_predictors::AbstractArray{Symbol}

    Ecoeff_chain::Flux.Chain
    Ecoeff_predictors::AbstractArray{Symbol}

    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}

    Q10
end

"""
`FluxPartModel_NEE_ET2`(
    `RUE_predictors`::AbstractArray{Symbol},
    `Rb_predictors`::AbstractArray{Symbol},
    `WUE_predictors`::AbstractArray{Symbol},
    `Ecoeff_predictors`::AbstractArray{Symbol};
    forcing=[:SW_IN, :TA],
    Q10=[1.5f0],
    neurons=15
)
"""
function FluxPartModel_NEE_ET2(
        RUE_predictors::AbstractArray{Symbol},
        Rb_predictors::AbstractArray{Symbol},
        WUE_predictors::AbstractArray{Symbol},
        Ecoeff_predictors::AbstractArray{Symbol};
        forcing = [:SW_IN, :TA],
        Q10 = [1.5f0],
        neurons = 15
    )
    in_dim_p = length(RUE_predictors)
    in_dim_Rb_p = length(Rb_predictors)
    in_dim_Ecoeff = length(Ecoeff_predictors)
    in_dim_wue = length(WUE_predictors)

    RUE_ch = Dense_RUE_Rb(in_dim_p; neurons)
    Rb_ch = Dense_RUE_Rb(in_dim_Rb_p; neurons)
    Ecoeff_ch = Dense_RUE_Rb(in_dim_Ecoeff; neurons)
    WUE_ch = Dense_RUE_Rb(in_dim_wue; neurons)

    return FluxPartModel_NEE_ET2(
        RUE_ch,
        RUE_predictors,
        Rb_ch,
        Rb_predictors,
        WUE_ch,
        WUE_predictors,
        Ecoeff_ch,
        Ecoeff_predictors,
        union(RUE_predictors, Rb_predictors, Ecoeff_predictors),
        forcing,
        Q10
    )
end

function (m::FluxPartModel_NEE_ET2)(dk, ::Val{:infer})
    RUE_input4Chain = select_predictors(dk, m.RUE_predictors)
    Rb_input4Chain = select_predictors(dk, m.Rb_predictors)
    WUE_input4Chain = select_predictors(dk, m.WUE_predictors)
    Ecoeff_input4Chain = select_predictors(dk, m.Ecoeff_predictors)
    Rb = 100.0f0 * m.Rb_chain(Rb_input4Chain)
    RUE = 1.0f0 * m.RUE_chain(RUE_input4Chain)
    Ecoeff = 1.0f0 * m.Ecoeff_chain(Ecoeff_input4Chain)
    WUE = 1.0f0 * m.Ecoeff_chain(WUE_input4Chain)
    #SW_IN = Matrix(x([:SW_IN]))
    #TA = Matrix(x([:TA]))
    #µmol/m2/s1 =  J/s/m2 * g/MJ / g/mol
    sw_in = select_variable(dk, m.forcing[1])
    ta = select_variable(dk, m.forcing[2])
    GPP = sw_in .* RUE ./ 12.011f0
    Reco = Rb .* m.Q10[1] .^ (0.1f0(ta .- 15.0f0))

    Tr = GPP / WUE
    Evap = Ecoeff .* sw_in

    return (; RUE, Rb, Tr, Evap, GPP = GPP, RECO = Reco, NEE = Reco - GPP, ET = Tr + Evap)
end

function (m::FluxPartModel_NEE_ET2)(dk)
    res = m(dk, :infer)
    return vcat(res.NEE, res.ET)
end

"""
(m::`FluxPartModel_NEE_ET2`)(dk, infer::Symbol)
"""
function (m::FluxPartModel_NEE_ET2)(dk, infer::Symbol)
    return m(dk, Val(infer))
end

# Call @layer to allow for training the custom model
Flux.@layer FluxPartModel_NEE_ET2
