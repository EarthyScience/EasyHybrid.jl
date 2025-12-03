export LinearHybridModel_2outputs

struct LinearHybridModel_2outputs <: EasyHybridModels # lhm
    DenseLayers::Flux.Chain
    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}
    a
    b
end

## Define construction functions for a FF and a recurrent NN

"""
`LinearHybridModel_2outputs`(predictors::AbstractArray{Symbol}, forcing::AbstractArray{Symbol},
    `out_dim`::Int, neurons::Int, `nn_chain`; a=[1.0f0], b=[1.5f0])

    - nn_chain :: DenseNN, a Dense neural network
"""
function LinearHybridModel_2outputs(
        predictors::AbstractArray{Symbol}, forcing::AbstractArray{Symbol},
        out_dim::Int, neurons::Int, nn_chain; a = [1.0f0], b = [1.5f0]
    )

    in_dim = length(predictors)
    ch = nn_chain(in_dim, out_dim, neurons)
    return LinearHybridModel_2outputs(ch, predictors, forcing, a, b)
end


function (lhm::LinearHybridModel_2outputs)(dk, ::Val{:infer})
    x_matrix = select_predictors(dk, lhm.predictors)
    a = lhm.DenseLayers(x_matrix)
    x1 = select_variable(dk, lhm.forcing[1])
    x2 = select_variable(dk, lhm.forcing[2])

    pred1 = a .* x1 .+ lhm.b
    pred2 = a .* lhm.a .* x2

    return (; a, pred1, pred2)
end

function (lhm::LinearHybridModel_2outputs)(df)
    res = lhm(df, :infer)
    return vcat(res.pred1, res.pred2)
end

"""
`LinearHybridModel_2outputs`(df, infer::Symbol)
"""
function (lhm::LinearHybridModel_2outputs)(df, infer::Symbol)
    return lhm(df, Val(infer))
end
# Call @layer to allow for training the custom model
Flux.@layer LinearHybridModel_2outputs
