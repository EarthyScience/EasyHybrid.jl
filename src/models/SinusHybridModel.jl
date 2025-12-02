export SinusHybridModel
########################################
# Model definition y = ax + b, where
########################################
struct SinusHybridModel <: EasyHybridModels
    DenseLayers::Flux.Chain
    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}
    b
end

"""
SinusHybridModel(predictors, forcing, out_dim; neurons=15, b=[1.5f0])
"""
function SinusHybridModel(predictors, forcing, out_dim; neurons = 15, b = [1.5f0])
    in_dim = length(predictors)
    ch = DenseNN(in_dim, out_dim, neurons; activation = Flux.softplus)
    return SinusHybridModel(ch, predictors, forcing, b)
end


function (lhm::SinusHybridModel)(dk, ::Val{:infer})
    a = select_predictors(dk, lhm.predictors) |> lhm.DenseLayers
    xi = select_variable(x, lhm.predictors[1])
    pred = a .* sin.(8.0f0 * π * xi) .+ lhm.b
    return (; a, pred)
end

function (lhm::SinusHybridModel)(dk)
    res = lhm(dk, :infer)
    return res.pred
end

"""
(lhm::SinusHybridModel)(dk, infer::Symbol)
"""
function (lhm::SinusHybridModel)(dk, infer::Symbol)
    return lhm(dk, Val(infer))
end

# Call @layer to allow for training the custom model
Flux.@layer SinusHybridModel
