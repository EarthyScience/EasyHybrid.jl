export LinearHybridModel
########################################
# Model definition y = ax + b, where 
########################################

struct LinearHybridModel <: EasyHybridModels # lhm
    DenseLayers::Flux.Chain
    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}
    b
end

## Define construction functions for a FF and a recurrent NN

"""
LinearHybridModel(predictors::AbstractArray{Symbol}, forcing::AbstractArray{Symbol},
    out_dim::Int, neurons::Int; b=[1.5f0])

"""
function LinearHybridModel(predictors::AbstractArray{Symbol}, forcing::AbstractArray{Symbol},
    out_dim::Int, neurons::Int; b=[1.5f0])

    in_dim = length(predictors)
    #ch = nn_chain(in_dim, out_dim, neurons)
    ch = Flux.Chain(Flux.Dense(in_dim, neurons), Flux.Dense(neurons, out_dim))
    LinearHybridModel(ch, predictors, forcing, b)
end


function (lhm::LinearHybridModel)(dk, ::Val{:infer})
    #dk = permutedims(dk, (1,3,2))x
    x_matrix = select_predictors(dk, lhm.predictors)
    a = lhm.DenseLayers(x_matrix)
    x = select_variable(dk, lhm.forcing[1])
    pred = a .* x .+ lhm.b # :x1 was hard-coded, this should come from forcing
    return (; a, pred)
end

function (lhm::LinearHybridModel)(df)
    res = lhm(df, :infer)
    return res.pred
end

"""
(lhm::LinearHybridModel)(df, infer::Symbol)
"""
function (lhm::LinearHybridModel)(df, infer::Symbol)
    return lhm(df, Val(infer))
end
# Call @layer to allow for training the custom model
Flux.@layer LinearHybridModel