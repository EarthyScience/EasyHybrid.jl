export GRU_NN, DenseNN

"""
DenseNN(`in_dim`, `out_dim`, neurons)
"""
function DenseNN(in_dim, out_dim, neurons; activation=σ)
    return Flux.Chain(
        Flux.BatchNorm(in_dim),
        Flux.Dense(in_dim => neurons, activation),
        Flux.Dense(neurons => out_dim),
    )
end

"""
GRU_NN(`in_dim`, `out_dim`, neurons)
"""
function GRU_NN(in_dim, out_dim, neurons)
    return Flux.Chain(
        #BatchNorm(in_dim),
        Flux.GRU(in_dim => neurons),
        Flux.Dense(neurons => out_dim),
    )
end

"""
`Dense_RUE_Rb`(`in_dim`; neurons=15, `out_dim`=1, affine=true)
"""
function Dense_RUE_Rb(in_dim; neurons=15, out_dim=1, affine=true)
    return Flux.Chain(
        Flux.BatchNorm(in_dim, affine=affine),
        Flux.Dense(in_dim => neurons, Flux.relu),
        #GRU(5 => 5),
        #Dense(12 => 12, σ), 
        Flux.Dense(neurons => out_dim, σ)
    )
end