using Pkg
Pkg.activate("projects/BulkDSOC")
Pkg.develop(path=pwd())
Pkg.instantiate()

using Revise
using EasyHybrid
using Lux
using Optimisers
using GLMakie
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils
using Statistics
using Plots
using Flux

# 01 - univariate naive NN
testid = "00_search"

# input
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true);
df_d = dropmissing(df); # complete SOCD
# BD g/cm3 [0,2.2], SOCconc [0,1], CF [0,1], SOCdensity ton/m3
tgt = :SOCconc; # , :CF, :SOCdensity
df_d.SOCconc .= log.(df_d.SOCconc .* 1000 .+ 1); # to log1p(permille)
min = minimum(df_d.SOCconc)
max = maximum(df_d.SOCconc)
df_d.SOCconc .= (df_d.SOCconc .- min) ./ (max - min); # normalize to [0,1]

hidden_configs = [
    (128, 64, 16),
    (256, 128, 64),
    (64, 32)
]
batch_sizes = [64, 128, 256]
epochs_list = [100, 200]
lrs = [1e-3, 5e-4, 1e-4]

results = []

for h in hidden_configs, bs in batch_sizes, nepochs in epochs_list, lr in lrs
    println("Testing h=$h, bs=$bs, epochs=$nepochs, lr=$lr")
    
    # build NN dynamically
    NN = EasyHybrid.constructNNModel(predictors, targets; hidden_layers=Chain(Dense(32, 32, tanh), Dense(32, 16, tanh)), scale_nn_outputs=false)
    ps, st = LuxCore.setup(Random.default_rng(), model)

    layers = []
    push!(layers, Dense(nfeatures, h[1], relu))
    for i in 2:length(h)
        push!(layers, Dense(h[i-1], h[i], relu))
    end
    push!(layers, Dense(h[end], 1))
    NN = Chain(layers...)

    # train
    result = train(NN, (ds_p, y); nepochs=nepochs, batchsize=bs, opt=AdamW(lr))
    
    # evaluate (reuse your code)
    y_val_true = vec(result[:y_val])
    y_val_pred = vec(result[:ŷ_val])
    ss_res = sum((y_val_true .- y_val_pred).^2)
    ss_tot = sum((y_val_true .- mean(y_val_true)).^2)
    r2 = 1 - ss_res / ss_tot

    push!(results, (h, bs, nepochs, lr, r2))
end

# Sort and show best
best = sort(results, by=x->x[end], rev=true)[1]
println("Best: ", best)
