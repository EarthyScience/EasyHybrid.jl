using Pkg
Pkg.activate("projects/BulkDSOC")
Pkg.develop(path=pwd())
Pkg.instantiate()

using AxisKeys
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
using NNlib

# 00 - parameter space search 
testid = "00_search"

df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true);
df_d = dropmissing(df);
df_d .= Float32.(df_d);
tgt = :SOCconc;
df_d.SOCconc .= log.(df_d.SOCconc .* 1000 .+ 1);
minv = minimum(df_d.SOCconc)
maxv = maximum(df_d.SOCconc)
df_d.SOCconc .= (df_d.SOCconc .- minv) ./ (maxv - minv);

# predictors/targets
predictors = Symbol.(names(df_d))[4:end-1];
target = [tgt];
ka = to_keyedArray(df_d);

# parameter space
nf = length(predictors)
hidden_configs = [  
    (32, 16),
    # (256, 128, 64, 32, 16),
    (256, 128, 64, 32),
    (256, 128, 64),
    (256, 128),
    (128, 64, 32, 16),
    (128, 64, 32),
    (128, 64),
    (64, 32, 16),
    (64, 32)
]
batch_sizes = [32, 64, 128, 256, 512];
lrs = [1e-2, 1e-3, 1e-4];
activations = [relu, tanh, swish, gelu];

results = []

for h in hidden_configs, bs in batch_sizes, lr in lrs, act in activations
    println("Testing h=$h, bs=$bs, lr=$lr, activation=$act")

    nn = EasyHybrid.constructNNModel(
        predictors, target;
        hidden_layers = collect(h),     
        activation = act,
        scale_nn_outputs = false
    )

    result = train(
        nn,
        ka,                 
        ();
        nepochs = 20,
        batchsize = bs,
        opt = AdamW(lr),
        training_loss = :mse,
        loss_types = [:mse, :r2],
        shuffleobs = true,
        file_name = nothing # skip for now
    )

    # metrics
    best_epoch = argmax(map(vh -> vh.r2.sum, result.val_history)) # flatten then arg
    best_val = result.val_history[best_epoch]

    push!(results, (h, bs, lr, act,best_val.r2.sum,best_val.mse.sum,best_epoch[1]))
end


best = sort(results, by = x -> x[end], rev = true)[1]
println("Best: ", best)
