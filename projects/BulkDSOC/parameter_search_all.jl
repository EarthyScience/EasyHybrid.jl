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
target_names = [:BD, :CF, :SOCconc, :SOCdensity] #
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true);

# parameter space
predictors = Symbol.(names(df))[4:end-1];
nf = length(predictors)
hidden_configs = [  
    (256, 128, 64, 32, 16),
    (256, 128, 64, 32),
    (256, 128, 64),
    (256, 128),
    (128, 64, 32, 16),
    (128, 64, 32),
    (128, 64),
    (64, 32, 16),
    (64, 32),
    (32, 16)
];
batch_sizes = [32, 64, 128, 256, 512];
lrs = [1e-2, 1e-3, 1e-4];
activations = [relu, tanh, swish, gelu];

function preprocess!(df, target)
    df = dropmissing(df)
    df .= Float32.(df)

    if target in [:SOCconc, :CF, :SOCdensity]
        df[!, target] .= log.(df[!, target] .* 1000 .+ 1)  
    end

    minv = minimum(df[!, target])
    maxv = maximum(df[!, target])
    df[!, target] = (df[!, target] .- minv) ./ (maxv - minv)

    return df, minv, maxv
end

for tgt in target_names
    df_d, minv, maxv = preprocess!(deepcopy(df), tgt)
    println(tgt, " ", minv, " ", maxv)

    predictors = Symbol.(names(df_d))[4:end-1]
    ka = to_keyedArray(df_d)

    results = []

    for h in hidden_configs, bs in batch_sizes, lr in lrs, act in activations
        println("Testing h=$h, bs=$bs, lr=$lr, activation=$act")

        nn = EasyHybrid.constructNNModel(
            predictors, [tgt];
            hidden_layers = collect(h),
            activation = act,
            scale_nn_outputs = false
        )

        result = train(
            nn, ka, ();
            nepochs = 200,
            batchsize = bs,
            opt = AdamW(lr),
            training_loss = :mse,
            loss_types = [:mse, :r2],
            shuffleobs = true,
            file_name = nothing # skip for now
        )

        best_epoch = argmax(map(vh -> vh.r2.sum, result.val_history))
        best_val = result.val_history[best_epoch]
        safe_r2 = isnan(best_val.r2.sum) ? -Inf : best_val.r2.sum

        push!(results, (h, bs, lr, act, safe_r2, best_val.mse.sum, best_epoch[1]))
    end

    df_results = DataFrame(
        hidden_layers = [string(r[1]) for r in results],
        batch_size    = [r[2] for r in results],
        learning_rate = [r[3] for r in results],
        activation    = [string(r[4]) for r in results],
        r2            = [r[5] for r in results],
        mse           = [r[6] for r in results],
        best_epoch    = [r[7] for r in results]
    )

    out_file = joinpath(@__DIR__, "./eval/parameter_search_$(string(tgt)).csv")
    CSV.write(out_file, df_results)

end

