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
using Zygote
import Plots as pl

# input
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true)
df_d = dropmissing(df) # drop missing values in targets

for col in Symbol.(names(df_d))
    df_d[!, col] = Float32.(df_d[!, col])
end
target_names = [:BD, :SOCconc, :CF, :SOCdensity]
names_cov = Symbol.(names(df_d))[4:end-1]

ds_all = to_keyedArray(df_d);
x = ds_all(names_cov); 

# model structure
nfeatures = length(names_cov)
p_dropout = 0.2

NNs = Dict()
params = Dict()
states = Dict()

# Define model for each target
for tname in target_names
    model = Chain(
        Dense(nfeatures, 256, sigmoid),
        Dropout(p_dropout),
        Dense(256, 128, sigmoid),
        Dropout(p_dropout),
        Dense(128, 64, sigmoid),
        Dropout(p_dropout),
        Dense(64, 32, sigmoid),
        Dropout(p_dropout),
        Dense(32, 1, sigmoid)
    )
    
    ps, st = LuxCore.setup(Random.default_rng(), model)
    NNs[tname] = model
    params[tname] = ps
    states[tname] = st
end

nepochs = 100
opt = Adam(0.001)

for tname in target_names[2:2]
    println("Training for target: $tname")
    model = NNs[tname]
    ps = params[tname]
    st = states[tname]
    y = ds_all([tname])
    y = ndims(y) == 1 ? reshape(y, :, 1) : y
    data_loader = DataLoader((x, y); batchsize=32, shuffle=true)
    optstate = Optimisers.setup(opt, ps)

    # @show ps.layer_1.weight
    # @show model

    lossfn(x, y, ps, st) = begin
        ŷ, _ = model(x, ps, st)
        return sum((ŷ .- y).^2)  # Mean Squared Error
    end

    for epoch in 1:nepochs
        for (xb, yb) in data_loader
            # @show typeof(ps.layer_1.weight)
            grads = Zygote.gradient(ps -> lossfn(xb, yb, ps, st), ps)[1]
            Optimisers.update!(optstate, ps, grads)
        end
    end

    params[tname] = ps

    # Final evaluation
    ŷ, _ = model(x, ps, LuxCore.testmode(st)) # use test mode
    mse = mean((ŷ .- y).^2)
    println("Final MSE for $tname: $mse")
end