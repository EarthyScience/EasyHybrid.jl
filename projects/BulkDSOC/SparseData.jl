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
# using StatsBase

# 04 - sparse data NN
testid = "04_sparcity"

# input
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true)
names_cov = Symbol.(names(df))[4:end-1]
df_d = dropmissing(df, names_cov)              # only predictors checked

# BD g/cm3 [0,2.2], SOCconc [0,1], CF [0,1], SOCdensity ton/m3
target_names1 = [:BD, :SOCconc, :CF]
target_names2 = [:BD, :SOCconc, :CF, :SOCdensity]
for t in target_names1
    df_d[!, t] = Float64.( coalesce.(df_d[!, t], NaN) )
end

ds_all = to_keyedArray(df_d);
ds_p = ds_all(names_cov);
ds_t = ds_all(target_names1);  

# mimic the one in BulkDensitySOC
nfeatures = length(names_cov)
p_dropout = 0.2
trunk = Chain(Dense(nfeatures, 256, sigmoid), Dropout(0.2),
              Dense(256,128,sigmoid), Dropout(0.2),
              Dense(128,64,sigmoid), Dropout(0.2),
              Dense(64,32,sigmoid))

head_bd  = Dense(32,1)  # not (0,1)
head_soc = Dense(32,1,sigmoid) 
head_cf  = Dense(32,1,sigmoid)

mh = MultiHeadNN(trunk, head_bd, head_soc, head_cf, names_cov, [:BD, :SOCconc, :CF])

ds_t =  ds_all(target_names1)
result = train(mh, (ds_p, ds_t), save_ps;nepochs=100, batchsize=32, opt=Adam(0.01))

# eval
using AxisKeys
y_true   = result.y_val  # key things
y_pred = KeyedArray(result.ŷ_val, (target_names1, axes(y_pred, 2)))

for k in target_names1
    true_vec = y_true(k) |> collect      
    pred_vec = y_pred(k) |> collect

    ss_res = sum((true_vec .- pred_vec).^2)
    ss_tot = sum((true_vec .- mean(true_vec)).^2)
    r2     = 1 - ss_res / ss_tot
    mae    = mean(abs.(pred_vec .- true_vec))
    bias   = mean(pred_vec .- true_vec)

    plt = histogram2d(
        true_vec, pred_vec;
        nbins     = (30, 30),
        cbar      = true,
        xlab      = "True $k",
        ylab      = "Predicted $k",
        title     = "$k\nR2=$(round(r2, digits=3)),MAE=$(round(mae, digits=3)),bias=$(round(bias, digits=3))",
        color     = cgrad(:bamako, rev = true),
        normalize = false,
    )
    lims = extrema(vcat(true_vec, pred_vec))
    Plots.plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]];
          color = :black, linewidth = 2, label = "1:1")

    savefig(plt, joinpath(@__DIR__,"eval/$(testid)_accuracy_$(k)_val.png"))
end
# check BD vs SOCconc
BD_pred = vec(y_pred(:BD))    
SOCconc_pred  = vec(y_pred(:SOCconc))
plt = histogram2d(
    BD_pred, SOCconc_pred;
    nbins      = (30, 30),
    cbar       = true,
    xlab       = "BD",
    ylab       = "SOCconc",
    #title      = "SOCdensity-MTD\nR2=$(round(r2, digits=3)), MAE=$(round(mae, digits=3)), bias=$(round(bias, digits=3))",
    color      = cgrad(:bamako, rev=true),
    normalize  = false
)
savefig(plt, joinpath(@__DIR__, "./eval/$(testid)_BD.vs.SOCconc.png"))

CF_pred = vec(y_pred(:CF))
SOCD_pred = SOCconc_pred .* BD_pred .* (1 .- CF_pred) 
(_, _), (_, SOCD_true_val) = splitobs((ds_p, ds_all(:SOCdensity)); at = 0.8, shuffle = false)
pred_vec = collect(SOCD_pred)   
true_vec = vec(Array(SOCD_true_val))
ss_res = sum((true_vec .- pred_vec).^2)
ss_tot = sum((true_vec .- mean(true_vec)).^2)
r2     = 1 - ss_res / ss_tot
mae    = mean(abs.(pred_vec .- true_vec))
bias   = mean(pred_vec .- true_vec)
plt = histogram2d(
    true_vec, pred_vec;
    nbins      = (30, 30),
    cbar       = true,
    xlab       = "True",
    ylab       = "Predicted",
    title      = "SOCdensity-MTD\nR2=$(round(r2, digits=3)), MAE=$(round(mae, digits=3)), bias=$(round(bias, digits=3))",
    color      = cgrad(:bamako, rev=true),
    normalize  = false
)
lims = extrema(vcat(true_vec, pred_vec))
Plots.plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]];
          color = :black, linewidth = 2, label = "1:1")
savefig(plt, joinpath(@__DIR__,"eval/$(testid)_accuracy_SOCdensity_val.png"))
