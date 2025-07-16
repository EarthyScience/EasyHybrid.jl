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

# check how spacrce the data is
present_counts = [count(!ismissing, df[!, t]) for t in target_names1]
missing_counts = [count(ismissing, df[!, t]) for t in target_names1]
for t in target_names1
    println("$t: present = ", count(!ismissing, df[!, t]),
            ", missing = ", count(ismissing, df[!, t]))
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
result = train(mh, (ds_p, ds_t), (); nepochs=20, batchsize=64, opt=Adam(0.01))

using AxisKeys
y_true = result.y_val
y_pred = KeyedArray(result.ŷ_val, (target_names1, axes(y_true, 2)))

for k in target_names1
    true_vec = collect(y_true(k))
    pred_vec = collect(y_pred(k))

    mask = .!isnan.(true_vec) 
    nk   = count(mask)
    t = true_vec[mask]
    p = pred_vec[mask]

    ss_res = sum((t .- p).^2)
    ss_tot = sum((t .- mean(t)).^2)
    r2     = 1 - ss_res / ss_tot
    mae    = mean(abs.(p .- t))
    bias   = mean(p .- t)

    plt = histogram2d(
        t, p;
        nbins     = (30, 30),
        cbar      = true,
        xlab      = "True",
        ylab      = "Predicted",
        title     = "$k\nR²=$(round(r2, digits=3)),  MAE=$(round(mae,digits=3)),  bias=$(round(bias,digits=3))",
        color     = cgrad(:bamako, rev=true),
        normalize = false,
    )
    lims = extrema(vcat(t, p))
    Plots.plot!(plt,
        [lims[1], lims[2]], [lims[1], lims[2]];
        color=:black, linewidth=2, label="1:1 line",
        aspect_ratio=:equal, xlims=lims, ylims=lims
    )
    savefig(plt, joinpath(@__DIR__, "eval/$(testid)_accuracy_$(k)_val.png"))
end

# BD vs SOCconc, when both are present
mask_bd   = .!isnan.(y_pred(:BD))
mask_soc  = .!isnan.(y_pred(:SOCconc))
mask_pair = mask_bd .& mask_soc

if any(mask_pair)
    BD_pred      = vec(y_pred(:BD))[mask_pair]
    SOCconc_pred = vec(y_pred(:SOCconc))[mask_pair]
    bd_lims = extrema(BD_pred)      
    soc_lims = extrema(SOCconc_pred)

    plt = histogram2d(
        BD_pred, SOCconc_pred;
        nbins     = (30, 30),
        cbar      = true,
        xlab      = "BD",
        ylab      = "SOCconc",
        xlims=bd_lims, ylims=soc_lims,
        color     = cgrad(:bamako, rev=true),
        normalize = false,
        size = (460, 400)
    )
    savefig(plt, joinpath(@__DIR__, "eval/$(testid)_BD.vs.SOCconc.png"))
end

# SOCdensity
BD_pred      = vec(y_pred(:BD))
SOCconc_pred = vec(y_pred(:SOCconc))
CF_pred      = vec(y_pred(:CF))
SOCD_pred = SOCconc_pred .* BD_pred .* (1 .- CF_pred)

(_, _), (_, SOCD_true_val) = splitobs((ds_p, ds_all(:SOCdensity)); at=0.8, shuffle=false)
true_vec = vec(Array(SOCD_true_val))
mask = .!ismissing.(true_vec)

if any(mask)
    t = true_vec[mask]; p = SOCD_pred[mask]

    ss_res = sum((t .- p).^2)
    ss_tot = sum((t .- mean(t)).^2)
    r2     = 1 - ss_res / ss_tot
    mae    = mean(abs.(p .- t))
    bias   = mean(p .- t)

    plt = histogram2d(
        t, p;
        nbins     = (30, 30),
        cbar      = true,
        xlab      = "True",
        ylab      = "Predicted",
        title     = "SOCdensity\nR²=$(round(r2,digits=3)),  MAE=$(round(mae,digits=3)),  bias=$(round(bias,digits=3))",
        color     = cgrad(:bamako, rev=true),
        normalize = false,
    )
    lims = extrema(vcat(collect(t), collect(p)))
    
    Plots.plot!(plt,
        [lims[1], lims[2]], [lims[1], lims[2]];
        color=:black, linewidth=2, label="1:1 line",
        aspect_ratio=:equal, xlims=lims, ylims=lims)    
    savefig(plt, joinpath(@__DIR__, "eval/$(testid)_accuracy_SOCdensity_MTD.png"))
end
