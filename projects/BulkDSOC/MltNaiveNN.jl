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

# 02 - multivariate naive(?) NN
testid = "02_multivariate"

# input
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true)
df_d = dropmissing(df) # complete SOCD

# BD g/cm3 [0,2.2], SOCconc [0,1], CF [0,1], SOCdensity ton/m3
target_names1 = [:BD, :SOCconc, :CF]
target_names2 = [:BD, :SOCconc, :CF, :SOCdensity]
names_cov = Symbol.(names(df_d))[4:end-1]
ds_all = to_keyedArray(df_d);
ds_p = ds_all(names_cov);
 ds_t =  ds_all(target_names1)


# mimic the one in BulkDensitySOC
nfeatures = length(names_cov)
p_dropout = 0.2
NN = Chain(
    Dense(nfeatures, 256, sigmoid), 
    Dropout(p_dropout),
    Dense(256, 128, sigmoid),       
    Dropout(p_dropout),
    Dense(128,  64, sigmoid),       
    Dropout(p_dropout),
    Dense( 64,  32, sigmoid),       
    Dropout(p_dropout),
    Dense( 32,   3),   # raw
    x -> cat(x[1:1, :], sigmoid.(x[2:3, :]); dims=1)  # BD stays the same, SOCconc and CF are sigmoid  
)

result = train(NN, (ds_p, ds_t), (); nepochs=100, batchsize=32, opt=AdaMax(0.01));

# eval
using AxisKeys
y_true   = result.y_val  # key things
y_pred = KeyedArray(result.ŷ_val, (target_names1, axes(y_true, 2)))

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
        xlab      = "True",
        ylab      = "Predicted",
        title     = "$k\nR2=$(round(r2, digits=3)),MAE=$(round(mae, digits=3)),bias=$(round(bias, digits=3))",
        color     = cgrad(:bamako, rev = true),
        normalize = false,
    )
    lims = extrema(vcat(true_vec, pred_vec))
    Plots.plot!(plt,
        [lims[1], lims[2]], [lims[1], lims[2]];
        color=:black, linewidth=2, label="1:1 line",
        aspect_ratio=:equal, xlims=lims, ylims=lims
    )
    savefig(plt, joinpath(@__DIR__,"eval/$(testid)_accuracy_$(k)_val.png"))
end

# check BD vs SOCconc
BD_pred = vec(y_pred(:BD))    
SOCconc_pred  = vec(y_pred(:SOCconc))
bd_lims = extrema(BD_pred)      
soc_lims = extrema(SOCconc_pred)
plt = histogram2d(
    BD_pred, SOCconc_pred;
    nbins      = (30, 30),
    cbar       = true,
    xlab       = "BD",
    ylab       = "SOCconc",
    xlims=bd_lims, ylims=soc_lims,
    #title      = "SOCdensity-MTD\nR2=$(round(r2, digits=3)), MAE=$(round(mae, digits=3)), bias=$(round(bias, digits=3))",
    color      = cgrad(:bamako, rev=true),
    normalize  = false,
    size = (460, 400)
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
Plots.plot!(plt,
    [lims[1], lims[2]], [lims[1], lims[2]];
    color=:black, linewidth=2, label="1:1 line",
    aspect_ratio=:equal, xlims=lims, ylims=lims
)
savefig(plt, joinpath(@__DIR__,"eval/$(testid)_accuracy_SOCdensity_MTD.png"))
