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
raw = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true);
raw = dropmissing(raw);
raw .= Float32.(raw);
predictors = Symbol.(names(raw))[4:end-1];

# parameter space
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

# preprocess, normalize targets
df = deepcopy(raw)
targets = [:BD, :CF, :SOCconc]
all_targets  = [:BD, :CF, :SOCconc, :SOCdensity]
for tgt in targets
    if target in [:CF, :SOCconc, :SOCdensity]
        df[!,tgt] = log.(df[!, tgt] .* 1000 .+ 1) 
    end
    (mn, mx) = MINMAX[tgt]
    df[!, tgt] = (df[!, tgt] .- mn) ./ (mx - mn)
end
    
for h in hidden_configs, bs in batch_sizes, lr in lrs, act in activations
    println("Testing h=$h, bs=$bs, lr=$lr, activation=$act")

    nn = EasyHybrid.constructNNModel(
        predictors, targets;
        hidden_layers = collect(h),
        activation = act,
        scale_nn_outputs = true # since targets scaled
    )

    out_file = joinpath(results_dir, "$(testid)_model_$(string(tgt)).jld2")

    result = train(
        nn, ka, ();
        nepochs = 200,
        batchsize = bs,
        opt = AdamW(lr),
        training_loss = :mse,
        loss_types = [:mse, :r2],
        shuffleobs = true,
        file_name = out_file,
        random_seed=42,
        patience = 5
    )


    best_epoch = argmax(map(vh -> vh.r2.sum, result.val_history))
    best_val = result.val_history[best_epoch]
    safe_r2 = isnan(best_val.r2.sum) ? -Inf : best_val.r2.sum

    push!(results, (h, bs, lr, act, safe_r2, best_val.mse.sum, best_epoch[1]))
    
end

# save results
df_results = DataFrame(
    hidden_layers = [string(r[1]) for r in results],
    batch_size    = [r[2] for r in results],
    learning_rate = [r[3] for r in results],
    activation    = [string(r[4]) for r in results],
    r2            = [r[5] for r in results],
    mse           = [r[6] for r in results],
    best_epoch    = [r[7] for r in results]
)
out_file = joinpath(results_dir, "$(testid)_parameter.search_$(string(tgt)).csv")
CSV.write(out_file, df_results)

# get best
row = sort(df_results, [:r2_sum, :mse_sum], rev=[true,false])[1, :]
hconf = parse_hidden(row.hidden_layers)
bs    = Int(row.batch_size)
lr    = Float64(row.learning_rate)
actf  = ACT[row.activation]
nn_best = EasyHybrid.constructNNModel(
    predictors, targets;
    hidden_layers = collect(hconf),
    activation = actf,
    scale_nn_outputs = true,
)

final_outfile = joinpath(result_dir, "$(testid)_best.jld2")
final = train(
    nn_best, ka, ();
    nepochs = 150,
    batchsize = bs,
    opt = AdamW(lr),
    training_loss = :mse,
    loss_types = [:mse, :r2],
    shuffleobs = true,
    file_name = final_outfile,
    random_seed = 42,
    patience = 10
)

# get validation
dval = final.val_obs_pred
val_idx = Int.(dval.index) 
socd_obs  = raw.SOCdensity[val_idx] 

(mnBD,mxBD) = MINMAX[:BD];       
bd_pred_model  = dval.BD_pred .* (mxBD - mnBD) .+ mnBD
(mnCF,mxCF) = MINMAX[:CF];       
cf_pred_model  = dval.CF_pred .* (mxCF - mnCF) .+ mnCF
(mnSC,mxSC) = MINMAX[:SOCconc];  
sc_pred_model  = dval.SOCconc_pred .* (mxSC - mnSC) .+ mnSC

bd_obs_model   = dval.BD .* (mxBDo - mnBDo) .+ mnBD
cf_obs_model   = dval.CF .* (mxCFo - mnCFo) .+ mnCF
sc_obs_model   = dval.SOCconc .* (mxSCo - mnSCo) .+ mnSC

# back log
cf_pred_raw = (exp.(cf_pred_model) .- 1) ./ 1000
sc_pred_raw = (exp.(sc_pred_model) .- 1) ./ 1000
bd_pred_raw = bd_pred_model

cf_obs_raw  = (exp.(cf_obs_model)  .- 1) ./ 1000
sc_obs_raw  = (exp.(sc_obs_model)  .- 1) ./ 1000
bd_obs_raw  = bd_obs_model

# BD VS SOCconc
plt = histogram2d(
    bd_pred_raw, sc_pred_raw;
    nbins=(30,30), cbar=true,
    xlab="BD (pred, raw)", ylab="SOCconc (pred, raw)",
    normalize=false, size=(520,420)
)
savefig(joinpath(results_dir, "$(testid)_BD.vs.SOCconc.png"))

# SOCdensity accuracy
socd_pred = sc_pred_raw .* bd_pred_raw .* (1 .- cf_pred_raw)
  
mse(y,yhat) = mean((y .- yhat).^2)
r2(y,yhat)  = 1 - sum((y.-yhat).^2) / sum((y .- mean(y)).^2)
plt = histogram2d(
    socd_obs, socd_pred;
    nbins      = (30, 30),
    cbar       = true,
    xlab       = "True",
    ylab       = "Predicted",
    title      = "SOCdensity-MTD\nR2=$(round(r2, digits=3)), MSE=$(round(mse, digits=3))",
    color      = cgrad(:bamako, rev=true),
    normalize  = false
)
lims = extrema(vcat(true_SOCdensity, pred_SOCdensity))
Plots.plot!(plt,
    [lims[1], lims[2]], [lims[1], lims[2]];
    color=:black, linewidth=2, label="1:1 line",
    aspect_ratio=:equal, xlims=lims, ylims=lims
)
savefig(plt, joinpath(@__DIR__, "./eval/$(testid)_accuracy_SOCdensity_MTD.png"))