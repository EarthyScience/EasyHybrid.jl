# CC BY-SA 4.0
project_path = "projects/BulkDSOC"
Pkg.activate(project_path)

# Only instantiate if Manifest.toml is missing
manifest_path = joinpath(project_path, "Manifest.toml")
if !isfile(manifest_path)
    Pkg.develop(path = pwd())
    Pkg.instantiate()
end
using EasyHybrid
using WGLMakie
using Random
using EasyHybrid.MLUtils
using Statistics
using Plots
# using StatsBase

import Flux

# 01 - univariate naive NN using EasyHybrid's SingleNNModel

testid = "01_univariate_EasyHybridNN"

# input
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true)
df_d = dropmissing(df) # complete SOCD

target_names = [:BD, :SOCconc, :CF, :SOCdensity]
names_cov = Symbol.(names(df_d))[4:end-1]
ds_all = to_keyedArray(Float32.(df_d))
ds_p = ds_all(names_cov)
ds_t = ds_all(target_names)
ds_t = Flux.normalise(ds_t)

df_out = DataFrame()

nfeatures = length(names_cov)
p_dropout = 0.2

for (i, tname) in enumerate(target_names)

    y = ds_t([tname])
    # Use EasyHybrid's constructNNModel
    predictors = names_cov
    targets = [tname]
    neural_param_names = [tname]
    model = EasyHybrid.constructNNModel(predictors, targets; hidden_layers=[32], scale_nn_outputs=false)
    #model = EasyHybrid.constructNNModel(predictors, targets; hidden_layers=Chain(Dense(32, 32, tanh), Dense(32, 16, tanh)), scale_nn_outputs=false)

    ps, st = LuxCore.setup(Random.default_rng(), model)
    # Training using EasyHybrid's train function
    result = train(model, (ds_p, y), (); nepochs=100, batchsize=512, opt=AdamW(0.0001, (0.9, 0.999), 0.01), training_loss=:nse, loss_types=[:mse, :nse], shuffleobs=true, yscale=identity)

    y_val_true = vec(result.val_obs_pred[!, tname])
    y_val_pred = vec(result.val_obs_pred[!, Symbol(string(tname, "_pred"))])
    df_out[!, "true_$(tname)"] = y_val_true
    df_out[!, "pred_$(tname)"] = y_val_pred
    ss_res = sum((y_val_true .- y_val_pred).^2)
    ss_tot = sum((y_val_true .- mean(y_val_true)).^2)
    r2 = 1 - ss_res / ss_tot
    mae = mean(abs.(y_val_pred .- y_val_true))
    bias = mean(y_val_pred .- y_val_true)
    plt = histogram2d(
        y_val_true, y_val_pred;
        nbins      = (30, 30),
        cbar       = true,
        xlab       = "True",
        ylab       = "Predicted",
        title      = "$tname\nR2=$(round(r2, digits=3)),MAE=$(round(mae, digits=3)),bias=$(round(bias, digits=3))",
        color      = cgrad(:bamako, rev=true),
        normalize  = false
    )
    lims = extrema(vcat(y_val_true, y_val_pred))
    Plots.plot!(plt,
        [lims[1], lims[2]], [lims[1], lims[2]];
        color=:black, linewidth=2, label="1:1 line",
        aspect_ratio=:equal, xlims=lims, ylims=lims
    )
    savefig(plt, joinpath(@__DIR__, "./eval/$(testid)_accuracy_$(tname).png"))

end

# TODO undo the z-transformation
df_out[:,"pred_calc_SOCdensity"] = df_out[:,"pred_SOCconc"] .* df_out[:,"pred_BD"] .* (1 .- df_out[:,"pred_CF"])
true_SOCdensity = df_out[:, "true_SOCdensity"]
pred_SOCdensity = df_out[:, "pred_calc_SOCdensity"]
ss_res = sum((true_SOCdensity .- pred_SOCdensity).^2)
ss_tot = sum((true_SOCdensity .- mean(true_SOCdensity)).^2)
r2 = 1 - ss_res / ss_tot
mae = mean(abs.(pred_SOCdensity .- true_SOCdensity))
bias = mean(pred_SOCdensity .- true_SOCdensity)
plt = histogram2d(
    true_SOCdensity, pred_SOCdensity;
    nbins      = (30, 30),
    cbar       = true,
    xlab       = "True",
    ylab       = "Predicted",
    title      = "SOCdensity-MTD\nR2=$(round(r2, digits=3)), MAE=$(round(mae, digits=3)), bias=$(round(bias, digits=3))",
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

bd_lims = extrema(skipmissing(df_out[:, "pred_BD"]))      
soc_lims = extrema(skipmissing(df_out[:, "pred_SOCconc"]))
plt = histogram2d(
    df_out[:, "pred_BD"], df_out[:, "pred_SOCconc"];
    nbins      = (30, 30),
    cbar       = true,
    xlab       = "BD",
    ylab       = "SOCconc",
    xlims=bd_lims, ylims=soc_lims,
    color      = cgrad(:bamako, rev=true),
    normalize  = false,
    size = (460, 400)
)   
savefig(plt, joinpath(@__DIR__, "./eval/$(testid)_BD.vs.SOCconc.png")) 