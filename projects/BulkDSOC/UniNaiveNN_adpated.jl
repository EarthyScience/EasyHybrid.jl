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
testid = "01_univariate"

# input
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true);
df_d = dropmissing(df); # complete SOCD
# BD g/cm3 [0,2.2], SOCconc [0,1], CF [0,1], SOCdensity ton/m3
target_names = [:BD, :SOCconc]; # , :CF, :SOCdensity
# df_d.SOCconc .= log.(df_d.SOCconc .* 1000 .+ 1); # to log1p(permille)
# df_d.CF .= log.(df_d.CF .* 100 .+ 1); # to log1p(percent)
# df_d.SOCdensity .= log.(df_d.SOCdensity .* 1000 .+ 1); # to log1p(percent)

names_cov = Symbol.(names(df_d))[4:end-1];
ds_all = to_keyedArray(df_d);
ds_p = ds_all(names_cov);
ds_t =  ds_all(target_names);
ds_t = Flux.normalise(ds_t);

# store predicted matrix
df_out = DataFrame()

# naive NN
nfeatures = length(names_cov)
p_dropout = 0.2;
# tailor activation at output for each target
const OUT_RANGE = Dict(
    :BD         => (0.0, 2.2),     # suggested by Florian
    :SOCconc    => (0.0, 6.5),  # log1p(permille [0,650])
    :CF         => (0.0, 5.0),   # log1p(percent [0,100])
    :SOCdensity => (0.0, 6.0)      # kg/m3 max
)
function plug_head(outname::Symbol)
    last_width = 16

    min_val, max_val = OUT_RANGE[outname]
    return Chain(
        Dense(last_width, 1),  # linear layer
        x -> min_val .+ (max_val - min_val) .* sigmoid.(x)
    )
end
values = ds_t([:SOCconc]);
histogram(
    values;
    # bins = 50,
    xlabel = "SOCdensity",
    ylabel = "Frequency",
    # title = "Histogram of $col",
    lw = 1,
    legend = false
)
for (i, tname) in enumerate(target_names)
    y = ds_t([tname])
    NN = Chain(
        Dense(nfeatures, 128, relu),
        # Dropout(p_dropout),
        # Dense(256, 128, relu),
        # Dropout(p_dropout),
        Dense(128, 64, relu),
        # Dropout(p_dropout),
        Dense(64, 16, relu),
        # Dropout(p_dropout),
        Dense(16, 1)
        # plug_head(tname)  # plug the head for the target
    )

    result = train(NN, (ds_p, y), (); nepochs=150, batchsize=256, opt=AdamW(0.001))

    # converge
    train_loss = result.train_history
    val_loss   = result.val_history
    epochs = 0:length(train_loss)-1
    Plots.plot(epochs, train_loss; label = "Train Loss", lw=2, yscale = :log10)
    Plots.plot!(epochs, val_loss;   label = "Validation Loss", lw=2, yscale = :log10)
    Plots.xlabel!("Epoch")
    Plots.ylabel!("Loss")
    title!("$(tname)")
    savefig(joinpath(@__DIR__, "./eval/$(testid)_converge_$(tname).png"))

    # evaluation
    y_val_true = vec(result[:y_val])
    y_val_pred = vec(result[:ŷ_val])
    print("true", size(y_val_true))
    print("pred", size(y_val_pred))

    # save to matrix
    df_out[!, "true_$(tname)"] = y_val_true
    df_out[!, "pred_$(tname)"] = y_val_pred

    # metrics
    ss_res = sum((y_val_true .- y_val_pred).^2)
    ss_tot = sum((y_val_true .- mean(y_val_true)).^2)
    r2 = 1 - ss_res / ss_tot
    mae = mean(abs.(y_val_pred .- y_val_true))
    bias = mean(y_val_pred .- y_val_true)

    # plot and save
    plt = histogram2d(
        y_val_true, y_val_pred;
        nbins      = (40, 40),
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

# model-then-derive: calulate SOCdensity from individually predicted BD, SOCconc and CF
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

# check BD vs SOCconc
bd_lims = extrema(skipmissing(df_out[:, "pred_BD"]))      
soc_lims = extrema(skipmissing(df_out[:, "pred_SOCconc"]))
plt = histogram2d(
    df_out[:, "pred_BD"], df_out[:, "pred_SOCconc"];
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