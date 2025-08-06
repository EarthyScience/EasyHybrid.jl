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
using NNlib

# 01 - univariate naive NN
testid = "01_univariate"

# input
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true);
df_d = dropmissing(df); # complete SOCD
# BD g/cm3 [0,2.2], SOCconc [0,1], CF [0,1], SOCdensity ton/m3
target_names = [:BD, :SOCconc, :CF, :SOCdensity];
predictors = Symbol.(names(df))[4:end-1];

# pre process targets function
df = dropmissing(df);
df .= Float32.(df);
struct Scaler
    min::Float64
    max::Float64
end
function transform_targets!(df::DataFrame)
    scalers = Dict{Symbol, Scaler}()
    log_targets = [:CF, :SOCconc, :SOCdensity]
    
    for target in [:BD, :CF, :SOCconc, :SOCdensity]
        y = df[!, target]
        if target in log_targets
            y = log.(y .* 1000 .+ 1)
        end
        minv = minimum(y)
        maxv = maximum(y)
        df[!, target] .= (y .- minv) ./ (maxv - minv)
        scalers[target] = Scaler(minv, maxv)
    end
    
    return scalers
end
scalers = transform_targets!(df);


nepoch = 20;

# store predicted matrix
df_out = DataFrame()
for tname in target_names    
    # df_d = dropmissing!(df, subset=[tgt])
    ka = to_keyedArray(df_d)

    results = []

    nn = EasyHybrid.constructNNModel(
        predictors, [tname];
        hidden_layers = [256,128,64,32],
        activation = swish,
        scale_nn_outputs = false
    )

    result = train(
        nn, ka, ();
        nepochs = nepoch,
        batchsize = 128,
        opt = AdamW(0.001),
        training_loss = :mse,
        loss_types = [:mse, :r2],
        shuffleobs = true,
        file_name = nothing, # skip for now
        random_seed = 42
        )

    # converge?
    train_loss = map(l -> l.mse.sum, result.train_history)
    val_loss   = map(l -> l.mse.sum, result.val_history)

    epochs = 0:nepoch
    Plots.plot(epochs, train_loss; label = "Train Loss", lw=2, yscale = :log10)
    Plots.plot!(epochs, val_loss;   label = "Validation Loss", lw=2, yscale = :log10)
    Plots.xlabel!("Epoch")
    Plots.ylabel!("Loss")
    title!("$(tname)")
    savefig(joinpath(@__DIR__, "./eval/$(testid)_converge_$(tname).png"))

    # save pred to matrix
    df_out[!, "true_$(tname)"] = result.val_obs_pred[!, tname]
    df_out[!, "pred_$(tname)"] = result.val_obs_pred[!, "ŷ_$tname"]

 
    last_val = result.val_history[end]

    # modelled SOCD
    plt = histogram2d(
        y_val_true, y_val_pred;
        nbins      = (40, 40),
        cbar       = true,
        xlab       = "True",
        ylab       = "Predicted",
        title      = "$tname\nR2=$(round(last_val.r2.sum, digits=3)),MSE=$(round(last_val.mse.sum, digits=3))",
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


# back transform
function inverse_transform_targets!(df::DataFrame, scalers::Dict{Symbol, Scaler})
    log_targets = [:CF, :SOCconc, :SOCdensity]

    for var in [:BD, :CF, :SOCconc, :SOCdensity]
        scaler = scalers[var]
        target = Symbol("pred_$(var)")
        y_scaled = df[!, target]
        y_log = y_scaled .* (scaler.max - scaler.min) .+ scaler.min
        if var in log_targets
            df[!, target] .= (exp.(y_log) .- 1) ./ 1000
        else
            df[!, target] .= y_log
        end
    end
end

inverse_transform_targets!(df_out, scalers)

# derived SOCD from predicted BD, SOCconc and CF
df_out[:,"calc_pred_SOCdensity"] = df_out[:,"pred_SOCconc"] .* df_out[:,"pred_BD"] .* (1 .- df_out[:,"pred_CF"]) 
sc =  scalers[:SOCdensity]
calc_SOCdensity = (log.(df_out[:, "calc_pred_SOCdensity"] .* 1000 .+ 1) .- sc.min) ./ (sc.max - sc.min)
true_SOCdensity = (log.(df_out[:, "true_SOCdensity"] .* 1000 .+ 1) .- sc.min) ./ (sc.max - sc.min)
 
ss_res = sum((true_SOCdensity .- pred_SOCdensity).^2)
ss_tot = sum((true_SOCdensity .- mean(true_SOCdensity)).^2)
r2 = 1 - ss_res / ss_tot
mse = mean((pred_SOCdensity .- true_SOCdensity).^2)

plt = histogram2d(
    true_SOCdensity, pred_SOCdensity;
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