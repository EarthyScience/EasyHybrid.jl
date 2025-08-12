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
df_d = dropmissing(df_d);
df_d .= Float32.(df_d);
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
scalers = transform_targets!(df_d);


# store predicted matrix
dfo = DataFrame[];

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
        nepochs = 150,
        batchsize = 128,
        opt = AdamW(0.001),
        training_loss = :mse,
        loss_types = [:mse, :r2],
        shuffleobs = true,
        file_name = nothing,
        random_seed = 42,
        patience = 5  
    )
    
    # save pred to matrix
    # 
    dft = result.val_obs_pred
    rename!(dft, :index => Symbol("$(tname)_index"))
    push!(dfo, dft)
end

df_out = reduce(hcat, dfo)

# back transform
function inverse_transform_targets!(df::DataFrame, scalers::Dict{Symbol, Scaler})

    for var in [:BD, :CF, :SOCconc, :SOCdensity]
        scaler = scalers[var]
        is_log = var in [:CF, :SOCconc, :SOCdensity]

        for col in [var, Symbol("$(var)_pred")]
            if col in names(df)
                y_scaled = df[!, col]
                y = y_scaled .* (scaler.max - scaler.min) .+ scaler.min
                if is_log
                    df[!, col] .= (exp.(y) .- 1) ./ 1000
                else
                    df[!, col] .= y
                end
            end
        end
    end
end


#inverse_transform_targets!(df_out, scalers)

for tname in target_names
    y_val_true = df_out[:, tname]
    y_val_pred = df_out[:, Symbol("$(tname)_pred")]

    ss_res = sum((y_val_true .- y_val_pred).^2)
    ss_tot = sum((y_val_true .- mean(y_val_true)).^2)
    r2  = 1 - ss_res / ss_tot
    mse = mean((y_val_true .- y_val_pred).^2)

    plt = histogram2d(
        y_val_true, y_val_pred;
        nbins = (40, 40),
        cbar = true,
        xlab = "True",
        ylab = "Predicted",
        title = "$tname\nR²=$(round(r2, digits=3)), MSE=$(round(mse, digits=3))",
        color = cgrad(:bamako, rev=true),
        normalize = false
    )
    lims = extrema(vcat(y_val_true, y_val_pred))
    Plots.plot!(plt,
        [lims[1], lims[2]], [lims[1], lims[2]];
        color = :black, linewidth = 2, label = "1:1 line",
        aspect_ratio = :equal, xlims = lims, ylims = lims
    )
    savefig(plt, joinpath(@__DIR__, "./eval/$(testid)_accuracy_$(tname).png"))
end


# MTD, model then derive, derived SOCD from predicted BD, SOCconc and CF
df_out[:,"calc_pred_SOCdensity"] = df_out[:,"SOCconc_pred"] .* df_out[:,"BD_pred"] .* (1 .- df_out[:,"CF_pred"]); 
true_SOCdensity = df_out[:,"SOCdensity"];
pred_SOCdensity = df_out[:,"calc_pred_SOCdensity"]; 
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