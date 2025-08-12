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


targets = [:BD, :CF, :SOCconc, :SOCdensity]
Random.seed!(42)
raw = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed.csv"), DataFrame; normalizenames=true)
results_dir = joinpath(@__DIR__, "eval")

# activations were saved as strings; map them back to functions
const ACT = Dict(
    "relu"  => relu,
    "tanh"  => tanh,
    "swish" => swish,
    "gelu"  => gelu,
)

# get hidden layers from results
parse_hidden(s::AbstractString) = begin
    s1 = strip(s, ['(', ')', ' '])
    isempty(s1) && return Int[]
    parse.(Int, split(s1, ','))
end

# same preprocessing you used during search (keep min/max if you need back-transform)
function preprocess!(df::DataFrame, target::Symbol)
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

# pick best row by R² (descending), tie‑break lower MSE
function best_row(df::DataFrame)
    
end



# we’ll store predictions for BD/SOC relationship at the end
pred_store = DataFrame()

for tgt in targets
    println("\n=== Finalizing target: $tgt ===")
    df_res_path = joinpath(results_dir, "parameter_search_$(string(tgt)).csv")
    @assert isfile(df_res_path) "Missing results CSV for $tgt at $df_res_path"

    resdf = CSV.read(df_res_path, DataFrame)
    row = sort(resdf, [:r2, :mse], rev = [true, false])[1, :]

    hconf = parse_hidden(row.hidden_layers)
    bs    = Int(row.batch_size)
    lr    = Float64(row.learning_rate)
    actf  = ACT[row.activation]

    # prep data
    df_d, minv, maxv = preprocess!(deepcopy(raw), tgt)
    predictors = Symbol.(names(df_d))[4:end-1]  # same slice as before
    ka = to_keyedArray(df_d)

    # build model with best hyperparams
    nn = EasyHybrid.constructNNModel(
        predictors, [tgt];
        hidden_layers = collect(hconf),
        activation = actf,
        scale_nn_outputs = false
    )

    # train “for real”
    out_file = joinpath(results_dir, "trained_$(string(tgt)).jld2")
    tr = train(
        nn, ka, ();
        nepochs = 300,                 # bump epochs a bit if you like
        batchsize = bs,
        opt = AdamW(lr),
        training_loss = :mse,
        loss_types = [:mse, :r2],
        shuffleobs = true,
        file_name = out_file           # saves TrainResults & params via EasyHybrid
    )

    # capture best validation metrics
    val_r2s = map(vh -> vh.r2.sum, tr.val_history)
    val_mses = map(vh -> vh.mse.sum, tr.val_history)
    i_best = argmax(val_r2s)
    println("Best epoch: ", i_best, "  R²=", val_r2s[i_best], "  MSE=", val_mses[i_best])
    
    # optional: store predictions for BD/SOC analysis on the full (preprocessed) data
    # If EasyHybrid provides a predict, use it; otherwise adapt to your framework.
    try
        ŷ = EasyHybrid.predict(nn, ka)  # should return AxisKeys/NamedDims-like
        df_pred = DataFrame(ŷ)
        rename!(df_pred, names(df_pred) .=> (n->Symbol(string(n)*"_pred")).(names(df_pred)))
        if isempty(pred_store)
            pred_store = hcat(df_d[:, [:BD, :SOCconc, :CF, :SOCdensity]], df_pred; makeunique=true)
        else
            # add this target’s prediction column if not present
            newcols = setdiff(names(df_pred), names(pred_store))
            pred_store = hcat(pred_store, df_pred[:, newcols]; makeunique=true)
        end
    catch e
        @warn "Prediction collection skipped for $tgt" error = e
    end
end

# ---------------- BD–SOC relationship (quick look) ----------------
# Run this block after all four models are trained AND predictions collected.
if !isempty(pred_store)
    # Choose SOC variable of interest (conc or density). Do both if available.
    for socsym in (:SOCconc_pred, :SOCdensity_pred)
        if hasproperty(pred_store, socsym) && hasproperty(pred_store, :BD_pred)
            x = pred_store.BD_pred
            y = getproperty(pred_store, socsym)
            mask = .!(isnan.(x) .| isnan.(y))
            r = cor(x[mask], y[mask])
            println("Corr(BD_pred, $(String(socsym))) = ", r)

            # quick scatter for sanity
            scatter(x[mask], y[mask], title="BD vs $(String(socsym)) (predicted)",
                    xlabel="BD_pred (scaled/log as trained)", ylabel=String(socsym))
            png(joinpath(results_dir, "rel_BD_vs_$(String(socsym)).png"))
        end
    end
end
