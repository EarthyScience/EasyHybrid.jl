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
using JLD2

testid = "01_univariate"
results_dir = joinpath(@__DIR__, "eval");

targets = [:BD, :CF, :SOCconc, :SOCdensity];
Random.seed!(42);

train_df = CSV.read(joinpath(@__DIR__, "data/lucas_train.csv"), DataFrame; normalizenames=true)
train_df = dropmissing(train_df)
test_df = CSV.read(joinpath(@__DIR__, "data/lucas_test.csv"), DataFrame; normalizenames=true)
test_df = dropmissing(test_df)

# just exclude targets explicitly to be safe
predictors = Symbol.(names(train_df))[5:end-1]; # first 3 and last 1
nf = length(predictors)

# search space
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

best_models = Dict{Symbol, NamedTuple}()
for tgt in targets
    @info "==== target: $tgt ===="

    # save the grid results
    # results = Vector{Tuple{String,Int,Float64,String,Float64,Float64,Int}}() # (h, bs, lr, act, r2, mse, best_epoch)
    results = []

    # store best models states/params
    best_r2 = -Inf
    best_bundle = nothing  # to store (ps, st, model, val_obs_pred DataFrame, meta)

    
    # as well as the validation index, so that we could recover the train, test split that created the
    # models. and we could also visualize later.

    for h in hidden_configs, bs in batch_sizes, lr in lrs, act in activations
        println("Testing h=$h, bs=$bs, lr=$lr, activation=$act")

        nn = EasyHybrid.constructNNModel(
            predictors, [tgt];
            hidden_layers = collect(h),
            activation = act,
            scale_nn_outputs = true
        )

        res = train(
            nn, (train_df,test_df), ();                   
            nepochs = 200,
            batchsize = bs,
            opt = AdamW(lr),
            training_loss = :mse,
            loss_types = [:mse, :r2],
            shuffleobs = true,            
            file_name = nothing,
            random_seed = 42,
            patience = 10,                  
            agg = mean,
            return_model = :best,
            plotting = false
        )

        # retrieve the best epoch metrics: mse and r2
        agg_name = Symbol("mean") 
        r2s  = map(vh -> getproperty(vh, agg_name),  res.val_history.r2)
        mses = map(vh -> getproperty(vh, agg_name), res.val_history.mse)
        best_idx = findmax(r2s)[2]                # index of best r2
        best_r2_here = r2s[best_idx]
        best_mse_here = mses[best_idx]

        push!(results, (h, bs, lr, act, best_r2_here, best_mse_here, best_idx))


        # keep the whole bundle if better
        if !isnan(best_r2_here) && best_r2_here > best_r2
            best_r2 = best_r2_here
            # Keep everything needed to reuse:
            best_bundle = (
                ps = deepcopy(res.ps),
                st = deepcopy(res.st),
                model = nn,
                val_obs_pred = deepcopy(res.val_obs_pred),
                meta = (h=h, bs=bs, lr=lr, act=act, best_epoch=best_idx)
            )
        end
    end

    df_results = DataFrame(
        hidden_layers = [string(r[1]) for r in results],
        batch_size    = [r[2] for r in results],
        learning_rate = [r[3] for r in results],
        activation    = [string(r[4]) for r in results],
        r2            = [r[5] for r in results],
        mse           = [r[6] for r in results],
        best_epoch    = [r[7] for r in results]
    )

    out_file = joinpath(results_dir, "parameter_search_$(string(tgt)).csv")
    CSV.write(out_file, df_results)
 
    # print best model
    @assert best_bundle !== nothing "No valid model found for $tgt"
    bm = best_bundle
    @save joinpath(results_dir, "best_model_$(tgt).jld2") ps=bm.ps st=bm.st model=bm.model val_obs_pred=bm.val_obs_pred meta=bm.meta
    # @load joinpath(results_dir, "best_model_$(tgt).jld2") ps st model val_obs_pred meta
    @info "Best for $tgt: h=$(bm.meta.h), bs=$(bm.meta.bs), lr=$(bm.meta.lr), act=$(bm.meta.act), epoch=$(bm.meta.best_epoch), R2=$(round(best_r2, digits=4))"

end

val_tables = Dict{Symbol,DataFrame}()
best_meta  = Dict{Symbol,NamedTuple}()

for tname in targets
    jld = joinpath(results_dir, "best_model_$(tname).jld2")
    @assert isfile(jld) "Missing $(jld). Did you train & save best model for $(tname)?"
    @load jld val_obs_pred meta
    val_tables[tname] = val_obs_pred
    best_meta[tname]  = meta
end

# helper for metrics calculation
r2_mse(y_true, y_pred) = begin
    ss_res = sum((y_true .- y_pred).^2)
    ss_tot = sum((y_true .- mean(y_true)).^2)
    r2  = 1 - ss_res / ss_tot
    mse = mean((y_true .- y_pred).^2)
    (r2, mse)
end

# accuracy plots for SOCconc, BD, CF in original space
for tname in targets
    y_val_true = val_tables[tname][:, tname]
    y_val_pred = val_tables[tname][:, Symbol("$(tname)_pred")]
    @assert all(in(Symbol.(names(df_out))).([tname, Symbol("$(tname)_pred")])) "Expected columns $(tname) and $(tname)_pred in saved val table."

    r2, mse = r2_mse(y_val_true, y_val_pred)

    plt = histogram2d(
        y_val_pred, y_val_true;
        nbins=(40, 40), cbar=true, xlab="Predicted", ylab="Observed",
        title = string(tname, "\nR²=", round(r2, digits=3), ", MSE=", round(mse, digits=3)),
        normalize=false
    )
    lims = extrema(vcat(y_val_true, y_val_pred))
    Plots.plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]];
        color=:black, linewidth=2, label="1:1 line",
        aspect_ratio=:equal, xlims=lims, ylims=lims
    )
    savefig(plt, joinpath(results_dir, "$(testid)_accuracy_$(tname).png"))
end

# MTD SOCdensity
socdensity_pred = val_tables[:SOCconc][:, :SOCconc_pred] .* val_tables[:BD][:, :BD_pred] .* (1 .- val_tables[:CF][:, :CF_pred]);
socdensity_true = val_tables[:SOCdensity][:, :SOCdensity]
r2_sd, mse_sd = r2_mse(socdensity_true, socdensity_pred);
plt = histogram2d(
    socdensity_pred; socdensity_true, 
    nbins=(40,40), cbar=true, xlab="Pred SOCdensity MTD", ylab="True SOCdensity",
    title = "SOCdensity\nR²=$(round(r2_sd,digits=3)), MSE=$(round(mse_sd,digits=3))",
    normalize=false
)
lims = extrema(vcat(socdensity_true, socdensity_pred))
Plots.plot!(plt, [lims[1], lims[2]], [lims[1], lims[2]];
    color=:black, linewidth=2, label="1:1 line",
    aspect_ratio=:equal, xlims=lims, ylims=lims
)
savefig(plt, joinpath(results_dir, "$(testid)_accuracy_SOCdensity.MTD.png"));

# BD vs SOCconc predictions
plt = histogram2d(
    val_tables[:BD][:, :BD_pred], val_tables[:SOCconc][:, :SOCconc_pred];
    nbins      = (30, 30),
    cbar       = true,
    xlab       = "BD",
    ylab       = "SOCconc",
    color      = cgrad(:bamako, rev=true),
    normalize  = false,
    size = (460, 400)
)   
savefig(plt, joinpath(results_dir, "$(testid)_BD.vs.SOCconc.png"));
