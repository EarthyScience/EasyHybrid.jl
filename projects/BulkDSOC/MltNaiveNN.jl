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
using JLD2

# 02 - multivariate naive NN
testid = "02_multivariate";
results_dir = joinpath(@__DIR__, "eval");

# input
targets = [:BD, :CF, :SOCconc, :SOCdensity];
Random.seed!(42)

raw = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed.csv"), DataFrame; normalizenames=true);
raw = dropmissing(raw); # to be discussed, as now train.jl seems to allow training with sparse data
raw .= Float32.(raw);

# store min/max scalers for back-transform
MINMAX = Dict{Symbol, Tuple{Float64, Float64}}();
# transform per target
df = deepcopy(raw);
for tgt in targets
    if tgt in [:SOCconc, :CF, :SOCdensity]
        df[!, tgt] .= log.(df[!, tgt] .* 1000 .+ 1)  
    end
    minv = minimum(df[!, tgt])
    maxv = maximum(df[!, tgt])
    df[!, tgt] = (df[!, tgt] .- minv) ./ (maxv - minv)
    MINMAX[tgt] = (minv, maxv)
end

# just exclude targets explicitly to be safe
predictors = setdiff(Symbol.(names(df)), targets); # first 3 and last 1
nf = length(predictors);

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

# store results
results = []
best_r2 = -Inf
best_bundle = nothing

# param search by looping....    
for h in hidden_configs, bs in batch_sizes, lr in lrs, act in activations
    println("Testing h=$h, bs=$bs, lr=$lr, activation=$act")

    nn = EasyHybrid.constructNNModel(
        predictors, targets;
        hidden_layers = collect(h),
        activation = act,
        scale_nn_outputs = true
    )

    res = train(
        nn, df, ();                   
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
        return_model = :best
    )

    # retrieve the best epoch metrics: mse and r2
    agg_name = Symbol("mean") 
    r2s  = map(vh -> getproperty(vh, agg_name),  res.val_history.r2)
    mses = map(vh -> getproperty(vh, agg_name), res.val_history.mse)
    best_idx = findmax(r2s)[2]   # index of best r2
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

out_file = joinpath(results_dir, "$(testid)_parameter_search.csv")
CSV.write(out_file, df_results)

# print best model
@assert best_bundle !== nothing "No valid model found for $testid"
bm = best_bundle
@save joinpath(results_dir, "$(testid)_best_model.jld2") ps=bm.ps st=bm.st model=bm.model val_obs_pred=bm.val_obs_pred meta=bm.meta
# @load joinpath(results_dir, "best_model_$(tgt).jld2") ps st model val_obs_pred meta
@info "Best for $testid: h=$(bm.meta.h), bs=$(bm.meta.bs), lr=$(bm.meta.lr), act=$(bm.meta.act), epoch=$(bm.meta.best_epoch), R2=$(round(best_r2, digits=4))"

# back-transform helper
function back_transform(vec::AbstractVector, tgt::Symbol, minmax::Dict{Symbol,<:Tuple})
    mn, mx = minmax[tgt]
    v = vec .* (mx - mn) .+ mn
    if tgt in [:SOCconc, :CF, :SOCdensity]
        v = (exp.(v) .- 1) ./ 1000
    end
    return v
end

# load predictions
jld = joinpath(results_dir, "$(testid)_best_model.jld2")
@assert isfile(jld) "Missing $(jld). Did you train & save best model for $(tname)?"
@load jld val_obs_pred meta
# split output table
val_tables = Dict{Symbol,DataFrame}()
for t in targets
    # expected: t (true), t_pred (pred), and maybe :index if the framework saved it
    have_pred = Symbol(t, :_pred)
    req = Set((t, have_pred))
    @assert issubset(req, Symbol.(names(val_obs_pred))) "val_obs_pred missing $(collect(req)) for $(t). Columns: $(names(val_obs_pred))"
    keep = [:index, t, have_pred] 
    val_tables[t] = val_obs_pred[:, keep]
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
    df_out = val_tables[tname]
    @assert all(in(Symbol.(names(df_out))).([tname, Symbol("$(tname)_pred")])) "Expected columns $(tname) and $(tname)_pred in saved val table."

    y_val_true = back_transform(df_out[:, tname], tname, MINMAX)
    y_val_pred = back_transform(df_out[:, Symbol("$(tname)_pred")], tname, MINMAX)

    r2, mse = r2_mse(y_val_true, y_val_pred)

    plt = histogram2d(
        y_val_true, y_val_pred;
        nbins=(40, 40), cbar=true, xlab="True", ylab="Predicted",
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
df_soc = DataFrame(
    SOCconc_true = back_transform(val_tables[:SOCconc][:, :SOCconc],       :SOCconc, MINMAX),
    SOCconc_pred = back_transform(val_tables[:SOCconc][:, :SOCconc_pred],  :SOCconc, MINMAX),
    BD_true      = back_transform(val_tables[:BD][:,       :BD],           :BD,      MINMAX),
    BD_pred      = back_transform(val_tables[:BD][:,       :BD_pred],      :BD,      MINMAX),
    CF_true      = back_transform(val_tables[:CF][:,       :CF],           :CF,      MINMAX),
    CF_pred      = back_transform(val_tables[:CF][:,       :CF_pred],      :CF,      MINMAX),
);
socdensity_pred = df_soc.SOCconc_pred .* df_soc.BD_pred .* (1 .- df_soc.CF_pred);
socdensity_true = back_transform(val_tables[:SOCdensity][:, :SOCdensity], :SOCdensity, MINMAX);
r2_sd, mse_sd = r2_mse(socdensity_true, socdensity_pred);
plt = histogram2d(
    socdensity_true, socdensity_pred;
    nbins=(40,40), cbar=true, xlab="True SOCdensity", ylab="Pred SOCdensity MTD",
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
    df_soc[:,:BD_pred], df_soc[:,:SOCconc_pred];
    nbins      = (30, 30),
    cbar       = true,
    xlab       = "BD",
    ylab       = "SOCconc",
    color      = cgrad(:bamako, rev=true),
    normalize  = false,
    size = (460, 400)
)   
savefig(plt, joinpath(results_dir, "$(testid)_BD.vs.SOCconc.png"));
