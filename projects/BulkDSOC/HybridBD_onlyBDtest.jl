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

# 04 - hybrid
testid = "04a_hybridBD";
results_dir = joinpath(@__DIR__, "eval");

# input
raw = CSV.read(joinpath(@__DIR__, "data/lucas_preprocessed.csv"), DataFrame; normalizenames=true);
raw = dropmissing(raw); # to be discussed, as now train.jl seems to allow training with sparse data
raw .= Float32.(raw);
df = raw

# mechanistic model
function BD_model(; SOCconc, oBD, mBD)
    BD = (oBD .* mBD) ./ (1.724f0 .* SOCconc .* mBD .+ (1f0 .- 1.724f0 .* SOCconc) .* oBD)
    return (; BD, SOCconc, oBD, mBD)  # supervise both BD and SOCconc
end

# param bounds
parameters = (
    SOCconc = (0.01f0, 0.0f0, 1.0f0),   # fraction
    oBD     = (1.30f0, 0.90f0, 1.80f0),  # g/cm^3
    mBD     = (1.50f0, 0.80f0, 2.0f0),  # global
)

# define param for hybrid model
neural_param_names = [:SOCconc, :oBD]
global_param_names = [:mBD]
forcing = Symbol[]     
targets = [:BD, :SOCconc]       # SOCconc is both a param and a target

# just exclude targets explicitly to be safe
predictors = setdiff(Symbol.(names(df)), targets); # first 3 and last 1
nf = length(predictors);

# search space
batch_sizes = [32, 64, 128, 256, 512];
lrs = [1e-3, 1e-4];
acts = [swish, gelu];

# store results
results = []
best_r2 = -Inf
best_bundle = nothing

for bs in batch_sizes, lr in lrs, act in acts
    @info "Testing bs=$(bs), lr=$(lr), act=$(act)"

    hm = constructHybridModel(
        predictors,              # single NN uses a Vector of predictors
        forcing,
        targets,
        BD_model,
        parameters,
        neural_param_names,
        global_param_names;     
        hidden_layers = [256, 128, 64, 32, 16],
        activation    = act,
        scale_nn_outputs = true,
        input_batchnorm = true,
        start_from_default = true
    )

    res = train(
        hm, df, ();  
        nepochs = 200,
        batchsize = bs,
        opt = AdamW(lr),
        training_loss = :mse,
        loss_types = [:mse, :r2],
        shuffleobs = true,
        file_name = nothing,
        random_seed = 42,
        patience = 20,
        yscale = identity,
        monitor_names = [:oBD, :mBD],
        agg = mean,
        return_model = :best,
        show_progress = false
    )

    # retrieve the best epoch metrics: mse and r2
    agg_name = Symbol("mean") 
    r2s  = map(vh -> getproperty(vh, agg_name),  res.val_history.r2)
    mses = map(vh -> getproperty(vh, agg_name), res.val_history.mse)
    best_idx = findmax(r2s)[2]   # index of best r2
    best_r2_here = r2s[best_idx]
    best_mse_here = mses[best_idx]

    push!(results, (bs, lr, act, best_r2_here, best_mse_here, best_idx))

    # keep the whole bundle if better
    if !isnan(best_r2_here) && best_r2_here > best_r2
        best_r2 = best_r2_here

        # map global mBD -> physical
        mBD_phys = EasyHybrid.scale_single_param(:mBD, res.ps[:mBD], hm.parameters) |> vec |> first
        mBD_raw  = res.ps[:mBD][1]  # unconstrained optimizer value

        # per-sample oBD
        oBD_phys = (hasproperty(res, :val_diffs) && hasproperty(res.val_diffs, :oBD)) ?
                collect(res.val_diffs.oBD) : nothing

        best_bundle = (
            ps = deepcopy(res.ps),
            st = deepcopy(res.st),
            model = hm,
            val_obs_pred = deepcopy(res.val_obs_pred),
            val_diffs = hasproperty(res, :val_diffs) ? deepcopy(res.val_diffs) : nothing,
            meta = (bs=bs, lr=lr, act=act, best_epoch=best_idx,
                    r2=best_r2_here, mse=best_mse_here),
            # convenience fields
            mBD_physical = mBD_phys,
            mBD_unconstr = mBD_raw,
            oBD_phys = oBD_phys
        )
    end
end

df_results = DataFrame(
    batch_size    = [r[1] for r in results],
    learning_rate = [r[2] for r in results],
    activation    = [string(r[3]) for r in results],
    r2            = [r[4] for r in results],
    mse           = [r[5] for r in results],
    best_epoch    = [r[6] for r in results]
)

out_file = joinpath(results_dir, "$(testid)_parameter_search.csv")
CSV.write(out_file, df_results)


# print best model
@assert best_bundle !== nothing "No valid model found for $testid"
bm = best_bundle
@save joinpath(results_dir, "$(testid)_best_model.jld2") \
    ps=best_bundle.ps st=best_bundle.st model=best_bundle.model \
    val_obs_pred=best_bundle.val_obs_pred val_diffs=best_bundle.val_diffs \
    meta=best_bundle.meta \
    mBD_physical=best_bundle.mBD_physical mBD_unconstr=best_bundle.mBD_unconstr \
    oBD_phys=best_bundle.oBD_phys
# @load joinpath(results_dir, "best_model_$(tgt).jld2") ps st model val_obs_pred meta
@info "Best for $testid: bs=$(bm.meta.bs), lr=$(bm.meta.lr), act=$(bm.meta.act), epoch=$(bm.meta.best_epoch), R2=$(round(best_r2, digits=4))"

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


# save / print parameters: mBD and per-sample oBD
# mBD global
mBD_learned = EasyHybrid.scale_single_param(:mBD, bm.ps[:mBD], bm.model.parameters) |> vec |> first
@info "Learned mBD ≈ $(round(mBD_learned, digits=4))"

# Try to fetch per-sample oBD predictions from val_diffs (if the trainer provided them)
oBD_vals = nothing
if bm.val_diffs !== nothing && hasproperty(bm.val_diffs, :oBD)
    oBD_vals = Array(bm.val_diffs.oBD)  # should be a vector matching val rows
    @info "Collected $(length(oBD_vals)) oBD predictions from validation."
    @save joinpath(results_dir, "$(testid)_val_oBD.jld2") oBD_vals
end
