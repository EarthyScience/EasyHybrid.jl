using Pkg
Pkg.activate("projects/RsoilComponents")
Pkg.develop(path=pwd())
Pkg.instantiate()

using Revise

using EasyHybrid
using GLMakie
using AlgebraOfGraphics
using Statistics

script_dir = @__DIR__
include(joinpath(script_dir, "data", "prec_process_data.jl"))

df = dfall[!, Not(:timesteps)]
ds_keyed = to_keyedArray(Float32.(df))

target_names = [:R_soil]
forcing_names = [:cham_temp_filled]

Temp = Array(ds_keyed(collect(target_names)))
mRbQ10(hybridRs.Rb, hybridRs.Q10, Temp, 15.0f0)

hybridRs = RbQ10_2p(target_names, forcing_names, 2.5f0, 1.f0)
hybridRs = RbQ10_2p(target_names, (:cham_temp_filled,), 2.5f0, 1.f0)

out = train(hybridRs, ds_keyed, (:Rb, :Q10); nepochs=100, batchsize=512, opt=Adam(0.01));

series(out.ps_history; axis=(; xlabel = "epoch", ylabel=""))

# Define neural network
NN = Chain(Dense(1, 15, relu), Dense(15, 15, relu), Dense(15, 1));
# instantiate Hybrid Model
RbQ10 = RespirationRbQ10(NN, (:moisture_filled,), target_names, forcing_names, 1.5f0) # ? do different initial Q10s
# train model
out2 = train(RbQ10, ds_keyed, (:Q10, ); nepochs=200, batchsize=512, opt=Adam(0.01));

series(out2.ps_history; axis=(; xlabel = "epoch", ylabel=""))

import Plots as pl


pl.scatter(df.cham_temp_filled, df.R_soil, alpha = 0.05)

TempRange = collect(range(0,30,100))

mod = mRbQ10(out.ps.Rb,out.ps.Q10, TempRange, 0)
pl.plot!(TempRange,mod, linewidth=3, label = "dumb Q10")

mod = mRbQ10(out.ps.Rb,out2.ps.Q10, TempRange, 0)
pl.plot!(TempRange,mod, linewidth=3, label = "Hybrid Q10")

pl.scatter(df.moisture, df.R_soil, alpha = 0.05)


series(out.train_history; axis = (; xlabel = "epoch", ylabel = "loss", xscale=log10, yscale=log10))

target_names = [:R_soil, :R_root, :R_myc, :R_het]
hybridRs = RbQ10_2p(target_names, (:cham_temp_filled,), 2.5f0, 1.f0)




NN = Lux.Chain(Dense(2, 15, Lux.sigmoid), Dense(15, 15, Lux.sigmoid), Dense(15, 3, x -> x^2));
Rsc = Rs_components(NN, (:rgpot, :moisture_filled), target_names, (:cham_temp_filled,), 2.5f0, 2.5f0, 2.5f0)

out = train(Rsc, ds_keyed, (:Q10_het, :Q10_myc, :Q10_root, ); nepochs=100, batchsize=512, opt=Adam(0.01));

series(out.ps_history; axis=(; xlabel = "epoch", ylabel=""))

series(out.train_history; axis = (; xlabel = "epoch", ylabel = "loss", xscale=log10, yscale=log10))

out.train_obs_pred

using AlgebraOfGraphics
scatter_layer = 
    data(out.train_obs_pred) *                        # our DataFrame source
    mapping(:R_soil, :R_soil_pred) *  # first pos. arg → x, second → y
    visual(Scatter;                   # choose a Makie scatter
           markersize = 8,            # tweak marker size
           color       = :steelblue) # choose a color


draw(scatter_layer)


# AoG
yvars = target_names # [:BD, :SOCconc, :CF, :SOCdensity]
xvars = Symbol.(string.(yvars) .* "_pred")

layers = visual(Scatter, alpha = 0.1)
plt = data(out.train_obs_pred) * layers * mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))
plt *= mapping(color = dims(1) => renamer(string.(yvars))=> "Obs vs Pred")
# linear
l_linear = linear() * visual(color=:grey25)
plt += data(out.train_obs_pred) * l_linear *  mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))

with_theme(theme_minimal()) do 
   draw(plt, scales(
        X = (; label = rich("Prediction", font=:bold)),
        Y = (; label = "Observation"),
        Color = (; palette = [:tomato, :teal, :orange, :dodgerblue3])
   ),
    # legend = (; show = false),
    legend = (; position=:bottom, titleposition=:left, merge=false),
    facet = (; linkxaxes = :none, linkyaxes = :none,),
    figure = (; size = (1400, 400))
) 
end
using DataFrames, Statistics, AlgebraOfGraphics, CairoMakie

# ─────────────────────────────────────────────────────────────
# 1. your existing vars + data
# ─────────────────────────────────────────────────────────────
yvars   = target_names
xvars   = Symbol.(string.(yvars) .* "_pred")
df      = out.train_obs_pred

# ─────────────────────────────────────────────────────────────
# 2. compute R² per variable, dropping NaN/missing first
# ─────────────────────────────────────────────────────────────
r2_list     = Float64[]
xmin_list   = Float64[]
xrange_list = Float64[]
ymax_list   = Float64[]
yrange_list = Float64[]

for (y, x) in zip(yvars, xvars)
    yobs = df[!, y]
    ypred = df[!, x]

    # 2a) build mask: drop missing or NaN in either vector
    good = .!ismissing.(yobs) .& .!ismissing.(ypred) .&
           .!isnan.(yobs)    .& .!isnan.(ypred)

    yobs_clean = yobs[good]
    ypred_clean = ypred[good]

    yall = vcat(yobs_clean, ypred_clean)

    # 2b) compute R² = 1 - SS_res/SS_tot
    ss_res = sum((yobs_clean .- ypred_clean).^2)
    ss_tot = sum((yobs_clean .- mean(yobs_clean)).^2)
    push!(r2_list, 1 - ss_res/ss_tot)

    # 2c) for positioning the annotation
    push!(xmin_list, minimum(ypred_clean))
    push!(xrange_list, maximum(ypred_clean) - minimum(ypred_clean))
    push!(ymax_list, maximum(yall))
    push!(yrange_list, maximum(yobs_clean) - minimum(yobs_clean))
end

# 2d) make string labels
r2_labels = ["R²=$(round(r, digits=2))" for r in r2_list]

r2_df = DataFrame(
  dims1    = string.(yvars),   # must match your facet key
  R2_label = r2_labels,
  x        = xmin_list .+ 0.05 .* xrange_list,
  y        = ymax_list .- 0.05 .* yrange_list
)

# ─────────────────────────────────────────────────────────────
# 3. rest of plotting (same as before)
# ─────────────────────────────────────────────────────────────

yvars = target_names # [:BD, :SOCconc, :CF, :SOCdensity]
xvars = Symbol.(string.(yvars) .* "_pred")

layers = visual(Scatter, alpha = 0.05)
plt = data(out.train_obs_pred) * layers * mapping(xvars, yvars, col = dims(1) => renamer(string.(yvars) .* "\n" .* r2_df.R2_label))
plt *= mapping(color = dims(1) => renamer(string.(yvars))=> "Obs vs Pred")

draw(plt, axis=(aspect=1, limits = ((0, ymax_list[1]), (0, ymax_list[1]))), figure = (size = (1600, 400),),)







# legacy
NN = Lux.Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 3));
Rsc = Rs_components(NN, (:rgpot, :moisture_filled), target_names, (:cham_temp_filled,), 2.5f0, 2.5f0, 2.5f0)
ds_p_f, ds_t = EasyHybrid.prepare_data(Rsc, ds_keyed)

ps, st = LuxCore.setup(Random.default_rng(), Rsc)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_t_nan = .!isnan.(ds_t)

ls = EasyHybrid.lossfn(Rsc, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss(train_mode=false))

out = train(Rsc, (ds_p_f, ds_t), (:Q10_het, :Q10_myc, :Q10_root, ); nepochs=100, batchsize=512, opt=Adam(0.01));

series(out, :Q10_het, :Q10_myc, :Q10_root, title="Rs components Q10 parameters", xlabel="Epochs", ylabel="Q10 value")