using Pkg
Pkg.activate("projects/BulkDSOC")
Pkg.develop(path=pwd())
Pkg.instantiate()

using Revise
using EasyHybrid
using EasyHybrid.Lux
# plotting
using GLMakie
using AlgebraOfGraphics
import Plots as pl

# ? move the `csv` file into the `BulkDSOC/data` folder (create folder)
df = CSV.read(joinpath(@__DIR__, "./data/lucas_preprocessed.csv"), DataFrame, normalizenames=true)
println(size(df))
df_d = dropmissing(df)
println(size(df_d))

target_names = [:BD, :SOCconc, :CF, :SOCdensity]


names_cov = Symbol.(names(df_d))[4:end-1]
#names_cov = Symbol.(names(df_d))[end-40:end-1]
ds_all = to_keyedArray(df_d);


ds_p = ds_all(names_cov);
ds_t =  ds_all(target_names)

length(ds_t) * 10 # number of samples as guidance for number of parameters
nfeatures = length(names_cov)
p_dropout = 0.2
NN = Lux.Chain(
    Dense(nfeatures, 256, Lux.sigmoid),
    Dropout(p_dropout),
    Dense(256, 128, Lux.sigmoid),
    Dropout(p_dropout),
    Dense(128, 64, Lux.sigmoid),
    Dropout(p_dropout),
    Dense(64, 32, Lux.sigmoid),
    Dropout(p_dropout),
    Dense(32, 3, Lux.sigmoid) # Output layer
)
# ? we might need to set output bounds for the expected parameter values

# ? do different initial oBDs
BulkDSOC = BulkDensitySOC(NN, names_cov, target_names, 1.f0)

ps, st = LuxCore.setup(Random.default_rng(), BulkDSOC)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_t_nan = .!isnan.(ds_t)

ls = EasyHybrid.lossfn(BulkDSOC, ds_p, (ds_t, ds_t_nan), ps, LuxCore.testmode(st),  LoggingLoss(train_mode=false)) # testmode to switch of dropout

println(length(names_cov))
out = train(BulkDSOC, (ds_p, ds_t), (:oBD, ); nepochs=100, batchsize=32, opt=AdaMax(0.01));

# plot train history
series(WrappedTuples(out.train_history.mse); axis = (; xlabel = "epoch", ylabel = "loss", xscale=log10, yscale=log10))

# physical parameter
series(out.ps_history; axis=(; xlabel = "epoch", ylabel=""))

# plot trained bulk density function
trained_oBD = out.ps_history.oBD[end] # taking the last one
trained_mBD = out.train_diffs.mBD

SOCrange = range(0.0,0.5; step = 0.01)
median_BDc = compute_bulk_density(SOCrange, trained_oBD, median(trained_mBD))
q25_BDc = compute_bulk_density(SOCrange, trained_oBD, quantile(trained_mBD, 0.25))
q75_BDc = compute_bulk_density(SOCrange, trained_oBD, quantile(trained_mBD, 0.75))

pl.plot(ds_t(:SOCconc), ds_t(:BD), seriestype = :scatter, ylabel = :BD, xlabel = :SOCconc)
pl.plot!(collect(SOCrange), median_BDc, width = 4.0, label = "q50")
pl.plot!(collect(SOCrange), q25_BDc, width = 4.0, label = "q25")
pl.plot!(collect(SOCrange), q75_BDc, width = 4.0, label = "q75")

pl.scatter(out.train_obs_pred[!,:SOCconc], out.train_obs_pred[!,:SOCconc_pred])
pl.scatter(out.train_obs_pred[!,:BD], out.train_obs_pred[!,:BD_pred])
pl.scatter(out.train_obs_pred[!,:CF], out.train_obs_pred[!,:CF_pred])
pl.scatter(out.train_obs_pred[!,:SOCdensity], out.train_obs_pred[!,:SOCdensity_pred])

# AoG
yvars = target_names # [:BD, :SOCconc, :CF, :SOCdensity]
xvars = Symbol.(string.(yvars) .* "_pred")

layers = visual(Scatter, alpha = 0.35)
plt = data(out.train_obs_pred) * layers * mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))
plt *= mapping(color = dims(1) => renamer(string.(xvars))=> "Metrics")
# linear
l_linear = linear() * visual(color=:grey25)
plt += data(out.train_obs_pred) * l_linear *  mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))

with_theme(theme_light()) do 
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

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 300))
    ax = Makie.Axis(fig[1,1], title = "Loss",
        # yscale=log10, 
        xscale=log10
        )
    lines!(ax, WrappedTuples(out.train_history.mse).sum, color=:orangered, label = "train")
    lines!(ax, WrappedTuples(out.val_history.mse).sum, color=:dodgerblue, label ="validation")
    # limits!(ax, 1, 1000, 0.04, 1)
    axislegend()
    fig
end

layers = visual(Scatter, alpha=0.65)
plt = data(out.val_obs_pred) * layers * mapping(:index, [:BD, :BD_pred]) *
    mapping(color=dims(1) => renamer(string.([:BD, :BD_pred])))

with_theme(theme_light()) do
    colors = ["a" => :tomato, "c" => :lime,]
    draw(plt, scales(Color = (; palette = [:grey25, :tomato])),
        legend =(position=:top,),
        figure = (; size = (1400, 400)))    
end