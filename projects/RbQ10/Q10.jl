# activate the project's environment and instantiate dependencies
using Pkg
Pkg.activate("projects/RbQ10")
Pkg.develop(path=pwd())
Pkg.instantiate()

# start using the package
using EasyHybrid

# for Plotting
using GLMakie
using AlgebraOfGraphics

# load data
df_o = CSV.read(joinpath(@__DIR__, "./data/Rh_AliceHolt_forcing_filled.csv"), DataFrame)

# some pre-processing
df = copy(df_o)
df[!, :Temp] = df[!, :Temp] .- 273.15 # convert to Celsius
# df = filter(:Respiration_heterotrophic => !isnan, df)
rename!(df, :Respiration_heterotrophic => :Rh)  # rename as in hybrid model

ds_keyed = to_keyedArray(Float32.(df)) # predictors + forcing

# Define neural network
NN = Chain(Dense(2, 15, relu), Dense(15, 15, relu), Dense(15, 1));
# instantiate Hybrid Model
RbQ10 = RespirationRbQ10(NN, (:Rgpot, :Moist), (:Rh, ), (:Temp,), 2.5f0) # ? do different initial Q10s
# train model
out = train(RbQ10, ds_keyed, (:Q10, ); nepochs=200, batchsize=512, opt=Adam(0.01));

## legacy
# ? test lossfn
ps, st = LuxCore.setup(Random.default_rng(), RbQ10)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_p_f, ds_t = EasyHybrid.prepare_data(RbQ10, ds_keyed)
ds_t_nan = .!isnan.(ds_t)
ls = EasyHybrid.lossfn(RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss())

ls_logs = EasyHybrid.lossfn(RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss(train_mode=false))

# ? play with :Temp as predictors in NN, temperature sensitivity!
# TODO: variance effect due to LSTM vs NN
out = train(RbQ10, (ds_p_f, ds_t), (:Q10, ); nepochs=200, batchsize=512, opt=Adam(0.01));

output_file = joinpath(@__DIR__, "output_tmp/trained_model.jld2")
# o = jldopen(output_file, "r")
# close(o)

all_groups = get_all_groups(output_file)

predictions = load_group(output_file, :predictions)

physical_params, _ = load_group(output_file, :physical_params)

series(WrappedTuples(physical_params); axis=(; xlabel = "epoch", ylabel=""))

training_loss, _ = load_group(output_file, :training_loss)
series(WrappedTuples(WrappedTuples(training_loss).mse); axis=(; xlabel = "epoch", ylabel="training loss", xscale=log10, yscale=log10))

validation_loss, _ = load_group(output_file, :validation_loss)
series(WrappedTuples(WrappedTuples(validation_loss).mse); axis=(; xlabel = "epoch", ylabel="validation loss", xscale=log10, yscale=log10))


# load_group(output_file, "RespirationRbQ10")

## Plotting results
series(out.ps_history; axis=(; xlabel = "epoch", ylabel=""))

# with AoG
yvars = [:Rh]
xvars = Symbol.(string.(yvars) .* "_pred")

layers = visual(Scatter, alpha = 0.35)
plt = data(out.train_obs_pred) * layers * mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))
plt *= mapping(color = dims(1) => renamer(string.(xvars))=> "Metrics")
# linear
l_linear = linear() * visual(color=:grey25)
plt += data(out.train_obs_pred) * l_linear *  mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))

let
   draw(plt, scales(
        X = (; label = rich("Prediction", font=:bold)),
        Y = (; label = "Observation"),
        Color = (; palette = [:tomato, :teal, :orange, :dodgerblue3])
   ),
    legend = (; position=:right, titleposition=:top, merge=false),
    facet = (; linkxaxes = :none, linkyaxes = :none,),
) 
end



let
    fig = Figure(; size = (1200, 600))
    ax_train = Makie.Axis(fig[1, 1], title = "training")
    ax_val = Makie.Axis(fig[2, 1], title = "validation")
    lines!(ax_train, out.train_obs_pred[!, :Rh_pred], color=:orangered, label = "prediction")
    lines!(ax_train, out.train_obs_pred[!, :Rh], color=:dodgerblue, label ="observation")
    # validation
    lines!(ax_val, out.val_obs_pred[!, :Rh_pred], color=:orangered, label = "prediction")
    lines!(ax_val, out.val_obs_pred[!, :Rh], color=:dodgerblue, label ="observation")
    axislegend(; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 300))
    ax = Makie.Axis(fig[1,1], title = "Loss",
        yscale=log10, xscale=log10
        )
    lines!(ax, WrappedTuples(out.train_history.mse).sum, color=:orangered, label = "train")
    lines!(ax, WrappedTuples(out.val_history.mse).sum, color=:dodgerblue, label ="validation")
    # limits!(ax, 1, 1000, 0.04, 1)
    axislegend()
    fig
end


yobs_all =  ds_keyed(:Rh)

ŷ, RbQ10_st = LuxCore.apply(RbQ10, ds_p_f, out.ps, out.st)

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 600))
    ax_train = Makie.Axis(fig[1, 1], title = "full time series")
    lines!(ax_train, ŷ.Rh[:], color=:orangered, label = "prediction")
    lines!(ax_train, yobs_all[:], color=:dodgerblue, label ="observation")
    axislegend(ax_train; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end

# ? Rb
lines(out.αst_train.Rb[:])
lines!(ds_p_f(:Moist)[:])