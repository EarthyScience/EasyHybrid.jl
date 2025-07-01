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
GLMakie.activate!(inline=true) # for plots in vscode true, separate window false


# For plotting when GLMakie is not available or not opening new panels
#import Pkg
#Pkg.add("CairoMakie")
#using CairoMakie

# load data
df_o = CSV.read(joinpath(@__DIR__, "./data/Rh_AliceHolt_forcing_filled.csv"), DataFrame)



# some pre-processing
df = copy(df_o)
df[!, :Temp] = df[!, :Temp] .- 273.15 # convert to Celsius
# df = filter(:Respiration_heterotrophic => !isnan, df)
rename!(df, :Respiration_heterotrophic => :Rh)  # rename as in hybrid model

names(df) #check if the names are correct and complete 

ds_keyed = to_keyedArray(Float32.(df)) # predictors + forcing

# Define neural network
NN = Chain(Dense(1, 15, sigmoid), Dense(15, 15, sigmoid), Dense(15, 1, x -> x^2))

# instantiate Hybrid Model
# RbQ10 = RespirationRbQ10(NN, (:Rgpot, :Moist), (:Rh, ), (:Temp,), 2.5f0) # ? do different initial Q10s
# train model
# out = train(RbQ10, ds_keyed, (:Q10, ); nepochs=200, batchsize=512, opt=Adam(0.01));

# Model with DAMM equation from script "Respiration_DAMM"
RDAMM = RespirationDAMM(NN, (:Rgpot,), (:Rh, ), (:Temp, :Moist), 2.5f0, 9.97f-7, 1.21f0 ) # ? do different initial Q10s and kmo and kms. f0 takes values as Float32. The f-7 is simply scientific notation (like e-7) 
ps, st = LuxCore.setup(Random.default_rng(), RDAMM)  
hout = RDAMM(ds_keyed, ps, st)[1]


# put an Axis into cell (1,1), it becomes the “current axis”
fig1 = Figure()
fig1[1, 1] = Makie.Axis(fig1; xlabel = "Time", title = "Initial RDAMM Output")

# now all of these plot calls go to that axis implicitly
lines!(vec(hout.Rh),           label = "Rh (predicted)")
lines!(vec(hout.Q10term),      label = "Q10 term")
lines!(vec(hout.Rb),           label = "Rb (base respiration)")
lines!(vec(hout.S_limitation), label = "Substrate limitation")
lines!(vec(hout.Ox_limitation), label = "Oxygen limitation")

# observed
lines!(vec(ds_t(:Rh)),         label = "Rh (observed)", linewidth = 2)

axislegend(position = :rt)     # also applies to the current axis

fig1  # display


# train model
out = train(RDAMM, ds_keyed, (:Q10, :kmo, :kms); nepochs=200, batchsize=512, opt=Adam(0.01)); 

tout = RDAMM(ds_keyed, out.ps, out.st)[1]

# put an Axis into cell (1,1), it becomes the “current axis”
fig1 = Figure()
fig1[1, 1] = Makie.Axis(fig1; xlabel = "Time", title = "Initial RDAMM Output")

# now all of these plot calls go to that axis implicitly
lines!(vec(tout.Rh),           label = "Rh (predicted)")
lines!(vec(tout.Q10term),      label = "Q10 term")
lines!(vec(tout.Rb),           label = "Rb (base respiration)")
lines!(vec(tout.S_limitation), label = "Substrate limitation")
lines!(vec(tout.Ox_limitation), label = "Oxygen limitation")

# observed
lines!(vec(ds_t(:Rh)),         label = "Rh (observed)", linewidth = 2)

axislegend(position = :rt)     # also applies to the current axis

fig1  # display


## legacy for RBQ10 model 
            # ? test lossfn
            #=
            ps, st = LuxCore.setup(Random.default_rng(), RbQ10)
            # the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
            ds_p_f, ds_t = EasyHybrid.prepare_data(RbQ10, ds_keyed)
            ds_t_nan = .!isnan.(ds_t)
            ls = EasyHybrid.lossfn(RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss())
            ls_logs = EasyHybrid.lossfn(RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss(train_mode=false))
            # ? play with :Temp as predictors in NN, temperature sensitivity!
            # TODO: variance effect due to LSTM vs NN
            out = train(RbQ10, (ds_p_f, ds_t), (:Q10, ); nepochs=200, batchsize=512, opt=Adam(0.01));
            =#
output_file = joinpath(@__DIR__, "output_tmp/trained_model.jld2")

# o = jldopen(output_file, "r")
# close(o)

all_groups = get_all_groups(output_file)
predictions = load_group(output_file, :predictions)
physical_params, _ = load_group(output_file, :physical_params)


# Plots Loss over epochs for training and validation datasets. 

training_loss, _ = load_group(output_file, :training_loss)
series(WrappedTuples(WrappedTuples(training_loss).mse); axis=(; xlabel = "epoch", ylabel="training loss", xscale=log10, yscale=log10))

validation_loss, _ = load_group(output_file, :validation_loss)
series(WrappedTuples(WrappedTuples(validation_loss).mse); axis=(; xlabel = "epoch", ylabel="validation loss", xscale=log10, yscale=log10))


# load_group(output_file, "RespirationRbQ10")

## Plotting results---  redundant with the below  
#series(out.ps_history; axis=(; xlabel = "epoch", ylabel=""))

# Plot for Q10 value as the training advances 
series(WrappedTuples(physical_params); axis=(; xlabel = "epoch", ylabel=""))

# AoG plotting

# Convert all columns to Float64 as they are mixed of Float32 and Float64 and present problems to plot 
df_plot = deepcopy(out.train_obs_pred)
for c in names(df_plot)
    if eltype(df_plot[!, c]) <: AbstractFloat
        df_plot[!, c] = Float64.(df_plot[!, c])
    end
end
   
yvars = [:Rh]
xvars = Symbol.(string.(yvars) .* "_pred")
layers = visual(Scatter, alpha = 0.35)
plt = data(df_plot) * layers * mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))
plt *= mapping(color = dims(1) => renamer(string.(xvars)) => "Metrics")
# Add regression line 
l_linear = linear() * visual(color = :grey25)
plt += data(df_plot) * l_linear * mapping(xvars, yvars, col=dims(1) => renamer(string.(yvars)))
display(draw(plt))

"""
# Same plot as AoG but with editable scales and legend 
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
"""

# Plotting observations vs predictions 
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

# Plotting Training and Validation Loss 
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