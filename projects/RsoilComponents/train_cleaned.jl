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

# Define neural network
NN = Chain(Dense(2, 15, relu), Dense(15, 15, relu), Dense(15, 1));
# instantiate Hybrid Model
RbQ10 = RespirationRbQ10(NN, (:moisture_filled, :rgpot2), target_names, forcing_names, 2.5f0) # ? do different initial Q10s
# train model
o_Rsonly = train(RbQ10, ds_keyed, (:Q10, ); nepochs=2000, batchsize=512, opt=Adam(0.01), file_name = "o_Rsonly.jld2");

series(o_Rsonly.ps_history; axis=(; xlabel = "epoch", ylabel=""))

ŷ = RbQ10(ds_keyed, o_Rsonly.ps, o_Rsonly.st)[1]
yobs_all =  ds_keyed(:R_soil)

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 600))
    ax_train = Makie.Axis(fig[1, 1], title = "full time series")
    lines!(ax_train, ŷ.R_soil[:], color=:orangered, label = "prediction")
    lines!(ax_train, yobs_all[:], color=:dodgerblue, label ="observation")
    axislegend(ax_train; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end


# Three respiration components
NN = Lux.Chain(Dense(2, 15, Lux.sigmoid), Dense(15, 15, Lux.sigmoid), Dense(15, 3, x -> x^2));
target_names = [:R_soil, :R_root, :R_myc, :R_het]
Rsc = Rs_components(NN, (:rgpot, :moisture_filled), target_names, (:cham_temp_filled,), 2.5f0, 2.5f0, 2.5f0)

o_Rscomponents = train(Rsc, ds_keyed, (:Q10_het, :Q10_myc, :Q10_root, ); nepochs=1000, batchsize=512, opt=Adam(0.01), file_name = "o_Rscomponents.jld2");

series(o_Rscomponents.ps_history; axis=(; xlabel = "epoch", ylabel=""))

ŷ = Rsc(ds_keyed, o_Rscomponents.ps, o_Rscomponents.st)[1]
yobs_all =  ds_keyed(:R_soil)

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 600))
    ax_train = Makie.Axis(fig[1, 1], title = "full time series")
    lines!(ax_train, ŷ.R_soil[:], color=:orangered, label = "prediction")
    lines!(ax_train, yobs_all[:], color=:dodgerblue, label ="observation")
    axislegend(ax_train; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end



ds_exp = ds_keyed

ds_exp(:rgpot) .= mean(filter(!isnan, Array(ds_exp(:rgpot))))
ds_exp(:moisture_filled) .= mean(filter(!isnan, Array(ds_exp(:moisture_filled))))

ŷ = Rsc(ds_exp, o_Rscomponents.ps, o_Rscomponents.st)[1]
ŷ2 = RbQ10(ds_exp, o_Rsonly.ps, o_Rsonly.st)[1]

plot(ds_keyed(:cham_temp_filled), ŷ.R_soil)
plot!(ds_keyed(:cham_temp_filled), ŷ2.R_soil[1,:])

output_file = joinpath(@__DIR__, "output_tmp/o_Rsonly.jld2")
# o = jldopen(output_file, "r")
# close(o)

all_groups = get_all_groups(output_file)

predictions = load_group(output_file, :predictions)

physical_params, _ = load_group(output_file, :physical_params)

series(WrappedTuples(physical_params); axis=(; xlabel = "epoch", ylabel=""))

training_loss, _ = load_group(output_file, :training_loss)
series(WrappedTuples(WrappedTuples(training_loss).r2); axis=(; xlabel = "epoch", ylabel="training loss"))

validation_loss, _ = load_group(output_file, :validation_loss)
series(WrappedTuples(WrappedTuples(validation_loss).r2); axis=(; xlabel = "epoch", ylabel="validation loss"))
