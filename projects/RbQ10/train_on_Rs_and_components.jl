# CC BY-SA 4.0
using Pkg
Pkg.activate("projects/RbQ10")
Pkg.develop(path=pwd())
Pkg.instantiate()

using EasyHybrid
using GLMakie
using Statistics

script_dir = @__DIR__
include(joinpath(script_dir, "data", "prec_process_data.jl"))

# Common data preprocessing
df = dfall[!, Not(:timesteps)]
ds_keyed = to_keyedArray(Float32.(df))

target_names = [:R_soil]
forcing_names = [:cham_temp_filled]

# =============================================================================
# Model 1: Simple Respiration Model (RbQ10)
# =============================================================================

println("Training RbQ10 model...")

# Define neural network
NN = Chain(Dense(2, 15, relu), Dense(15, 15, relu), Dense(15, 1));
# instantiate Hybrid Model
RbQ10 = RespirationRbQ10(NN, (:moisture_filled, :rgpot2), forcing_names, target_names, 2.5f0)
# train model
o_Rsonly = train(RbQ10, ds_keyed, (:Q10, ); nepochs=10, batchsize=512, opt=Adam(0.01), file_name = "o_Rsonly.jld2");

# Plot parameter history
series(o_Rsonly.ps_history; axis=(; xlabel = "epoch", ylabel=""))

include(joinpath(script_dir, "plotting.jl"))
plot_scatter(o_Rsonly, "train")

# Plot predictions vs observations
ŷ = RbQ10(ds_keyed, o_Rsonly.ps, o_Rsonly.st)[1]
yobs_all = ds_keyed(:R_soil)

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 600))
    ax_train = Makie.Axis(fig[1, 1], title = "full time series")
    lines!(ax_train, ŷ.R_soil[:], color=:orangered, label = "prediction")
    lines!(ax_train, yobs_all[:], color=:dodgerblue, label ="observation")
    axislegend(ax_train; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end

# =============================================================================
# Model 2: Respiration Components Model
# =============================================================================

println("Training Rs_components model...")

# Three respiration components
NN = Lux.Chain(Dense(2, 15, Lux.sigmoid), Dense(15, 15, Lux.sigmoid), Dense(15, 3, x -> x^2));
target_names = [:R_soil, :R_root, :R_myc, :R_het]
Rsc = Rs_components(NN, (:rgpot2, :moisture_filled), (:cham_temp_filled,), target_names, 2.5f0, 2.5f0, 2.5f0)

o_Rscomponents = train(Rsc, ds_keyed, (:Q10_het, :Q10_myc, :Q10_root, ); nepochs=10, batchsize=512, opt=Adam(0.01), file_name = "o_Rscomponents.jld2");

# Plot parameter history
series(o_Rscomponents.ps_history; axis=(; xlabel = "epoch", ylabel=""))

# Plot predictions vs observations
ŷ = Rsc(ds_keyed, o_Rscomponents.ps, o_Rscomponents.st)[1]
yobs_all = ds_keyed(:R_soil)

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 600))
    ax_train = Makie.Axis(fig[1, 1], title = "full time series")
    lines!(ax_train, ŷ.R_soil[:], color=:orangered, label = "prediction")
    lines!(ax_train, yobs_all[:], color=:dodgerblue, label ="observation")
    axislegend(ax_train; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end

# =============================================================================
# Model Comparison
# =============================================================================

println("Comparing models...")

# Create experimental dataset with constant conditions
ds_exp = ds_keyed
ds_exp(:rgpot2) .= mean(filter(!isnan, Array(ds_exp(:rgpot2))))
ds_exp(:moisture_filled) .= mean(filter(!isnan, Array(ds_exp(:moisture_filled))))

ŷ_Rsc = Rsc(ds_exp, o_Rscomponents.ps, o_Rscomponents.st)[1]
ŷ_RbQ10 = RbQ10(ds_exp, o_Rsonly.ps, o_Rsonly.st)[1]

# Plot temperature response comparison
fig = Figure(; size = (800, 600))
ax = Makie.Axis(fig[1, 1], title = "Temperature Response Comparison")

scatter!(ax, Array(ds_keyed(:cham_temp_filled)), vec(ŷ_Rsc.R_soil), color=:orangered, label="Rs_components")
scatter!(ax, Array(ds_keyed(:cham_temp_filled)), vec(ŷ_RbQ10.R_soil[1,:]), color=:dodgerblue, label="RbQ10")

axislegend(ax; position=:lt)
ax.xlabel = "Temperature"
ax.ylabel = "Soil respiration"

fig

using AlgebraOfGraphics
include(joinpath(script_dir, "plotting.jl"))
plot_scatter(o_Rscomponents, "train")