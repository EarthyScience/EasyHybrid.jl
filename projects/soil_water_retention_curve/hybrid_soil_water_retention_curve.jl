using Pkg
Pkg.activate("projects/soil_water_retention_curve")
Pkg.develop(path=pwd())
Pkg.instantiate()

using Revise

using EasyHybrid
using GLMakie
using Statistics

# ? move the `csv` file into the `BulkDSOC/data` folder (create folder)
@__DIR__
df_o = CSV.read(joinpath(@__DIR__, "./data/Norouzi_et_al_2024_WRR_Final.csv"), DataFrame, normalizenames=true)

df = copy(df_o)
df.h = 10 .^ df.pF

# Rename :WC to :θ in the DataFrame
rename!(df, :WC => :θ)

ds_keyed = to_keyedArray(Float32.(df))

ds_keyed(:θ)
ds_keyed(:h)

targets = [:θ]
forcing = [:h]
predictors = [:sand, :clay, :silt, :BD, :OC]

# =============================================================================
# Model 1: Simple Respiration Model (RbQ10)
# =============================================================================

println("Training Hybrid Soil Water Retention Curve Model")

# Define neural network: outputs θ_s, α, n, m
NN = Chain(Dense(5, 15, relu), Dense(15, 15, relu), Dense(15, 4))

# Set initial guesses for global parameters
h_r_init = 1.0f0
h_0_init = 1f6

# Instantiate Hybrid Model
hybridSWRC = WaterRetentionHybrid(NN, predictors, forcing, targets, h_r_init, h_0_init)

dta = EasyHybrid.prepare_data(hybridSWRC, ds_keyed)
ps, st = LuxCore.setup(Random.default_rng(), hybridSWRC)

ds_p_f, ds_t = EasyHybrid.prepare_data(hybridSWRC, ds_keyed)
ds_t_nan = .!isnan.(ds_t)

ls = EasyHybrid.lossfn(hybridSWRC, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss())

out = hybridSWRC(ds_keyed, ps, st)

# Train model (h_r and h_0 are global parameters to be estimated)o_SWRC = train(hybridSWRC, ds_keyed, (:h_r, :h_0, ); nepochs=10, batchsize=512, opt=Adam(0.01), file_name = "o_SWRC.jld2")
o_SWRC = train(hybridSWRC, ds_keyed, (:h_r, :h_0, ); nepochs=10, batchsize=512, opt=Adam(0.01), file_name = "o_SWRC.jld2")

# Plot parameter history
series(o_SWRC.ps_history; axis=(; xlabel = "epoch", ylabel=""))

include(joinpath(script_dir, "plotting.jl"))
plot_scatter(o_SWRC, "train")

# Plot predictions vs observations
ŷ = hybridSWRC(ds_keyed, o_SWRC.ps, o_SWRC.st)[1][:θ]
yobs_all = ds_keyed(:WC)

with_theme(theme_light()) do 
    fig = Figure(; size = (1200, 600))
    ax_train = Makie.Axis(fig[1, 1], title = "full time series")
    lines!(ax_train, ŷ[:], color=:orangered, label = "prediction")
    lines!(ax_train, yobs_all[:], color=:dodgerblue, label ="observation")
    axislegend(ax_train; position=:lt)
    Label(fig[0,1], "Observations vs predictions", tellwidth=false)
    fig
end
