# CC BY-SA 4.0
using Pkg
Pkg.activate("projects/RbQ10")
Pkg.develop(path=pwd())
Pkg.instantiate()

using CSV, DataFrames
using Dates
using GLMakie
using AlgebraOfGraphics
# include("../legacy/src/g_pot.jl")

df_forcing = CSV.read("/Users/lalonso/Documents/HybridML/data/Rh_AliceHolt_forcing_filled.csv", DataFrame)

let 
   lines(df_forcing[!, :Temp] .- 273.15)
    lines!(df_forcing[!, :Moist])
    lines!(df_forcing[!, :Rgpot])
    current_figure() 
end

with_theme(theme_light()) do
    labels = names(df_forcing)
    colors = resample_cmap(:mk_8, 8)
    fig = Figure(; size = (1200, 600))
    axs = [Axis(fig[j, 1]) for j in 1:4]
    [lines!(axs[j], df_forcing[!, j]; color = colors[j], label = labels[j], linewidth=0.65) for j in 1:4]
    axislegend.(axs; framevisible=true,  backgroundcolor=:grey95, framecolor=:white)
    fig
end
