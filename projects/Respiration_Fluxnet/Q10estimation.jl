using Pkg
Pkg.activate("projects/Respiration_Fluxnet")
Pkg.develop(path=pwd())
Pkg.instantiate()

include("Data/load_data.jl")

site = load_fluxnet_nc(joinpath(@__DIR__, "Data", "data20240123", "US-SRG.nc"), timevar="date")

println(names(site.timeseries))
site.scalars
println(names(site.profiles))

using GLMakie

GLMakie.activate!(inline=false)  # use non-inline (external) window for plots

# Create a figure and plot RECO_NT and RECO_DT time series
fig1 = Figure()

ax1 = fig1[1, 1] = Makie.Axis(fig1; ylabel = "RECO")
lines!(site.timeseries.time, site.timeseries.RECO_NT, label = "RECO_NT")
lines!(site.timeseries.time, site.timeseries.RECO_DT, label = "RECO_DT")
fig1[1, 2] = Legend(fig1, ax, framevisible = false)
hidexdecorations!()

fig1[2,1] = Makie.Axis(fig1; ylabel = "SWC")
lines!(site.timeseries.time, site.timeseries.SWC_shallow, label = "SWC_shallow")
hidexdecorations!()

fig1[3,1] = Makie.Axis(fig1; ylabel = "Precipitation")
lines!(site.timeseries.time, site.timeseries.P, label = "P")
hidexdecorations!()

ax4 = fig1[4,1] = Makie.Axis(fig1; xlabel = "Time", ylabel = "Temperature")
lines!(site.timeseries.time, site.timeseries.TA, label = "air")
lines!(site.timeseries.time, site.timeseries.TS_shallow, label = "soil")
fig1[4, 2] = Legend(fig1, ax4, framevisible = false)

linkxaxes!(filter(x -> x isa Axis, fig1.content)...)

fig1