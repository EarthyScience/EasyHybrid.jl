# CC BY-SA 4.0
import Pkg
project_path = @__DIR__
Pkg.activate(project_path)

using EasyHybrid
using AxisKeys
using CairoMakie
using TidierPlots

# Load helper(s)
include(joinpath(project_path, "Data", "load_data.jl"))
# data_dir = joinpath(project_path, "Data", "data20240123")
data_dir = "/Net/Groups/BGI/work_5/scratch/lalonso/data20240123"
s_names = sites_names(data_dir)

dfQ10 = DataFrame()  # initialize empty DataFrame
main_output_folder = "/Net/Groups/BGI/tscratch/lalonso/respiration_fluxnet"

for site in s_names
    output_file = joinpath(main_output_folder, "$(site)", "trained_model.jld2")

    if !isfile(output_file)
        @warn "Output file does not exist for site $site, skipping."
        continue
    end
    all_groups = get_all_groups(output_file)
    preds = load_group(output_file, :predictions)

    fluxnet_data = load_fluxnet_nc(joinpath(data_dir, "$site.nc"); timevar="date")
    df = fluxnet_data.timeseries

    q10 = preds["training"].Q10[1]
    MAT = mean(skipmissing(df.TA))

    # add a row to dfQ10
    push!(dfQ10, (site = site, Q10 = q10, MAT = MAT))
end

include(joinpath(@__DIR__, "plotting.jl"))
fig = plot_Q10_vs_MAT(dfQ10, 3.5)
save(joinpath(main_output_folder, "00_Q10_vs_MAT.png"), fig)

# ? what else?
beautiful_makie_theme = Attributes(fonts=(; regular="CMU Serif"))
Q10_vs_MAT = ggplot(dfQ10, aes(x=:MAT, y=:Q10)) + 
    geom_point() + 
    lims(y=(0, 7)) +
    geom_histogram(aes(x = :MAT), fill = "grey25") +
    beautiful_makie_theme

save(joinpath(main_output_folder, "01_Q10_vs_MAT.png"), Q10_vs_MAT)