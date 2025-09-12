# CC BY-SA 4.0
import Pkg
project_path = @__DIR__
Pkg.activate(project_path)

using EasyHybrid
using AxisKeys
using CairoMakie

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
    fluxnet_data = load_fluxnet_nc(joinpath(data_dir, "$site.nc"); timevar="date")
    df = fluxnet_data.timeseries

    all_groups = get_all_groups(output_file)
    try
        if isfile(output_file)
            preds = load_group(output_file, :predictions)
            q10 = preds["training"].Q10[1]
            MAT = mean(skipmissing(df.TA))

            push!(dfQ10, (site = site, Q10 = q10, MAT = MAT))
        end
    catch e
        @warn "Failed to load predictions for site $site"
        continue
    end
end

using Random
include(joinpath(@__DIR__, "plotting.jl"))
fig = plot_Q10_vs_MAT(dfQ10, 3.5; k =7)
save(joinpath(main_output_folder, "00_Q10_vs_MAT.png"), fig)