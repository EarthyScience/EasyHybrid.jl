using Revise
using EasyHybrid
using EasyHybrid.MLUtils

# ? move the `csv` file into the `BulkDSOC/data` folder (create folder)
df_o = CSV.read(joinpath(@__DIR__, "./data/lucas_overlaid.csv"), DataFrame, normalizenames=true)
println(size(df_o))

coords = collect(zip(df_o.lat, df_o.lon))

using Rasters

zarr_path = "/Net/Groups/BGI/scratch/HYCO/npp.zarr/"
data = Raster(zarr_path)

# t clean covariates
names_cov = Symbol.(names(df_o))[19:end]
names_meta = Symbol.(names(df_o))[1:18]

# Fix soilsuite and cropland extent columns
for col in names_cov
    if occursin("_soilsuite_", String(col))
        df_o[!, col] = replace(df_o[!, col], missing => 0)
    elseif occursin("cropland_extent_", String(col))
        df_o[!, col] = replace(df_o[!, col], missing => 0)
        df_o[!, col] .= ifelse.(df_o[!, col] .> 0, 1, 0)
    end
end

# rm missing values: 1. >5%, drop col; 2. <=5%, drop row
cols_to_drop_row = Symbol[]
cols_to_drop_col = Symbol[] 
for col in names_cov
    n_missing = count(ismissing, df_o[!, col])
    frac_missing = n_missing / nrow(df_o)
    if frac_missing > 0.05
        println(n_missing, " ", col)
        select!(df_o, Not(col))  # drop the column
        push!(cols_to_drop_col, col)  
    elseif n_missing > 0
        # println(n_missing, " ", col)
        push!(cols_to_drop_row, col)  # collect column name
    end

    if occursin("CHELSA_kg", String(col)) 
        push!(cols_to_drop_col, col) 
        select!(df_o, Not(col))  # rm kg catagorical col
    end 
end

names_cov = filter(x -> !(x in cols_to_drop_col), names_cov) # remove cols-to-drop from names_cov
if !isempty(cols_to_drop_row) 
    df_o = subset(df_o, cols_to_drop_row .=> ByRow(!ismissing)) # drop rows with missing values in cols_to_drop_row
end
println(size(df_o))

cols_to_drop_col = Symbol[] 
for col in names_cov
    if std(df_o[:,col])==0
        push!(cols_to_drop_col, col)  # rm constant col (std==0)
        select!(df_o, Not(col))
    end
end
names_cov = filter(x -> !(x in cols_to_drop_col), names_cov) # remove cols-to-drop from names_cov
println(size(df_o))


df = df_o[:, [:bulk_density_fe, :soc, :coarse_vol, names_cov...]]

# ? match target_names
rename!(df, :bulk_density_fe => :BD, :soc => :SOCconc, :coarse_vol => :CF) # rename as in hybrid model

df[:, :SOCconc] .= df[:, :SOCconc] ./ 1000 # convert to fraction

# ? calculate SOC density
df[!,:SOCdensity] = df.BD .* df.SOCconc .* (1 .- df.CF) # TODO: check units
target_names = [:BD, :SOCconc, :CF, :SOCdensity]
# df[:, target_names] = replace.(df[:, target_names], missing => NaN) # replace missing with NaN

# for col in names_cov # to check covairate distribution
#     println(string(col)[1:10], ' ', round(std(df[:, col]); digits=2), ' ', round(mean(df[:, col]); digits=2))
# end

# # Normalize covariates with std>1
means = mean.(eachcol(df[:, names_cov]))
stds = std.(eachcol(df[:, names_cov]))
for col in names_cov
    df[!, col] = Float64.(df[!, col])
end
df[:, names_cov] .= (df[:, names_cov] .- means') ./ stds'

println(size(df))

CSV.write(joinpath(@__DIR__, "data/lucas_preprocessed.csv"), df)