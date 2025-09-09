# using Pkg

# # Absolute path to your project folder
# project_path = "/Users/atangarife/Documents/Models/EasyHybrid.jl/projects/Respiration_Fluxnet"
# Pkg.activate(project_path)

# Pkg.develop(path=pwd())
# Pkg.instantiate()

# using EasyHybrid
# using AxisKeys
# using WGLMakie


using Pkg

project_path = "/Users/atangarife/Documents/Models/EasyHybrid.jl/projects/Respiration_Fluxnet"
Pkg.activate(project_path)
Pkg.instantiate()

# Correctly include the file using the absolute path
 include(joinpath(project_path, "Data", "load_data.jl"))

# Now you can call load_fluxnet_nc
fluxnet_data = load_fluxnet_nc(joinpath(project_path, "Data", "data20240123", "MY-PSO.nc"), timevar="date")

# GH-Ank.nc

# =====

using NCDatasets

ncfile = joinpath(project_path, "Data", "data20240123", "MY-PSO.nc")
ds = Dataset(ncfile)

lat = coalesce.(Array(ds["tower_lat"]), NaN)
lon = coalesce.(Array(ds["tower_lon"]), NaN)

println("Latitude values: ", lat)
println("Longitude values: ", lon)

close(ds)



# =============================================================================
# Load data
# =============================================================================

# copy data to data/data20240123/ from here /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123
# or adjust the path to /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123 + FluxNetSite


# explore data structure
println(names(fluxnet_data.timeseries))
println(fluxnet_data.scalars)
println(names(fluxnet_data.profiles))

# select timeseries data
nc = Dataset(ncfile)   # NetCDF dataset
lat = coalesce.(Array(nc["tower_lat"]), NaN)
lon = coalesce.(Array(nc["tower_lon"]), NaN)
println("Latitude values: ", lat)
println("Longitude values: ", lon)
close(nc)

# Now use the DataFrame from EasyHybrid
df = fluxnet_data.timeseries


# Replace missings with NaN and convert all columns to Float64 where possible
for col in names(df)
    if eltype(df[!, col]) <: Union{Missing, Number}
        df[!, col] = coalesce.(df[!, col], NaN)
        try
            df[!, col] = Float64.(df[!, col])
        catch
            # If conversion fails, leave as is
        end
    end
end


# =============================================================================
# Mechanistic Model Definition
# =============================================================================
function flux_part_mechanistic_model(;SW_IN, TA, RUE, Rb, Q10)
    # -------------------------------------------------------------------------
    # Arguments:
    #   SW_IN     : Incoming shortwave radiation
    #   TA      : Air temperature
    #   RUE     : Radiation Use Efficiency
    #   Rb      : Basal respiration
    #   Q10     : Temperature sensitivity 
    #
    # Returns:
    #   NEE     : Net Ecosystem Exchange
    #   RECO    : Ecosystem respiration
    #   GPP     : Gross Primary Production
    # -------------------------------------------------------------------------

    # Calculate fluxes
    GPP = SW_IN .* RUE ./ 12.011f0  # µmol/m²/s
    RECO = Rb .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
    NEE = RECO .- GPP
    
    return (;NEE, RECO, GPP, Q10, RUE)
end

# =============================================================================
# Parameter container for the mechanistic model
# =============================================================================

# Define parameter structure with bounds
parameters = (
    #            default                  lower                     upper                description
    RUE      = ( 0.1f0,                  0.0f0,                   1.0f0 ),            # Radiation Use Efficiency [g/MJ]
    Rb       = ( 1.0f0,                  0.0f0,                   6.0f0 ),            # Basal respiration [μmol/m²/s]
    Q10      = ( 1.5f0,                  1.0f0,                   4.0f0 ),            # Temperature sensitivity factor [-]
)

# =============================================================================
# Hybrid Model Creation
# =============================================================================
target_FluxPartModel = [:NEE]
forcing_FluxPartModel = [:SW_IN, :TA]

predictors = (Rb = [:SWC_shallow, :P, :WS], 
              RUE = [:TA, :P, :WS, :SWC_shallow, :VPD, :SW_IN_POT, :dSW_IN_POT, :dSW_IN_POT_DAY])

global_param_names = [:Q10]

hybrid_model = constructHybridModel(
    predictors,
    forcing_FluxPartModel,
    target_FluxPartModel,
    flux_part_mechanistic_model,
    parameters,
    global_param_names,
    scale_nn_outputs=true,
    hidden_layers = [15, 15],
    activation = sigmoid,
    input_batchnorm = true,
    start_from_default = false
)

# =============================================================================
# Model Training
# =============================================================================
out_Generic = train(hybrid_model, df, (); nepochs=250, batchsize=512, opt=RMSProp(0.01), 
                    loss_types=[:nse, :mse], training_loss=:nse, random_seed=123, 
                    yscale = identity, monitor_names=[:RUE, :Q10], patience = 30, 
                    shuffleobs = true)

EasyHybrid.poplot(out_Generic)



# =============================================================================

# RECO_DT vs time 

# Check column names first
names(ds)
println(names(ds))


# Extract the time column (assuming it's called :date or :time)
time_col = :date   # change this if your time column has a different name


# Plot RECO_NT vs time and RECO_DT vs time
fig1 = Figure(resolution=(1000,400))
ax1 = Makie.Axis(fig1[1,1], title="RECO vs Time", xlabel="Time", ylabel="Flux")
l1 = lines!(ax1, ds[!, time_col], ds[!, :RECO_NT], color=:blue)
l2 = lines!(ax1, ds[!, time_col], ds[!, :RECO_DT], color=:red)
axislegend(ax1, [l1, l2], ["RECO_NT", "RECO_DT"], position=:rt)

# Plot GPP_DT vs time and GPP_NT vs time
fig2 = Figure(resolution=(1000,400))
ax2 = Makie.Axis(fig2[1,1], title="GPP vs Time", xlabel="Time", ylabel="Flux")
l3 = lines!(ax2, df[!, time_col], df[!, :GPP_NT], color=:green)
l4 = lines!(ax2, df[!, time_col], df[!, :GPP_DT], color=:black)
axislegend(ax2, [l3, l4], ["GPP_NT", "GPP_DT"], position=:rt)



####

using Pkg
using NCDatasets
using EasyHybrid
using AxisKeys
using WGLMakie

# -----------------------------
# Activate project
# -----------------------------
project_path = "/Users/atangarife/Documents/Models/EasyHybrid.jl/projects/Respiration_Fluxnet"
Pkg.activate(project_path)
Pkg.instantiate()

# -----------------------------
# Include data loading function
# -----------------------------
include(joinpath(project_path, "Data", "load_data.jl"))

# -----------------------------
# Load raw FluxNet NetCDF file
# -----------------------------
ncfile_path = joinpath(project_path, "Data", "data20240123", "CN-Cha.nc")
fluxnet_data = load_fluxnet_nc(ncfile_path, timevar="date")

# -----------------------------
# Read raw latitude and longitude values
# -----------------------------
nc = Dataset(ncfile_path)
lat = coalesce.(Array(nc["tower_lat"]), NaN)
lon = coalesce.(Array(nc["tower_lon"]), NaN)
println("Latitude values: ", lat)
println("Longitude values: ", lon)
close(nc)

# -----------------------------
# Explore processed data structure
# -----------------------------
df = fluxnet_data.timeseries
println("Timeseries column names: ", names(df))
println("Scalars: ", fluxnet_data.scalars)
println("Profiles: ", names(fluxnet_data.profiles))

#-----------------------------
#Replace missing values with NaN
#-----------------------------
for col in names(df)
    if eltype(df[!, col]) <: Union{Missing, Number}
        df[!, col] = coalesce.(df[!, col], NaN)
        try
            df[!, col] = Float64.(df[!, col])
        catch
            # leave as is if conversion fails
        end
    end
end

# -----------------------------
# Mechanistic model definition
# -----------------------------
# function flux_part_mechanistic_model(;SW_IN, TA, RUE, Rb, Q10)
#     GPP = SW_IN .* RUE ./ 12.011f0
#     RECO = Rb .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
#     NEE = RECO .- GPP
#     return (;NEE, RECO, GPP, Q10, RUE)
# end

# -----------------------------
# Parameter container
# -----------------------------
# parameters = (
#     RUE = (0.1f0, 0.0f0, 1.0f0),
#     Rb  = (1.0f0, 0.0f0, 6.0f0),
#     Q10 = (1.5f0, 1.0f0, 4.0f0),
# )

# -----------------------------
# Hybrid model creation
# -----------------------------
# target_FluxPartModel = [:NEE]
# forcing_FluxPartModel = [:SW_IN, :TA]

# predictors = (
#     Rb  = [:SWC_shallow, :P, :WS],
#     RUE = [:TA, :P, :WS, :SWC_shallow, :VPD, :SW_IN_POT, :dSW_IN_POT, :dSW_IN_POT_DAY]
# )

# global_param_names = [:Q10]

# hybrid_model = constructHybridModel(
#     predictors,
#     forcing_FluxPartModel,
#     target_FluxPartModel,
#     flux_part_mechanistic_model,
#     parameters,
#     global_param_names,
#     scale_nn_outputs=true,
#     hidden_layers=[15, 15],
#     activation=sigmoid,
#     input_batchnorm=true,
#     start_from_default=false
# )

# # -----------------------------
# # Model training
# # -----------------------------
# out_Generic = train(
#     hybrid_model, df;
#     nepochs=250,
#     batchsize=512,
#     opt=RMSProp(0.01),
#     loss_types=[:nse, :mse],
#     training_loss=:nse,
#     random_seed=123,
#     yscale=identity,
#     monitor_names=[:RUE, :Q10],
#     patience=30,
#     shuffleobs=true
# )

# EasyHybrid.poplot(out_Generic)

# -----------------------------
# Time series plotting
# -----------------------------
time_col = :date  # adjust if your time column has a different name

# RECO
fig1 = Figure(resolution=(1000,400))
ax1 = Makie.Axis(fig1[1,1], title="RECO vs Time", xlabel="Time", ylabel="Flux")
l1 = lines!(ax1, df[!, time_col], df[!, :RECO_NT], color=:blue)
l2 = lines!(ax1, df[!, time_col], df[!, :RECO_DT], color=:red)
axislegend(ax1, [l1, l2], ["RECO_NT", "RECO_DT"], position=:rt)

# GPP
fig2 = Figure(resolution=(1000,400))
ax2 = Makie.Axis(fig2[1,1], title="GPP vs Time", xlabel="Time", ylabel="Flux")
l3 = lines!(ax2, df[!, time_col], df[!, :GPP_NT], color=:green)
l4 = lines!(ax2, df[!, time_col], df[!, :GPP_DT], color=:black)
axislegend(ax2, [l3, l4], ["GPP_NT", "GPP_DT"], position=:rt)
