# CC BY-SA 4.0
# =============================================================================
# Setup and Data Loading
# =============================================================================
using Pkg
project_path = "projects/Respiration_Fluxnet"
Pkg.activate(project_path)

Pkg.develop(path=pwd())
Pkg.instantiate()

# start using the package
using EasyHybrid
using AxisKeys
using WGLMakie

include("Data/load_data.jl")

# =============================================================================
# Load data
# =============================================================================

# copy data to data/data20240123/ from here /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123
# or adjust the path to /Net/Groups/BGI/work_4/scratch/jnelson/4Sinikka/data20240123 + FluxNetSite

fluxnet_data = load_fluxnet_nc(joinpath(project_path, "Data", "data20240123", "US-SRG.nc"), timevar="date")

# explore data structure
println(names(fluxnet_data.timeseries))
println(fluxnet_data.scalars)
println(names(fluxnet_data.profiles))

# select timeseries data
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
# Artificial data
# =============================================================================
function fM(;SWC, K_limit, K_inhib)
    rm_SWC = SWC ./ (SWC .+ K_limit) .* K_inhib ./ (K_inhib .+ SWC)
    return rm_SWC
end

function fT(;TA, Tref, Q10)
    return Q10 .^ ((TA .- Tref) ./ 10.0)
end

# moisture limitation
df.rm_SWC = fM.(SWC = df.SWC_shallow, K_limit = 10, K_inhib = 30)
df.rm_T = fT.(TA = df.TA, Tref = 15, Q10 = 1.5)
df.RECO_syn = df.rm_T .* df.rm_SWC .* (1.0 .+ 0.1 .* randn(length(df.rm_T)))

# =============================================================================
# Figure 1: SWC and rate modifier
# =============================================================================
fig1 = Figure()
ax1_1 = Makie.Axis(fig1[1, 1], xlabel = "Time", ylabel = "Soil water content")
lines!(ax1_1, df.time, df.SWC_shallow, color = :dodgerblue)

ax1_2 = Makie.Axis(fig1[2, 1], xlabel = "Soil water content", ylabel = "Rate modifier due to soil water content")
scatter!(ax1_2, df.SWC_shallow, df.rm_SWC, color = :purple, markersize = 6, alpha = 0.6)

# =============================================================================
# Figure 2: Temperature and rate modifier
# =============================================================================
fig2 = Figure()
ax2_1 = Makie.Axis(fig2[1, 1], xlabel = "Time", ylabel = "Air temperature")
lines!(ax2_1, df.time, df.TA, color = :green, linewidth = 0.5)

ax2_2 = Makie.Axis(fig2[2, 1], xlabel = "Air temperature", ylabel = "Rate modifier due to air temperature")
scatter!(ax2_2, df.TA, df.rm_T, color = :green, markersize = 6, alpha = 0.6)

# =============================================================================
# Figure 3: RECO_syn time series
# =============================================================================
fig3 = Figure()
ax3_1 = Makie.Axis(fig3[1, 1], xlabel = "Time", ylabel = "RECO_syn", title = "RECO_syn")
lines!(ax3_1, df.time, df.RECO_syn, color = :brown, linewidth = 0.5)

# =============================================================================
# Figure 4: TA vs RECO_syn with colorbar
# =============================================================================

# RbQ10 function
function RbQ10(; Rb, Q10, TA, Tref)
    Rb .* Q10 .^ ((TA .- Tref) ./ 10.0)
end

function plot_RECO_vs_TA_with_SWC_and_RbQ10(df)
    fig = Figure(size = (900, 550))
    ax, sc = scatter(fig[1,1], df.TA, df.RECO_syn;
                  color = df.SWC_shallow, colormap = Reverse(:coolwarm),
                  axis = (xlabel = "TA", ylabel = "RECO_syn"),
                  markersize = 1)
    Colorbar(fig[1, 2], sc, label = "SWC_shallow")

    RECO_Rb1 = RbQ10.(Rb = 0.2, Q10 = 1.5, TA = df.TA, Tref = 15)
    RECO_Rb2 = RbQ10.(Rb = 0.4, Q10 = 1.5, TA = df.TA, Tref = 15)

    lines!(ax, df.TA, RECO_Rb1, color = :grey60, label = "Q10 = 1.5, Rb = 0.2")
    lines!(ax, df.TA, RECO_Rb2, color = :grey40, label = "Q10 = 1.5, Rb = 0.4")

    return fig, ax
end
fig4, ax4_1 = plot_RECO_vs_TA_with_SWC_and_RbQ10(df)
display(fig4)
axislegend(ax4_1, position = :lt)
# =============================================================================
# let's say we want to use the RbQ10 model but want to fit Rb and Q10 as constants
# =============================================================================

function RbQ10_syn(;Rb, Q10, TA, Tref = 15)
    RECO_syn = Rb .* Q10 .^ ((TA .- Tref) ./ 10.0)
    return (;RECO_syn, Q10, Rb)
end

# Parameter   Default   Lower   Upper
parameters = (
    Q10 = (2.0, 1.0, 4.0),   #   Q10 temperature sensitivity factor
    Rb  = (0.2, 0.0, 6.0)    #   Rb basal respiration
)

target = [:RECO_syn]
forcing = [:TA]
global_param_names = [:Q10, :Rb]
nn_param_names = []

mRb_constant = constructHybridModel(
    Vector{Symbol}(), # no predictors
    forcing,
    target,
    RbQ10_syn,
    parameters,
    [],
    global_param_names
)

# =============================================================================
# Model Training
# =============================================================================
out_a = train(mRb_constant, df, (); nepochs=100, batchsize=512, opt=RMSProp(0.001), 
                    loss_types=[:nse, :mse], training_loss=:nse, random_seed=123, 
                    yscale = identity, monitor_names=[:Q10, :Rb], patience = 30, 
                    shuffleobs = true, hybrid_name = "constant Rb")

Q10_a = out_a.train_diffs.Q10
Rb_a = out_a.train_diffs.Rb
RECO_syn_a = RbQ10_syn.(Rb = Rb_a, Q10 = Q10_a, TA = df.TA, Tref = 15).RECO_syn

fig4, ax4_1 = plot_RECO_vs_TA_with_SWC_and_RbQ10(df)
display(fig4)
lines!(ax4_1, df.TA, RECO_syn_a, color = :black, 
       label = "calibrated Q10 = $(round(Q10_a[1], digits = 2)), Rb = $(round(Rb_a[1], digits = 2))")
       axislegend(ax4_1, position = :lt)

# =============================================================================
# Figure 5: TA vs Rb
# =============================================================================
fig5 = Figure()
ax5_1 = Makie.Axis(fig5[1, 1], xlabel = "TA", ylabel = "Rb")
scatter!(ax5_1, df.TA, df.rm_SWC, markersize = 2, color = df.SWC_shallow, colormap = :viridis)

# =============================================================================
# RbQ10 Model with constant Q10 and variable Rb
# =============================================================================
target = [:RECO_syn]
forcing = [:TA]
predictors = [:SWC_shallow]
global_param_names = [:Q10]
nn_param_names = [:Rb]

mRb_varying = constructHybridModel(
    predictors,
    forcing,
    target,
    RbQ10_syn,
    parameters,
    nn_param_names,
    global_param_names
)

out_b = train(mRb_varying, df, (); nepochs=100, batchsize=512, opt=RMSProp(0.001), 
              loss_types=[:nse, :mse], training_loss=:nse, random_seed=123, 
              yscale = identity, monitor_names=[:Q10, :Rb], patience = 30, 
              shuffleobs = true, hybrid_name = "varying Rb")

Q10_b = out_b.train_diffs.Q10
Rb_b = out_b.train_diffs.Rb
RECO_syn_b = RbQ10_syn.(Rb = mean(Rb_b), Q10 = Q10_b, TA = df.TA, Tref = 15).RECO_syn

fig4, ax4_1 = plot_RECO_vs_TA_with_SWC_and_RbQ10(df)
display(fig4)
lines!(ax4_1, df.TA, RECO_syn_a, color = :black, 
       label = "'Unhybrid' Q10 = $(round(Q10_a[1], digits = 2)), constant Rb = $(round(Rb_a[1], digits = 2))")
lines!(ax4_1, df.TA, RECO_syn_b, color = :gold, 
       label = "Hybrid Q10 = $(round(Q10_b[1], digits = 2)), mean of varying NN-Rb = $(round(mean(Rb_b), digits = 2))")
axislegend(ax4_1, position = :lt)

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

predictors = (Rb = [:SWC_shallow, :P, :WS, :sine_dayofyear, :cos_dayofyear], 
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
out_Generic = train(hybrid_model, df, (); nepochs=1000, batchsize=512, opt=RMSProp(0.01), 
                    loss_types=[:nse, :mse], training_loss=:nse, random_seed=123, 
                    yscale = identity, monitor_names=[:RUE, :Q10], patience = 50, 
                    shuffleobs = true)

EasyHybrid.poplot(out_Generic)
