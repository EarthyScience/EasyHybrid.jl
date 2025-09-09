# CC BY-SA 4.0
# =============================================================================
# Setup (Distributed) and Environment
# =============================================================================
using Pkg
project_path = "projects/Respiration_Fluxnet"
Pkg.activate(project_path)
using Distributed, RetryManagers
# Add worker processes (respect your project env on each worker)
if nprocs() == 1
    cpus = parse(Int, get(ENV, "JULIA_NUM_THREADS", "1"))
    addprocs(RetryManager(max(1, cpus - 1)); exeflags=["--project=$(abspath(project_path))"])
end

@info "Workers:" workers()
#rmprocs(workers())
# Make paths & deps available everywhere
@everywhere begin
    using Pkg
    const PROJECT_ROOT = abspath($project_path)
    Pkg.activate(PROJECT_ROOT)
    # If needed on first run:
    # Pkg.instantiate()

    # Core deps for model + data
    using EasyHybrid
    using AxisKeys
    using CairoMakie

    # Load helper(s)
    include(joinpath(PROJECT_ROOT, "Data", "load_data.jl"))

    # =============================================================================
    # Mechanistic Model Definition (must exist on every process)
    # =============================================================================
    function flux_part_mechanistic_model(;SW_IN, TA, RUE, Rb, Q10)
        GPP  = SW_IN .* RUE ./ 12.011f0          # µmol/m²/s
        RECO = Rb .* Q10 .^ (0.1f0 .* (TA .- 15.0f0))
        NEE  = RECO .- GPP
        return (; NEE, RECO, GPP, Q10, RUE, Rb)
    end

    # =============================================================================
    # Parameter container and hybrid model constructor (shared constants)
    # =============================================================================
    parameters = (
        RUE = (0.1f0, 0.0f0, 2.0f0),
        Rb  = (1.0f0, 0.0f0, 20.0f0),
        Q10 = (1.5f0, 1.0f0, 5.0f0),
    )

    target_FluxPartModel   = [:NEE]
    forcing_FluxPartModel  = [:SW_IN, :TA]
    predictors = (
        Rb  = [:SWC_shallow, :P, :WS],
        RUE = [:TA, :P, :WS, :SWC_shallow, :VPD, :SW_IN_POT, :dSW_IN_POT, :dSW_IN_POT_DAY]
    )
    global_param_names = [:Q10]

    make_hybrid_model() = constructHybridModel(
        predictors,
        forcing_FluxPartModel,
        target_FluxPartModel,
        flux_part_mechanistic_model,
        parameters,
        global_param_names;
        scale_nn_outputs = true,
        hidden_layers    = [32, 32],
        activation       = sigmoid,
        input_batchnorm  = true,
        start_from_default = true,
    )

    # =============================================================================
    # Per-site training function (runs on workers via pmap)
    # =============================================================================
    function train_site(site::AbstractString, data_dir::AbstractString, main_output_folder::AbstractString)
        try
            fluxnet_data = load_fluxnet_nc(joinpath(data_dir, "$site.nc"); timevar="date")
            df = fluxnet_data.timeseries

            # Good-quality data only
            if :NEE_QC ∉ propertynames(df)
                return (site=String(site), trained=false, reason="NEE_QC column missing")
            end
            df = df[df.NEE_QC .== 0, :]

            # Sanity check: sufficient non-missing data
            predictor_cols = unique(vcat(values(predictors)...))
            forcing_cols   = forcing_FluxPartModel
            target_cols    = target_FluxPartModel
            needed_cols    = vcat(predictor_cols, forcing_cols, target_cols)

            for col in needed_cols
                if col ∉ propertynames(df)
                    return (site=String(site), trained=false, reason="Missing column $(col)")
                end
                n_nonmiss = sum(.!ismissing.(df[!, col]))
                if n_nonmiss < 1000
                    return (site=String(site), trained=false, reason="Insufficient data in $(col): $(n_nonmiss)")
                end
            end

            # Build model locally on the worker
            hybrid_model = make_hybrid_model()

            # Train (no plotting on workers)
            train(
                hybrid_model, df, ();
                nepochs        = 500,
                batchsize      = 512,
                opt            = RMSProp(0.01),
                loss_types     = [:nse, :mse],
                training_loss  = :nse,
                random_seed    = 123,
                yscale         = identity,
                monitor_names  = [:RUE, :Q10, :Rb],
                patience       = 30,
                shuffleobs     = true,
                plotting       = true,
                show_progress  = false,
                hybrid_name    = "",
                folder_to_save = joinpath(main_output_folder, "$(site)")
            )

            return (site=String(site), trained=true, out_folder=out_folder)
        catch err
            return (site=String(site), trained=false, reason=string(err))
        end
    end
end

# =============================================================================
# Data discovery (on master)
# =============================================================================
# Paths
const PROJECT_ROOT = abspath(project_path)
data_dir = joinpath(PROJECT_ROOT, "Data", "data20240123")

# All *.nc files → site names
nc_files = filter(f -> endswith(f, ".nc") && isfile(f), readdir(data_dir; join=true))
sites = first.(splitext.(basename.(nc_files)))

# Or use a fixed subset (like your original sample)
selected_sites = sites
selected_sites = sites[randperm(length(sites))[1:11]]

# =============================================================================
# Parallel training with pmap
# =============================================================================
using ProgressMeter
@info "Starting parallel training on $(length(selected_sites)) site(s)…"
main_output_folder = "sigmoid_act_no_dayofyear"
results = @showprogress dt=1 pmap(site -> train_site(site, data_dir, main_output_folder), selected_sites)

# =============================================================================
# Post-processing / forward run (on master)
# =============================================================================
using AxisKeys     # only for your exploration later

# Load one site’s best model (consistent path with training)
dfQ10 = DataFrame()  # initialize empty DataFrame

for site in selected_sites
    output_file = joinpath(PROJECT_ROOT, main_output_folder, "$(site)", "trained_model.jld2")

    @show output_file
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

include(joinpath(PROJECT_ROOT, "plotting.jl"))
fig = plot_Q10_vs_MAT(dfQ10, 3.5)
display(fig)

save(joinpath(PROJECT_ROOT, main_output_folder, "Q10_vs_MAT.png"), fig)

using TidierPlots
beautiful_makie_theme = Attributes(fonts=(; regular="CMU Serif"))
Q10_vs_MAT = ggplot(dfQ10, aes(x=:MAT, y=:Q10)) + 
    geom_point() + 
    lims(y=(0, 7)) +
    geom_histogram(aes(x = :MAT), fill = "grey25") +
    beautiful_makie_theme
if !isinteractive()
    savefig(Q10_vs_MAT, "Q10_vs_MAT.png")
end

rmprocs(workers())
if !isinteractive()
    exit()
end