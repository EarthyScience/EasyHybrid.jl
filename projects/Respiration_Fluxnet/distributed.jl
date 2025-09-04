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
        return (; NEE, RECO, GPP, Q10, RUE)
    end

    # =============================================================================
    # Parameter container and hybrid model constructor (shared constants)
    # =============================================================================
    parameters = (
        RUE = (0.1f0, 0.0f0, 1.0f0),
        Rb  = (1.0f0, 0.0f0, 6.0f0),
        Q10 = (1.5f0, 1.0f0, 4.0f0),
    )

    target_FluxPartModel   = [:NEE]
    forcing_FluxPartModel  = [:SW_IN, :TA]
    predictors = (
        Rb  = [:SWC_shallow, :P, :WS],
        RUE = [:TA, :P, :WS, :SWC_shallow, :VPD, :SW_IN_POT, :dSW_IN_POT, :dSW_IN_POT_DAY],
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
        activation       = tanh,
        input_batchnorm  = true,
        start_from_default = false,
    )

    # =============================================================================
    # Per-site training function (runs on workers via pmap)
    # =============================================================================
    function train_site(site::AbstractString, data_dir::AbstractString, out_root::AbstractString)
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
                nepochs        = 100,
                batchsize      = 512,
                opt            = RMSProp(0.01),
                loss_types     = [:nse, :mse],
                training_loss  = :nse,
                random_seed    = 123,
                yscale         = identity,
                monitor_names  = [:RUE, :Q10],
                patience       = 50,
                shuffleobs     = true,
                plotting       = true,
                show_progress  = false,
                hybrid_name    = "",
                folder_to_save = "output_tmp_$(site)"
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

# =============================================================================
# Parallel training with pmap
# =============================================================================
@info "Starting parallel training on $(length(selected_sites)) site(s)…"
results = pmap(site -> train_site(site, data_dir, PROJECT_ROOT), selected_sites)

for r in results
    if r.trained
        @info "✓ Trained $(r.site) → $(r.out_folder)"
    else
        @warn "✗ Skipped $(r.site): $(get(r, :reason, "unknown reason"))"
    end
end

# =============================================================================
# Post-processing / forward run (on master)
# =============================================================================
using AxisKeys     # only for your exploration later
using CairoMakie
using TidierPlots

# Rebuild the model on master too
hybrid_model = make_hybrid_model()

# Load one site’s best model (consistent path with training)
site = "US-SRG"
output_file = joinpath(PROJECT_ROOT, "output_tmp_$(site)", "best_model.jld2")

all_groups = get_all_groups(output_file)
psst, _ = load_group(output_file, :HybridModel_MultiNNHybridModel)
ps_learned, st_learned = psst[end][1], psst[end][2]

# Prepare input dataframe for forward run
fluxnet_data = load_fluxnet_nc(joinpath(PROJECT_ROOT, "Data", "data20240123", "$site.nc"); timevar="date")
df = fluxnet_data.timeseries

forward_run = hybrid_model(df, ps_learned, st_learned)

forward_run.NEE_pred

# Quick diagnostics (plots are optional)
beautiful_makie_theme = Attributes(fonts=(; regular="CMU Serif"))
ggplot(forward_run, aes(x=:GPP_NT, y=:GPP_pred)) + geom_point() + beautiful_makie_theme

idx = .!isnan.(forward_run.GPP_NT) .& .!isnan.(forward_run.GPP_pred)
EasyHybrid.poplot(forward_run.GPP_NT[idx], forward_run.GPP_pred[idx], "GPP";
    xlabel = "Nighttime GPP", ylabel = "Hybrid GPP")
