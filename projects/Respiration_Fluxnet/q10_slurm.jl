# CC BY-SA 4.0
import Pkg
project_path = @__DIR__
Pkg.activate(project_path)
# Pkg.develop(path=pwd())
# Pkg.instantiate()

using EasyHybrid
using AxisKeys
using CairoMakie

# Load helper(s)
include(joinpath(project_path, "Data", "load_data.jl"))
# data_dir = joinpath(project_path, "Data", "data20240123")
data_dir = "/Net/Groups/BGI/work_5/scratch/lalonso/data20240123"
s_names = sites_names(data_dir)

slurm_array_id = Base.parse(Int, ARGS[1]) # get from command line argument
println("SLURM_ARRAY_ID = $slurm_array_id")

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

df, site, trained, reason = select_site(s_names[slurm_array_id+1], data_dir,  predictors, forcing_FluxPartModel, target_FluxPartModel)

if !trained
    println("Site $site not suitable for training: $reason")
    exit()
end

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

# Definition
hybrid_model = make_hybrid_model()

# TODO: training
# save things into `tscratch`, plently of space there!
main_output_folder = "/Net/Groups/BGI/tscratch/lalonso/respiration_fluxnet"

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