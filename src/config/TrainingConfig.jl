export TrainResults, TrainConfig, validate_config

"""
Configuration for training a hybrid model.

Controls all aspects of the training process including optimization,
loss computation, data handling, output, and visualization.
"""
@kwdef struct TrainConfig
    "Number of training epochs. Default: 200."
    nepochs::Int = 200

    "Size of the training batches. Default: 64."
    batchsize::Int = 64

    "Optimizer to use for training. Default: Adam(0.01)."
    opt = Adam(0.01)

    """
    Number of epochs to wait before early stopping.
    Default: `typemax(Int)` (no early stopping).
    """
    patience::Int = typemax(Int)

    "Automatic differentiation backend. Default: AutoZygote()."
    autodiff_backend = AutoZygote()

    "Whether to return gradients during training. Default: True()."
    return_gradients = True()

    "Select a gpu_device or default to cpu if none available"
    gdev = gpu_device()

    "Set the `cpu_device`, useful for sending back to the cpu model parameters"
    cdev = cpu_device()

    "Loss type to use during training. Default: `:mse`."
    training_loss::Symbol = :mse

    """
    Vector of loss types to compute during training. Default: `[:mse, :r2]`.
    The first entry is used for plotting in the dynamic trainboard and can be
    increasing (e.g. NSE) or decreasing (e.g. RMSE).
    """
    loss_types::Vector{Symbol} = [:mse, :r2]

    "Additional loss term to add to the training loss. Default: `nothing`."
    extra_loss = nothing

    "Aggregation function applied to computed losses. Default: `sum`."
    agg::Function = sum

    """
    Starting point for training: a tuple of physical parameters and state,
    or an output of `train`. Default: `nothing` (new training).
    """
    train_from = nothing

    "Random seed for reproducibility. Default: 161803."
    random_seed::Union{Int, Nothing} = 161803

    "Name identifier for the hybrid model. Default: empty string."
    model_name::String = ""

    """
    Which model to return after training: `:best` for the best-performing
    checkpoint, `:final` for the last epoch. Default: `:best`.
    """
    return_model::Symbol = :best

    "Vector of monitor names to track during training. Default: `[]`."
    monitor_names::Vector = []

    "Additional folder name string appended to the output path. Default: empty string."
    output_folder::String = ""

    "Whether to generate plots during training. Default: `true`."
    plotting::Bool = true

    "Whether to show progress bars during training. Default: `true`."
    show_progress::Bool = true

    "Scale applied to the y-axis for plotting. Default: `identity`."
    yscale = identity

    "Tuple of parameter names to track across epochs. Default: `()`."
    tracked_params::Tuple = ()
end

function validate_config(cfg::TrainConfig)
    cfg.return_model in (:best, :final) ||
        throw(ArgumentError("return_model must be :best or :final, got :$(cfg.return_model)"))

    cfg.batchsize > 0 ||
        throw(ArgumentError("batchsize must be positive, got $(cfg.batchsize)"))

    cfg.nepochs > 0 ||
        throw(ArgumentError("nepochs must be positive, got $(cfg.nepochs)"))

    cfg.patience > 0 ||
        throw(ArgumentError("patience must be positive, got $(cfg.patience)"))

    check_training_loss(cfg.training_loss) # TODO: revisit implementation

    return cfg.training_loss in cfg.loss_types ||
        @warn "training_loss :$(cfg.training_loss) is not in loss_types $(cfg.loss_types), it won't appear in plots"
end

"""
Output of [`train`](@ref), containing the full training history, model state, and diagnostics.
"""
struct TrainResults
    "Per-epoch training losses, wrapped as a `WrappedTuples` collection."
    train_history

    "Per-epoch validation losses, wrapped as a `WrappedTuples` collection."
    val_history

    "Model parameter snapshots taken throughout training, wrapped as a `WrappedTuples` collection."
    ps_history

    "Observed vs. predicted values on the training set."
    train_obs_pred

    "Observed vs. predicted values on the validation set."
    val_obs_pred

    "Additional diagnostic variables computed on the training set."
    train_diffs

    "Additional diagnostic variables computed on the validation set."
    val_diffs

    "Final or best model parameters, depending on `train_cfg.return_model`."
    ps

    "Final or best model state, depending on `train_cfg.return_model`."
    st

    "Epoch at which the best validation loss was achieved."
    best_epoch

    "Best validation loss recorded during training."
    best_loss
end
