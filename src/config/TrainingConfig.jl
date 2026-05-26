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

    """
    Optimizer to use for training. Default: `Adam(0.01)`.

    On the `Optimisers.jl` path (selected when `opt` originates from
    `Optimisers.jl`) three forms are accepted:

    1. A single `Optimisers.AbstractRule` (default), applied to the whole
       parameter tree, e.g. `Adam(0.01)`. `ps` is wrapped in a
       `ComponentArray`.
    2. A `NamedTuple` of rules — one per top-level branch of the parameter
       tree — e.g. for an `RbQ10` hybrid model:

       ```julia
       opt = (; Rb = Adam(1e-3), Q10 = Descent(1e-2))
       ```

       The framework calls `Optimisers.setup` per branch; branches not
       listed fall back to `Adam()`. `ps` is kept as a nested `NamedTuple`
       (no `ComponentArray` wrap) because `Optimisers.jl` treats a
       `ComponentArray` as a single leaf and would collapse the per-branch
       state tree.
    3. A `NamedTuple` of pre-built state trees (already returned by
       `Optimisers.setup`). Useful when one branch needs an
       `Optimisers.OptimiserChain` or a frozen leaf built up by hand.

    Forms 2 and 3 can be freely mixed in the same `NamedTuple`.
    """
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

    "Additional loss `(ŷ, ps; kwargs...) -> NamedTuple` added to the training loss. Default: `nothing`."
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

    """
    Whether to keep or not all training history, namely, all `epoch_history` entries. Default: `true`.
    """
    keep_history::Bool = true

    """
    Whether to save the training results to disk. Default: `true`.
     If `true`, the training history, model parameters, and diagnostics will be saved to a specified output path, allowing for later analysis and reproducibility.
     If `false`, the training results will not be saved to disk, and only the in-memory results will be available after training. This can be useful for quick experiments or when disk space is a concern, but it means that the training history and model parameters will not be preserved for future reference.
    """
    save_training::Bool = true

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

    """
    `Optimization.jl` path only: pass the full training set as a single tuple
    to `OptimizationProblem` instead of an `MLUtils.DataLoader`. Has no effect
    on the `Optimisers.jl` path. Default: `false`.
    """
    full_batch::Bool = false

    """
    `Optimization.jl` path only: promote `ps` to `Float64` before optimization.
    Mirrors the workaround documented in `projects/RbQ10/Q10_lbfgs.jl` for
    [Lux.jl#1260](https://github.com/LuxDL/Lux.jl/issues/1260). Has no effect
    on the `Optimisers.jl` path. Default: `false`.
    """
    promote_f64::Bool = false

    """
    `Optimization.jl` path only: build a validation `EpochSnapshot` (driving
    history / early-stopping / dashboard) every `eval_every` callback hits.
    Default: `1`.
    """
    eval_every::Int = 1
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

    "Best or all epoch snapshots taken throughout training, wrapped as a `WrappedTuples` collection, containing losses and predictions for both train and validation sets."
    epoch_history

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
