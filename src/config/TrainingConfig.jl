@kwdef struct TrainConfig
    nepochs::Int = 200
    batchsize::Int = 64
    opt = Adam(0.01)
    patience::Int = typemax(Int)
    autodiff_backend = AutoZygote()
    return_gradients = True()
    training_loss::Symbol = :mse
    loss_types::Vector{Symbol} = [:mse, :r2]
    extra_loss = nothing
    agg::Function = sum
    train_from = nothing
    random_seed::Union{Int, Nothing} = 161803
    file_name::Union{String, Nothing} = nothing
    model_name::String = ""          # was hybrid_name
    return_model::Symbol = :best
    monitor_names::Vector = []
    output_folder::String = ""       # was folder_to_save
    plotting::Bool = true
    show_progress::Bool = true
    yscale = identity
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
