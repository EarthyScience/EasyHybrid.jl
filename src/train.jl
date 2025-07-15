export train

"""
    train(hybridModel, data, save_ps; nepochs=200, batchsize=10, opt=Adam(0.01), file_name=nothing, loss_types=[:mse, :mae], training_loss=:mse, agg=sum)

Train a hybrid model using the provided data and save the training process to a file in JLD2 format. Default output file is `trained_model.jld2` at the current working directory under `output_tmp`.

# Arguments:
- `hybridModel`: The hybrid model to be trained.
- `data`: The training data, either a tuple of KeyedArrays or a single KeyedArray.
- `save_ps`: A tuple of physical parameters to save during training.
- `nepochs`: Number of training epochs (default: 200).
- `batchsize`: Size of the training batches (default: 10).
- `opt`: The optimizer to use for training (default: Adam(0.01)).
- `file_name`: The name of the file to save the training process (default: nothing-> "trained_model.jld2").
- `loss_types`: A vector of loss types to compute during training (default: `[:mse, :mae]`).
- `training_loss`: The loss type to use during training (default: `:mse`).
- `agg`: The aggregation function to apply to the computed losses (default: `sum`).
"""
function train(hybridModel, data, save_ps; nepochs=200, batchsize=10, opt=Adam(0.01),
    file_name=nothing, loss_types=[:mse, :r2], training_loss=:mse, agg=sum)
    data_ = prepare_data(hybridModel, data)
    # all the KeyedArray thing!

    # ? split training and validation data
    (x_train, y_train), (x_val, y_val) = splitobs(data_; at=0.8, shuffle=false)
    train_loader = DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true);
    # ? setup model
    ps, st = LuxCore.setup(Random.default_rng(), hybridModel)
    opt_state = Optimisers.setup(opt, ps)

    # ? initial losses
    is_no_nan_t = .!isnan.(y_train)
    is_no_nan_v = .!isnan.(y_val)
    l_init_train = lossfn(hybridModel, x_train, (y_train, is_no_nan_t), ps, LuxCore.testmode(st),
        LoggingLoss(train_mode=false, loss_types=loss_types, training_loss=training_loss, agg=agg))[1]
    l_init_val = lossfn(hybridModel, x_val, (y_val, is_no_nan_v), ps, LuxCore.testmode(st),
        LoggingLoss(train_mode=false, loss_types=loss_types, training_loss=training_loss, agg=agg))[1]

    train_history = [l_init_train]
    val_history = [l_init_val]
    # track physical parameters
    ps_values_init = [copy(getproperty(ps, e)[1]) for e in save_ps]
    ps_init = NamedTuple{save_ps}(ps_values_init)
    ps_history = [ps_init]
    
    file_name = resolve_path(file_name)
    save_ps_st(file_name, hybridModel, ps, st, save_ps)
    save_train_val_loss!(file_name,l_init_train, "training_loss", 0)
    save_train_val_loss!(file_name,l_init_val, "validation_loss", 0)

    prog = Progress(nepochs, desc="Training loss")
    for epoch in 1:nepochs
        for (x, y) in train_loader
            # ? check NaN indices before going forward, and pass filtered `x, y`.
            is_no_nan = .!isnan.(y)
            if length(is_no_nan)>0 # ! be careful here, multivariate needs fine tuning
                l, backtrace = Zygote.pullback((ps) -> lossfn(hybridModel, x, (y, is_no_nan), ps, st,
                    LoggingLoss(training_loss=training_loss, agg=agg)), ps)
                grads = backtrace(l)[1]
                Optimisers.update!(opt_state, ps, grads)
                st =(; l[2].st...)
            end
        end
        save_ps_st!(file_name, hybridModel, ps, st, save_ps, epoch)

        ps_values = [copy(getproperty(ps, e)[1]) for e in save_ps]
        tmp_e = NamedTuple{save_ps}(ps_values)
        push!(ps_history, tmp_e)

        l_train = lossfn(hybridModel, x_train,  (y_train, is_no_nan_t), ps, LuxCore.testmode(st),
            LoggingLoss(train_mode=false, loss_types=loss_types, training_loss=training_loss, agg=agg))[1]
        l_val = lossfn(hybridModel, x_val, (y_val, is_no_nan_v), ps, LuxCore.testmode(st),
            LoggingLoss(train_mode=false, loss_types=loss_types, training_loss=training_loss, agg=agg))[1]
        save_train_val_loss!(file_name, l_train, "training_loss", epoch)
        save_train_val_loss!(file_name, l_val, "validation_loss", epoch)
        
        push!(train_history, l_train)
        push!(val_history, l_val)

        _headers, paddings = header_and_paddings(getproperty(l_init_train, training_loss))

        next!(prog; showvalues = [
            ("epoch ", epoch),
            ("targets ", join(_headers, "  ")),
            (styled"{red:training-start }", styled_values(getproperty(l_init_train, training_loss); paddings)),
            (styled"{bright_red:current }", styled_values(getproperty(l_train, training_loss); color=:bright_red, paddings)),
            (styled"{cyan:validation-start }", styled_values(getproperty(l_init_val, training_loss); paddings)),
            (styled"{bright_cyan:current }", styled_values(getproperty(l_val, training_loss); color=:bright_cyan, paddings)),
            ]
            )
            # TODO: log metrics
    end

    train_history = WrappedTuples(train_history)
    val_history = WrappedTuples(val_history)
    ps_history = WrappedTuples(ps_history)

    # ? save final evaluation or best at best validation value

    ŷ_train, αst_train = hybridModel(x_train, ps, LuxCore.testmode(st))
    ŷ_val, αst_val = hybridModel(x_val, ps, LuxCore.testmode(st))
    save_predictions!(file_name, ŷ_train, αst_train, "training")
    save_predictions!(file_name, ŷ_val, αst_val, "validation")

    # training
    target_names = hybridModel.targets
    save_observations!(file_name, target_names, y_train, "training")
    save_observations!(file_name, target_names, y_val, "validation")
    # save split obs (targets)

    # ? this could be saved to disk if needed for big sizes.
    train_obs = toDataFrame(y_train)
    train_hats = toDataFrame(ŷ_train, target_names)
    train_obs_pred = hcat(train_obs, train_hats)
    # validation
    val_obs = toDataFrame(y_val)
    val_hats = toDataFrame(ŷ_val, target_names)
    val_obs_pred = hcat(val_obs, val_hats)
    # ? diffs, additional predictions without observational counterparts!
    # TODO: better!
    set_diff = setdiff(keys(ŷ_train), target_names)
    train_diffs = !isempty(set_diff) ? NamedTuple{Tuple(set_diff)}([getproperty(ŷ_train, e) for e in set_diff]) : nothing 
    val_diffs = !isempty(set_diff) ? NamedTuple{Tuple(set_diff)}([getproperty(ŷ_val, e) for e in set_diff]) : nothing

    # TODO: save/output metrics

    return (; train_history, val_history, ps_history, train_obs_pred, val_obs_pred, train_diffs, val_diffs, αst_train, αst_val, ps, st)
end

function styled_values(nt; digits=5, color=nothing, paddings=nothing)
    formatted = [
        begin
            value_str = @sprintf("%.*f", digits, v)
            padded = isnothing(paddings) ? value_str : rpad(value_str, paddings[i])
            isnothing(color) ? padded  : styled"{$color:$padded}"
        end
        for (i,v) in enumerate(values(nt))
    ]
    return join(formatted, "  ")
end

function header_and_paddings(nt; digits=5)
    min_val_width = digits + 2  # 1 for "0", 1 for ".", rest for digits
    paddings = map(k -> max(length(string(k)), min_val_width), keys(nt))
    headers = [rpad(string(k), w) for (k, w) in zip(keys(nt), paddings)]
    return headers, paddings
end

"""
    prepare_data(hm, data)
Utility function to see if the data is already in the expected format or if further filtering and re-packing is needed.

# Arguments:
- hm: The Hybrid Model
- data: either a Tuple of KeyedArrays or a single KeyedArray.

Returns a tuple of KeyedArrays
"""
function prepare_data(hm, data)
    data = if isa(data, Tuple) # tuple of key arrays
        return data
    else
        targets = hm.targets
        predictors_forcing = Symbol[]

        # Collect all predictors and forcing variables by checking property names
        for prop in propertynames(hm)
            if occursin("predictors", string(prop))
                val = getproperty(hm, prop)
                if isa(val, NamedTuple)
                    append!(predictors_forcing, unique(vcat(values(val)...)))
                elseif isa(val, AbstractVector)
                    append!(predictors_forcing, val)
                end
            end
        end
        for prop in propertynames(hm)
            if occursin("forcing", string(prop))
                val = getproperty(hm, prop)
                if isa(val, AbstractVector)
                    append!(predictors_forcing, val)
                end
            end
        end
        predictors_forcing = unique(predictors_forcing)
        
        if isempty(predictors_forcing)
            @warn "Note that you don't have predictors or forcing variables."
        end
        if isempty(targets)
            @warn "Note that you don't have target names."
        end
        return (data(predictors_forcing), data(targets))
    end
end