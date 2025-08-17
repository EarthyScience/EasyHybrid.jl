export train, TrainResults


"""
    split_data(data, split_by_id; shuffleobs=false, split_ratio=0.8)

Split data into training and validation sets, either randomly or by grouping by ID.

# Arguments:
- `data`: The data to split, typically a tuple of (x, y) KeyedArrays
- `split_by_id`: Either `nothing` for random splitting, a `Symbol` for column-based splitting, or an `AbstractVector` for custom ID-based splitting
- `shuffleobs`: Whether to shuffle observations during splitting (default: false)
- `split_ratio`: Ratio of data to use for training (default: 0.8)

# Returns:
- `(x_train, y_train)`: Training data tuple
- `(x_val, y_val)`: Validation data tuple
"""
function split_data(data, hybridModel; split_by_id=nothing, shuffleobs=false, split_data_at=0.8)
    
    data_ = prepare_data(hybridModel, data)
    # all the KeyedArray thing!

    if !isnothing(split_by_id)
        if isa(split_by_id, Symbol)
            ids = getbyname(data, split_by_id)
            unique_ids = unique(ids)
        elseif isa(split_by_id, AbstractVector)
            ids = split_by_id
            unique_ids = unique(ids)
            split_by_id = "split_by_id"
        end

        train_ids, val_ids = splitobs(unique_ids; at=split_data_at, shuffle=shuffleobs)

        train_idx = findall(id -> id in train_ids, ids)
        val_idx  = findall(id -> id in val_ids,  ids)

        @info "Splitting data by $split_by_id"
        @info "Number of unique $split_by_id's: $(length(unique_ids))"
        @info "Number of $split_by_id's in training set: $(length(train_ids))"
        @info "Number of $split_by_id's in validation set: $(length(val_ids))"
        
        x_all, y_all = data_

        x_train, y_train = x_all[:, train_idx], y_all[:, train_idx]
        x_val, y_val = x_all[:, val_idx], y_all[:, val_idx]
    else
        (x_train, y_train), (x_val, y_val) = splitobs(data_; at=split_data_at, shuffle=shuffleobs)
    end
    
    return (x_train, y_train), (x_val, y_val)
end

# beneficial for plotting based on type TrainResults?
struct TrainResults
    train_history
    val_history
    ps_history
    train_obs_pred
    val_obs_pred
    train_diffs
    val_diffs
    αst_train
    αst_val
    ps
    st
end

"""
    train(hybridModel, data, save_ps; nepochs=200, batchsize=10, opt=Adam(0.01), patience=typemax(Int),
          file_name=nothing, loss_types=[:mse, :r2], training_loss=:mse, agg=sum, train_from=nothing,
          random_seed=161803, shuffleobs=false, yscale=log10, monitor_names=[], return_model=:best, 
          split_by_id=nothing, split_data_at=0.8, plotting=true, show_progress=true, hybrid_name=randstring(10))

Train a hybrid model using the provided data and save the training process to a file in JLD2 format. 
Default output file is `trained_model.jld2` at the current working directory under `output_tmp`.

# Arguments:
- `hybridModel`: The hybrid model to be trained.
- `data`: The training data, either a single DataFrame, a single KeyedArray, or a tuple of KeyedArrays.
- `save_ps`: A tuple of physical parameters to save during training.

## Core Training Parameters:
- `nepochs`: Number of training epochs (default: 200).
- `batchsize`: Size of the training batches (default: 10).
- `opt`: The optimizer to use for training (default: Adam(0.01)).
- `patience`: The number of epochs to wait before early stopping (default: `typemax(Int)` -> no early stopping).

## Loss and Evaluation:
- `training_loss`: The loss type to use during training (default: `:mse`).
- `loss_types`: A vector of loss types to compute during training (default: `[:mse, :r2]`).
- `agg`: The aggregation function to apply to the computed losses (default: `sum`).

## Data Handling:
- `shuffleobs`: Whether to shuffle the training data (default: false).
- `split_by_id`: Column name or function to split data by ID (default: nothing -> no ID-based splitting).
- `split_data_at`: Fraction of data to use for training when splitting (default: 0.8).

## Training State and Reproducibility:
- `train_from`: A tuple of physical parameters and state to start training from or an output of `train` (default: nothing -> new training).
- `random_seed`: The random seed to use for training (default: 161803).

## Output and Monitoring:
- `file_name`: The name of the file to save the training process (default: nothing -> "trained_model.jld2").
- `hybrid_name`: Name identifier for the hybrid model (default: randomly generated 10-character string).
- `return_model`: The model to return: `:best` for the best model, `:final` for the final model (default: `:best`).
- `monitor_names`: A vector of monitor names to track during training (default: `[]`).

## Visualization and UI:
- `plotting`: Whether to generate plots during training (default: true).
- `show_progress`: Whether to show progress bars during training (default: true).
- `yscale`: The scale to apply to the y-axis for plotting (default: `log10`).
"""
function train(hybridModel, data, save_ps; 
               # Core training parameters
               nepochs=200, 
               batchsize=10, 
               opt=Adam(0.01), 
               patience=typemax(Int),
               
               # Loss and evaluation
               training_loss=:mse,
               loss_types=[:mse, :r2], 
               agg=sum, 
               
               # Data handling
               shuffleobs=false,
               split_by_id=nothing, 
               split_data_at=0.8, 
               
               # Training state and reproducibility
               train_from=nothing,
               random_seed=161803, 
               
               # Output and monitoring
               file_name=nothing, 
               hybrid_name=randstring(10),
               return_model=:best,
               monitor_names=[], 

               # Visualization and UI
               plotting=true, 
               show_progress=true,
               yscale=log10)
               
    #! check if the EasyHybridMakie extension is loaded.
    ext = Base.get_extension(@__MODULE__, :EasyHybridMakie)
    
    if ext === nothing
        @warn "Makie extension not loaded, no plots will be generated."
    end
    
    if !plotting
        ext = nothing
        @info "Plotting disabled."
    end

    if !isnothing(random_seed)
        Random.seed!(random_seed)
    end

    # ? split training and validation data
    (x_train, y_train), (x_val, y_val) = split_data(data, hybridModel; split_by_id=split_by_id, shuffleobs=shuffleobs, split_data_at=split_data_at)

    train_loader = DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true);

    if isnothing(train_from)
        ps, st = LuxCore.setup(Random.default_rng(), hybridModel)
    else
        ps, st = get_ps_st(train_from)
    end

    opt_state = Optimisers.setup(opt, ps)

    # ? initial losses
    is_no_nan_t = .!isnan.(y_train)
    is_no_nan_v = .!isnan.(y_val)

    l_init_train, _, init_ŷ_train =  evaluate_acc(hybridModel, x_train, y_train, is_no_nan_t, ps, st, loss_types, training_loss, agg)
    l_init_val, _, init_ŷ_val = evaluate_acc(hybridModel, x_val, y_val, is_no_nan_v, ps, st, loss_types, training_loss, agg)

    train_history = [l_init_train]
    val_history = [l_init_val]
    target_names = hybridModel.targets
    fig = nothing
    # Initialize plotting observables if extension is loaded
    if !isnothing(ext)
        init_observables, fixed_observations = initialize_plotting_observables(
            init_ŷ_train,
            init_ŷ_val,
            y_train,
            y_val,
            l_init_train,
            l_init_val,
            training_loss,
            agg,
            target_names;
            monitor_names
            )
        zoom_epochs = min(patience, 50)
        # ! Launch dashboard if extension is loaded
        EasyHybrid.train_board(init_observables..., fixed_observations..., yscale, target_names; monitor_names, zoom_epochs)
        fig = EasyHybrid.dashboard_figure()
    end

    # track physical parameters
    ps_values_init = [copy(getproperty(ps, e)[1]) for e in save_ps]
    ps_init = NamedTuple{save_ps}(ps_values_init)
    ps_history = [ps_init]

    # For Early stopping
    best_ps = deepcopy(ps)
    best_st = deepcopy(st)
    best_loss = l_init_val
    best_epoch = 0
    cnt_patience = 0
    
    # Initialize best_agg_loss for early stopping comparison based on the first loss_types in [:mse, :r2]
    best_agg_loss = getproperty(l_init_val[1], Symbol(agg))
    val_metric_name = first(keys(l_init_val))
    current_agg_loss = best_agg_loss  # Initialize for potential use in final logging
    
    file_name = resolve_path(file_name)
    save_ps_st(file_name, hybridModel, ps, st, save_ps)
    save_train_val_loss!(file_name,l_init_train, "training_loss", 0)
    save_train_val_loss!(file_name,l_init_val, "validation_loss", 0)

    # save/record
    tmp_folder = get_output_path()
    @info "Check the saved output (.png, .mp4, .jld2) from training at: $(tmp_folder)"

    prog = Progress(nepochs, desc="Training loss", enabled=show_progress)
    maybe_record_history(!isnothing(ext), fig, joinpath(tmp_folder, "training_history_$(hybrid_name).mp4"); framerate=24) do io
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
            
            l_train, _, current_ŷ_train = evaluate_acc(hybridModel, x_train, y_train, is_no_nan_t, ps, st, loss_types, training_loss, agg)
            l_val, _, current_ŷ_val = evaluate_acc(hybridModel, x_val, y_val, is_no_nan_v, ps, st, loss_types, training_loss, agg)

            save_train_val_loss!(file_name, l_train, "training_loss", epoch)
            save_train_val_loss!(file_name, l_val, "validation_loss", epoch)
            
            push!(train_history, l_train)
            push!(val_history, l_val)

            # Update plotting observables if extension is loaded
            if !isnothing(ext)
                EasyHybrid.update_plotting_observables(
                    init_observables...,
                    l_train,
                    l_val,
                    training_loss,
                    agg,
                    current_ŷ_train,
                    current_ŷ_val,
                    target_names,
                    epoch;
                    monitor_names)
                # record a new frame
                EasyHybrid.recordframe!(io)
            end

            current_agg_loss = getproperty(l_val[1], Symbol(agg))
            
            if current_agg_loss < best_agg_loss
                best_agg_loss = current_agg_loss
                best_ps = deepcopy(ps)
                best_st = deepcopy(st)
                cnt_patience = 0
                best_epoch = epoch
            else
                cnt_patience += 1
            end
            if cnt_patience >= patience
                metric_name = first(keys(l_val))
                if !isnothing(ext)
                    img_name = joinpath(tmp_folder, "train_history_best_epoch_$(best_epoch).png")
                    EasyHybrid.save_fig(img_name, EasyHybrid.dashboard_figure())
                    img_name = joinpath(tmp_folder, "train_history_$(hybrid_name).png")
                    EasyHybrid.save_fig(img_name, EasyHybrid.dashboard_figure())
                end
                @warn "Early stopping at epoch $epoch with best validation loss wrt $metric_name: $best_agg_loss"
                break
            end

            if !isnothing(ext) && epoch == nepochs
                img_name = joinpath(tmp_folder, "train_history_best_epoch_$(best_epoch).png")
                EasyHybrid.save_fig(img_name, EasyHybrid.dashboard_figure())
                img_name = joinpath(tmp_folder, "train_history_$(hybrid_name).png")
                EasyHybrid.save_fig(img_name, EasyHybrid.dashboard_figure())
            end

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
    end

    train_history = WrappedTuples(train_history)
    val_history = WrappedTuples(val_history)
    ps_history = WrappedTuples(ps_history)

    # ? save final evaluation or best at best validation value
    if return_model == :best
        ps, st = deepcopy(best_ps), deepcopy(best_st)
        @info "Returning best model from epoch $best_epoch of $nepochs epochs with best validation loss wrt $val_metric_name: $best_agg_loss"
    elseif return_model == :final
        ps, st = deepcopy(ps), deepcopy(st)
        @info "Returning final model from final of $nepochs epochs with validation loss: $current_agg_loss, the best validation loss was $best_agg_loss from epoch $best_epoch wrt $val_metric_name"
    else
        @warn "Invalid return_model: $return_model. Returning final model."
    end

    ŷ_train, αst_train = hybridModel(x_train, ps, LuxCore.testmode(st))
    ŷ_val, αst_val = hybridModel(x_val, ps, LuxCore.testmode(st))
    save_predictions!(file_name, ŷ_train, αst_train, "training")
    save_predictions!(file_name, ŷ_val, αst_val, "validation")

    # training
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
    return TrainResults(
        train_history,
        val_history,
        ps_history,
        train_obs_pred,
        val_obs_pred,
        train_diffs,
        val_diffs,
        αst_train,
        αst_val,
        ps,
        st
    )
end

function evaluate_acc(ghm, x, y, y_no_nan, ps, st, loss_types, training_loss, agg)
    loss_val, sts, ŷ = lossfn(ghm, x, (y, y_no_nan), ps, LuxCore.testmode(st),
        LoggingLoss(train_mode=false, loss_types=loss_types, training_loss=training_loss, agg=agg)
        )
    return loss_val, sts, ŷ
end
function maybe_record_history(block, should_record, fig, output_path; framerate=24)
    if should_record
        EasyHybrid.record_history(fig, output_path; framerate=framerate) do io
            block(io)
        end
    else
        block(nothing)  # call with dummy io
    end
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
function prepare_data(hm, data::KeyedArray)
        targets = hm.targets
        predictors_forcing = Symbol[]

        # Collect all predictors and forcing variables by checking property names
        for prop in propertynames(hm)
            if occursin("predictors", string(prop))
                val = getproperty(hm, prop)
                if isa(val, AbstractVector)
                    append!(predictors_forcing, val)
                elseif isa(val, Union{NamedTuple, Tuple})
                    append!(predictors_forcing, unique(vcat(values(val)...)))
                end
            end
        end
        for prop in propertynames(hm)
            if occursin("forcing", string(prop))
                val = getproperty(hm, prop)
                if isa(val, AbstractVector)
                    append!(predictors_forcing, val)
                elseif isa(val, Union{Tuple, NamedTuple})
                    append!(predictors_forcing, unique(vcat(values(val)...)))
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

    function prepare_data(hm, data::DataFrame)
        targets = hm.targets
        predictors_forcing = Symbol[]

        # Collect all predictors and forcing variables by checking property names
        for prop in propertynames(hm)
            if occursin("predictors", string(prop))
                val = getproperty(hm, prop)
                if isa(val, AbstractVector)
                    append!(predictors_forcing, val)
                elseif isa(val, Union{NamedTuple, Tuple})
                    append!(predictors_forcing, unique(vcat(values(val)...)))
                end
            end
        end
        for prop in propertynames(hm)
            if occursin("forcing", string(prop))
                val = getproperty(hm, prop)
                if isa(val, AbstractVector)
                    append!(predictors_forcing, val)
                elseif isa(val, Union{Tuple, NamedTuple})
                    append!(predictors_forcing, unique(vcat(values(val)...)))
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

        all_predictor_cols  = unique(vcat(values(predictors_forcing)...))
        col_to_select       = unique([all_predictor_cols; targets])
    
        # subset to only the cols we care about
        sdf = data[!, col_to_select]
    
        # Separate predictor/forcing vs. target columns
        predforce_cols = setdiff(col_to_select, targets)
        
        # For each row, check if *any* predictor/forcing is missing
        mask_missing_predforce = map(row -> any(ismissing, row), eachrow(sdf[:, predforce_cols]))
        
        # For each row, check if *at least one* target is present (i.e. not all missing)
        mask_at_least_one_target = map(row -> any(!ismissing, row), eachrow(sdf[:, targets]))
        
        # Keep rows where predictors/forcings are *complete* AND there's some target present
        keep = .!mask_missing_predforce .& mask_at_least_one_target
        sdf = sdf[keep, col_to_select]
    
        mapcols(col -> replace!(col, missing => NaN), sdf; cols = names(sdf, Union{Missing, Real}))
    
        # Convert to Float32 and to your keyed array
        ds_keyed = to_keyedArray(Float32.(sdf))
        return prepare_data(hm, ds_keyed)
    end

function prepare_data(hm, data::Tuple)
    return data
end

function get_ps_st(train_from::TrainResults)
    return train_from.ps, train_from.st
end

function get_ps_st(train_from::Tuple)
    return train_from
end

function getbyname(df::DataFrame, name::Symbol)
    return df[!, name]
end

function getbyname(ka::AxisKeys.KeyedArray, name::Symbol)
    return ka(name)
end
