mutable struct EarlyStopping
    best_loss
    best_ps
    best_st
    best_snapshot::EpochSnapshot
    best_epoch::Int
    counter::Int
    patience::Int
    done::Bool
end

function EarlyStopping(init_loss, init_snapshot::EpochSnapshot, ps, st, patience::Int)
    best_loss = extract_agg_loss(init_loss)
    return EarlyStopping(best_loss, deepcopy(ps), deepcopy(st), init_snapshot, 0, 0, patience, false)
end

function update!(es::EarlyStopping, history::TrainingHistory, snapshot::EpochSnapshot, ps, st, epoch, cfg::TrainConfig)
    push_snapshot = if cfg.keep_history
        EpochSnapshot(snapshot.l_train, snapshot.l_val, deepcopy(snapshot.ŷ_train), deepcopy(snapshot.ŷ_val))
    else
        snapshot
    end
    cfg.keep_history && push!(history.snapshots, push_snapshot)

    current_loss = extract_agg_loss(snapshot.l_val)

    if isbetter(current_loss, es.best_loss, first(cfg.loss_types))
        es.best_loss = current_loss
        es.best_ps = deepcopy(ps)
        es.best_st = deepcopy(st)
        es.best_epoch = epoch
        es.best_snapshot = EpochSnapshot(snapshot.l_train, snapshot.l_val, deepcopy(snapshot.ŷ_train), deepcopy(snapshot.ŷ_val))
        es.counter = 0
    else
        es.counter += 1
    end

    return if es.counter >= es.patience
        metric_name = first(keys(snapshot.l_val))
        @warn "Early stopping at epoch $epoch, best validation loss wrt $metric_name: $(es.best_loss) at epoch $(es.best_epoch)"
        es.done = true
    end
end

is_done(es::EarlyStopping) = es.done

# helper — extracts the aggregated scalar from a loss NamedTuple
function extract_agg_loss(l, agg::Union{Function, Symbol} = :sum)
    return getproperty(l[1], Symbol(agg))
end

function best_or_final(stopper::EarlyStopping, ps, st, cfg::TrainConfig)
    if cfg.return_model == :best
        @info """
        Returning best model from epoch $(stopper.best_epoch) \
        with validation loss: $(stopper.best_loss)
        """
        return deepcopy(stopper.best_ps), deepcopy(stopper.best_st)

    elseif cfg.return_model == :final
        @info """
        Returning final model. \
        Best validation loss was $(stopper.best_loss) at epoch $(stopper.best_epoch)
        """
        return ps, st

    else
        # should never reach here if validate_config ran
        @warn "Invalid return_model: $(cfg.return_model), returning final model."
        return ps, st
    end
end

function build_results(model, history::TrainingHistory, stopper::EarlyStopping, ps, st, x_train, y_train, x_val, y_val)
    target_names = model.targets

    # final predictions in test mode
    ŷ_train, _ = model(x_train, ps, LuxCore.testmode(st))
    ŷ_val, _ = model(x_val, ps, LuxCore.testmode(st))

    # observed vs predicted DataFrames
    train_obs_pred = hcat(toDataFrame(y_train), toDataFrame(ŷ_train, target_names))
    val_obs_pred = hcat(toDataFrame(y_val), toDataFrame(ŷ_val, target_names))

    # extra predictions without observational counterparts
    train_diffs, val_diffs = extract_diffs(ŷ_train, ŷ_val, target_names)

    return TrainResults(
        WrappedTuples(train_losses(history)),
        WrappedTuples(val_losses(history)),
        WrappedTuples(history.snapshots),
        train_obs_pred,
        val_obs_pred,
        train_diffs,
        val_diffs,
        ps,
        st,
        stopper.best_epoch,
        stopper.best_loss,
    )
end

function extract_diffs(ŷ_train, ŷ_val, target_names)
    extra_keys = setdiff(keys(ŷ_train), target_names)

    isempty(extra_keys) && return nothing, nothing

    train_diffs = NamedTuple{Tuple(extra_keys)}([getproperty(ŷ_train, k) for k in extra_keys])
    val_diffs = NamedTuple{Tuple(extra_keys)}([getproperty(ŷ_val, k) for k in extra_keys])

    return train_diffs, val_diffs
end
