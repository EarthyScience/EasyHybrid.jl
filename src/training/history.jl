struct TrainingHistory
    snapshots::Vector{EpochSnapshot}
end

TrainingHistory(init::EpochSnapshot) = TrainingHistory([init])

train_losses(history::TrainingHistory) = [s.l_train for s in history.snapshots]
val_losses(history::TrainingHistory) = [s.l_val   for s in history.snapshots]
predictions(history::TrainingHistory) = [s.ŷ_train for s in history.snapshots]
get_epochs(history::TrainingHistory) = [s.epoch for s in history.snapshots]

function get_loss_value_t(history::TrainingHistory, loss_type::Symbol, agg::Symbol)
    return [get_loss_value(s.l_train, loss_type, agg) for s in history.snapshots]
end

function get_loss_value_v(history::TrainingHistory, loss_type::Symbol, agg::Symbol)
    return [get_loss_value(s.l_val, loss_type, agg) for s in history.snapshots]
end

function get_monitor_values(history::TrainingHistory, monitor_names::Vector{Symbol}, dataset::Symbol = :train)
    if dataset == :train
        return [get_monitor_values(s.ŷ_train, monitor_names) for s in history.snapshots]
    elseif dataset == :val
        return [get_monitor_values(s.ŷ_val, monitor_names) for s in history.snapshots]
    else
        throw(ArgumentError("Invalid dataset specified. Use :train or :val."))
    end
end

function collect_monitor_history(history_vec::Vector, monitor_names::Vector{Symbol})
    return (;
        (m => _collect_monitor_field(history_vec, m) for m in monitor_names)...,
    )
end

function _collect_monitor_field(history_vec::Vector, name::Symbol)
    entries = [getfield(snap, name) for snap in history_vec]
    first_entry = first(entries)

    if haskey(first_entry, :scalar)
        # scalar case: collect into a single vector
        return (; :scalar => [e.scalar for e in entries])

    elseif haskey(first_entry, :quantile)
        # quantile case: collect each quantile level into its own vector
        qlabels = keys(first_entry.quantile)
        return (;
            :quantile => (;
                (q => [e.quantile[q] for e in entries] for q in qlabels)...,
            ),
        )
    else
        error("Unknown monitor entry format for field $name")
    end
end

export get_loss_value_t, get_loss_value_v
export collect_monitor_history
export get_epochs
export get_monitor_values
