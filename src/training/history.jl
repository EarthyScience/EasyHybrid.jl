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

export get_loss_value_t, get_loss_value_v
export get_epochs
