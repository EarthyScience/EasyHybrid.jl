struct TrainingHistory
    snapshots::Vector{EpochSnapshot}
end

# constructor from initial state
TrainingHistory(init::EpochSnapshot) = TrainingHistory([init])

function update!(history::TrainingHistory, snapshot::EpochSnapshot)
    return push!(history.snapshots, snapshot)
end

# convenience accessors
train_losses(history::TrainingHistory) = [s.l_train for s in history.snapshots]
val_losses(history::TrainingHistory) = [s.l_val   for s in history.snapshots]
predictions(history::TrainingHistory) = [s.ŷ_train for s in history.snapshots]
