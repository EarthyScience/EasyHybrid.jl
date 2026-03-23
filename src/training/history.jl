struct TrainingHistory
    snapshots::Vector{EpochSnapshot}
end

TrainingHistory(init::EpochSnapshot) = TrainingHistory([init])

train_losses(history::TrainingHistory) = [s.l_train for s in history.snapshots]
val_losses(history::TrainingHistory) = [s.l_val   for s in history.snapshots]
predictions(history::TrainingHistory) = [s.ŷ_train for s in history.snapshots]
