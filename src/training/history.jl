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

# mutable struct TrainingHistory
#     train::Vector
#     val::Vector
#     ps::Vector
# end

# function record_epoch!(history::TrainingHistory, l_train, l_val, ps_snapshot)
#     push!(history.train, l_train)
#     push!(history.val, l_val)
#     push!(history.ps, ps_snapshot)
# end
