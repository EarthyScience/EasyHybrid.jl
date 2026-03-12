@kwdef struct DataConfig
    array_type::Symbol = :KeyedArray
    shuffleobs::Bool = false
    split_by_id = nothing
    split_data_at::Float64 = 0.8
    folds = nothing
    val_fold = nothing
    # sequence-related
    sequence_length::Union{Int, Nothing} = nothing
    sequence_output_window::Int = 1
    sequence_output_shift::Int = 1
    sequence_lead_time::Int = 1
end
