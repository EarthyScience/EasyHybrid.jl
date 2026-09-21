export prepare_splits, maybe_build_sequences

function prepare_splits(data, model, cfg::DataConfig)
    # Already-split train/val tuple from split_data — skip re-splitting (e.g. cv_test folds).
    if data isa Tuple && length(data) == 2 &&
            data[1] isa Tuple && length(data[1]) == 2 &&
            data[2] isa Tuple && length(data[2]) == 2
        return data
    end

    ((x_train, forcings_train), y_train), ((x_val, forcings_val), y_val) = split_data(
        data, model;
        array_type = cfg.array_type,
        shuffleobs = cfg.shuffleobs,
        split_by_id = cfg.split_by_id,
        split_data_at = cfg.split_data_at,
        folds = cfg.folds,
        val_fold = cfg.val_fold,
        # pass sequence options through if set
        sequence_kwargs = maybe_build_sequence_kwargs(cfg),
    )

    @debug "Train size: $(size(x_train)), Val size: $(size(x_val))"
    @debug "Data type: $(typeof(x_train))"

    return ((x_train, forcings_train), y_train), ((x_val, forcings_val), y_val)
end

function maybe_build_sequence_kwargs(cfg::DataConfig)
    isnothing(cfg.sequence_length) && return nothing
    return (;
        input_window = cfg.sequence_length,
        output_window = cfg.sequence_output_window,
        output_shift = cfg.sequence_output_shift,
        lead_time = cfg.sequence_lead_time,
    )
end
