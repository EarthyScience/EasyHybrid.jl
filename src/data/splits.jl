export prepare_splits, maybe_build_sequences

function prepare_splits(data, model, cfg::DataConfig)
    data = maybe_build_sequences(data, cfg)   # no-op if sequence_length is nothing
    (x_train, y_train), (x_val, y_val) = split_data(
        data, model;
        array_type = cfg.array_type,
        shuffleobs = cfg.shuffleobs,
        split_by_id = cfg.split_by_id,
        split_data_at = cfg.split_data_at,
        folds = cfg.folds,
        val_fold = cfg.val_fold,
    )
    @debug "Train size: $(size(x_train)), Val size: $(size(x_val))"
    @debug "Data type: $(typeof(x_train))"
    return (x_train, y_train), (x_val, y_val)
end

function maybe_build_sequences(data, cfg::DataConfig)
    isnothing(cfg.sequence_length) && return data

    x, y = data
    return split_into_sequences(
        x, y;
        input_window = cfg.sequence_length,
        output_window = cfg.sequence_output_window,
        output_shift = cfg.sequence_output_shift,
        lead_time = cfg.sequence_lead_time,
    )
end
