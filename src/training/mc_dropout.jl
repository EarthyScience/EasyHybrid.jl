function evaluate_mc_dropout(
        ghm, x, y, y_no_nan, ps, st, loss_types, training_loss, extra_loss, agg;
        n_samples::Int = 100, file_path::Union{String, Nothing} = nothing, train_or_val_name::String = "val"
    )

    if !has_dropout(ghm)
        @info "MC Dropout skipped: no Dropout layers detected in the model.\nFalling back to standard deterministic evaluation."
        loss_val, sts, ŷ = evaluate_acc(ghm, x, y, y_no_nan, ps, st, loss_types, training_loss, extra_loss, agg)
        return _store_sample(file_path, train_or_val_name, ŷ, loss_val, nothing)
    end

    st_train = Lux.trainmode(st)

    for k in 1:n_samples
        loss_k, _, ŷ_k = compute_loss(
            ghm, ps, st_train,
            (x, (y, y_no_nan)),
            logging = LoggingLoss(
                train_mode = true,
                loss_types = loss_types,
                training_loss = training_loss,
                extra_loss = extra_loss,
                agg = agg
            )
        )
        _store_sample(file_path, train_or_val_name, ŷ_k, loss_k, k)
    end

    return nothing
end


function _store_sample(file_path::String, name, ŷ, loss, sample)
    return jldopen(file_path, "a+") do file
        key = isnothing(sample) ? name : "$(name)/sample_$(sample)"
        file["predictions/$key"] = ŷ
        file["losses/$key"] = loss
    end
end

function _store_sample(::Nothing, name, ŷ, loss, sample)
    return (; ŷ, loss)
end

function _has_dropout(model)
    return model isa Lux.Dropout || model isa Lux.AlphaDropout || model isa Lux.VariationalHiddenDropout
end

function _has_dropout(model::Lux.AbstractLuxContainerLayer)
    return any(_has_dropout, children(model))
end

function has_dropout(model)
    return _has_dropout(model)
end

function mc_dropout_statistics(storage::NamedTuple)
    predictions = [s.ŷ for s in storage]
    losses = [s.loss for s in storage]

    pred_stack = stack(predictions, dims = ndims(first(predictions)) + 1)
    mean_pred = mean(pred_stack, dims = ndims(pred_stack))
    var_pred = var(pred_stack, dims = ndims(pred_stack))
    mean_loss = mean(losses)

    return (; mean_pred, var_pred, mean_loss)
end

function mc_dropout_statistics(file_path::String, train_or_val_name::String)
    return jldopen(file_path, "r") do file
        keys = sort(keys(file["predictions/$train_or_val_name"]), by = k -> parse(Int, split(k, "_")[end]))
        losses = [file["losses/$train_or_val_name/$(k)"] for k in keys]

        # Welford online algorithm to avoid loading all predictions at once
        first_pred = file["predictions/$train_or_val_name/$(keys[1])"]
        mean_pred = copy(first_pred)
        M2 = zero(first_pred)
        mean_loss = first(losses)

        for (k, (key, loss)) in enumerate(zip(keys[2:end], losses[2:end]))
            ŷ_k = file["predictions/$train_or_val_name/$(key)"]
            delta = ŷ_k .- mean_pred
            mean_pred .+= delta ./ k
            delta2 = ŷ_k .- mean_pred
            M2 .+= delta .* delta2
            mean_loss += (loss - mean_loss) / k
        end

        var_pred = M2 ./ (length(keys) - 1)

        return (; mean_pred, var_pred, mean_loss)
    end
end
