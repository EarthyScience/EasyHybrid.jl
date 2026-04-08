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
