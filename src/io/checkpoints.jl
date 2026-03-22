function save_initial_state!(paths::TrainingPaths, model, ps, st, cfg::TrainConfig)
    save_ps_st(paths.checkpoint, model, ps, st, cfg.tracked_params)
    save_train_val_loss!(paths.checkpoint, nothing, "training_loss", 0)
    save_train_val_loss!(paths.checkpoint, nothing, "validation_loss", 0)
    return nothing
end

function save_epoch!(paths::TrainingPaths, model, ps, st, snapshot::EpochSnapshot, epoch::Int, cfg::TrainConfig)
    save_ps_st!(paths.checkpoint, model, ps, st, cfg.tracked_params, epoch)
    save_train_val_loss!(paths.checkpoint, snapshot.l_train, "training_loss", epoch)
    save_train_val_loss!(paths.checkpoint, snapshot.l_val, "validation_loss", epoch)
    return nothing
end

function save_final!(paths::TrainingPaths, model, ps, st, x_train, y_train, x_val, y_val, stopper::EarlyStopping, cfg::TrainConfig)
    target_names = model.targets
    save_epoch = stopper.best_epoch == 0 ? 0 : stopper.best_epoch
    save_ps_st!(paths.best_model, model, ps, st, cfg.tracked_params, save_epoch)

    ŷ_train, αst_train = model(x_train, ps, LuxCore.testmode(st))
    ŷ_val, αst_val = model(x_val, ps, LuxCore.testmode(st))

    save_predictions!(paths.checkpoint, ŷ_train, αst_train, "training")
    save_predictions!(paths.checkpoint, ŷ_val, αst_val, "validation")
    save_observations!(paths.checkpoint, target_names, y_train, "training")
    save_observations!(paths.checkpoint, target_names, y_val, "validation")

    config_settings = get_full_config(model, cfg)
    save_hybrid_config(config_settings, paths.config_yaml)
    return nothing
end
