export train

"""
    train(hybridModel, data; nepochs=200, batchsize=10, opt=Adam(0.01))
"""
function train(hybridModel, data, save_ps; nepochs=200, batchsize=10, opt=Adam(0.01))
    # ? split training and validation data
    (x_train, y_train), (x_val, y_val) = splitobs(data; at=0.8, shuffle=false)
    train_loader = DataLoader((x_train, y_train), batchsize=batchsize, shuffle=true);
    # ? setup model
    ps, st = LuxCore.setup(Random.default_rng(), hybridModel)
    opt_state = Optimisers.setup(opt, ps)

    # ? initial losses
    is_no_nan_t = .!isnan.(y_train)
    is_no_nan_v = .!isnan.(y_val)
    l_init_train = lossfn(hybridModel, x_train, (y_train, is_no_nan_t), ps, st)
    l_init_val = lossfn(hybridModel, x_val, (y_val, is_no_nan_v), ps, st)

    prog = Progress(nepochs, desc="Training")
    train_history = [l_init_train]
    val_history = [l_init_val]
    ps_history = [copy(getproperty(ps, e)[1]) for e in save_ps]
    for epoch in 1:nepochs
        for (x, y) in train_loader
            # ? check NaN indices before going forward, and pass filtered `x, y`.
            is_no_nan_batch = .!isnan.(y)
            # if length(is_no_nan)>0 # ! be careful here, multivariate needs fine tuning
            if any(is_no_nan_batch)
                grads = Zygote.gradient((ps) -> lossfn(hybridModel, x, (y, is_no_nan_batch), ps, st), ps)[1]
                Optimisers.update!(opt_state, ps, grads)
            end
        end
        tmp_e = [copy(getproperty(ps, e)[1]) for e in save_ps]
        push!(ps_history, tmp_e...)

        l_train = lossfn(hybridModel, x_train,  (y_train, is_no_nan_t), ps, st)
        l_val = lossfn(hybridModel, x_val, (y_val, is_no_nan_v), ps, st)

        push!(train_history, l_train)
        push!(val_history, l_val)
        next!(prog; showvalues = [("epoch", epoch), ("Initial_loss", l_init_train), ("Current_loss", l_train), ("Validation_loss", l_val)])
    end
    ŷ_train, αst_train = hybridModel(x_train, ps, st)
    ŷ_val, αst_val = hybridModel(x_val, ps, st)
    return (; train_history, val_history, ŷ_train, αst_train, ŷ_val, αst_val, y_train, y_val, ps_history, ps, st)
end