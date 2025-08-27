# CC BY-SA 4.0
# TODO put this into train
# ? test lossfn
ps, st = LuxCore.setup(Random.default_rng(), hm)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_p_f, ds_t = EasyHybrid.prepare_data(hm, ds_keyed)
ds_t_nan = .!isnan.(ds_t)

import EasyHybrid: loss_fn
function EasyHybrid.loss_fn(ŷ, y, y_nan, ::Val{:nse})
    return sum((ŷ[y_nan] .- y[y_nan]).^2) / sum((y[y_nan] .- mean(y[y_nan])).^2)
end

function EasyHybrid.lossfn(HM::HybridModel15, x, (y_t, y_nan), ps, st, logging::LoggingLoss)
    targets = HM.targets
    ŷ, y, y_nan = EasyHybrid.get_predictions_targets(HM, x, (y_t, y_nan), ps, st, targets)
    if logging.train_mode
        return EasyHybrid.compute_loss(ŷ, y, y_nan, targets, logging.training_loss, logging.agg), st
    else
        return EasyHybrid.compute_loss(ŷ, y, y_nan, targets, logging.loss_types, logging.agg), st
    end
end

ls = EasyHybrid.lossfn(hm, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss(training_loss=:nse, loss_types=[:mse, :nse]))

tout = train(hm, ds_keyed, (); nepochs=100, batchsize=512, opt=AdaGrad(0.01), file_name = "tout.jld2", training_loss=:nse, loss_types=[:mse, :nse])


ls2 = (p, data) -> EasyHybrid.lossfn(hm, ds_p_f, (ds_t, ds_t_nan), p, st, LoggingLoss(training_loss=:nse, loss_types=[:nse]))[1]

dta = (ds_p_f, ds_t, ds_t_nan)

# TODO check if minibatching is doing what is supposed to do - ncycle was used before:
# https://docs.sciml.ai/Optimization/stable/tutorials/minibatch/
using EasyHybrid.MLUtils
(x_train, y_train, nan_train), (x_val, y_val, nan_val) = splitobs(dta; at=0.8, shuffle=false)
dataloader = DataLoader((x_train, y_train, nan_train), batchsize=512, shuffle=true);

ps, st = LuxCore.setup(Random.default_rng(), hm)

ps_ca = ComponentArray(ps) .|> Float64
ls2(ps_ca, dta)

opt_func = OptimizationFunction(ls2, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps_ca, dataloader)

using EasyHybrid.Printf
epochs = 10
n_minibatches = length(dataloader)
function callback(state, l)
    state.iter % n_minibatches == 1 && @printf "Epoch: %5d, Loss: %.2e\n" state.iter/n_minibatches+1 l
    return l < 1e-8 ## Terminate if loss is small
end

res_adam = solve(opt_prob, Optimisers.Adam(0.001); callback=callback, epochs= epochs)
ls2(res_adam.u, dta)

opt_prob = remake(opt_prob; u0=res_adam.u)

res_lbfgs = solve(opt_prob, Optimization.LBFGS(); callback, maxiters=1000)
ls2(res_lbfgs.u, dta)





