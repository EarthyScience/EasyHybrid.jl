using Pkg
Pkg.activate("projects/RbQ10")
Pkg.develop(path=pwd())
Pkg.instantiate()

using EasyHybrid
using EasyHybrid.Printf
using EasyHybrid.MLUtils


script_dir = @__DIR__
include(joinpath(script_dir, "data", "prec_process_data.jl"))

# Common data preprocessing
df = dfall[!, Not(:timesteps)]
ds_keyed = to_keyedArray(Float32.(df))

target_names = [:R_soil]
forcing_names = [:cham_temp_filled]
predictor_names = [:moisture_filled, :rgpot2]

# Define neural network
NN = Chain(Dense(2, 15, relu), Dense(15, 15, relu), Dense(15, 1));
# instantiate Hybrid Model
RbQ10 = RespirationRbQ10(NN, predictor_names, forcing_names, target_names, 2.5f0) # ? do different initial Q10s
# Define neural network
NN = Lux.Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 1));

ps, st = LuxCore.setup(Random.default_rng(), RbQ10)

ps_ca = ComponentArray(ps) .|> Float64
# smodel = StatefulLuxLayer{false}(RbQ10, nothing, st)
# deal with the `Rb` state also here, (; Rb, st), since this is the output from LuxCore.apply.
# ! note that for now is set to `{false}`.

function callback(state, l)
    state.iter % 2 == 1 && @printf "Iteration: %5d, Loss: %.6f\n" state.iter l
    return l < 0.2 ## Terminate if loss is smaller than
end

function callback(state, l) #callback function to observe training
    display(l)
    return false
end

# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_p_f, ds_t = EasyHybrid.prepare_data(RbQ10, ds_keyed)
ds_t_nan = .!isnan.(ds_t)
ls = EasyHybrid.lossfn(RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss(train_mode=false))

ls2 = (p, data) -> EasyHybrid.lossfn(RbQ10, ds_p_f, (ds_t, ds_t_nan), p, st, LoggingLoss())[1]

dta = (ds_p_f, ds_t, ds_t_nan)

# TODO check if minibatching is doing what is supposed to do - ncycle was used before:
# https://docs.sciml.ai/Optimization/stable/tutorials/minibatch/
(x_train, y_train, nan_train), (x_val, y_val, nan_val) = splitobs(dta; at=0.8, shuffle=false)
dataloader = DataLoader((x_train, y_train, nan_train), batchsize=512, shuffle=true);

ls2(ps_ca, dta)

opt_func = OptimizationFunction(ls2, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps_ca, dataloader)

epochs = 10
n_minibatches = length(dataloader)
function callback(state, l)
    state.iter % n_minibatches == 1 && @printf "Epoch: %5d, Loss: %.6e\n" state.iter/n_minibatches+1 l
    return l < 1e-8 ## Terminate if loss is small
end

res_adam = solve(opt_prob, Optimisers.Adam(0.001); callback, epochs)
ls2(res_adam.u, dta)

opt_prob = remake(opt_prob; u0=res_adam.u)

res_lbfgs = solve(opt_prob, Optimization.LBFGS(); callback, maxiters=1000)
ls2(res_lbfgs.u, dta)


#res_shopia = solve(opt_prob, Optimization.Sophia(); callback, maxiters=epochs)

# ! finetune a bit with L-BFGS
# ?  LBFGS needs to this `convert(Float64, res_adam.u)` which it fails!
# ! but there is more, see issue: https://github.com/LuxDL/Lux.jl/issues/1260

