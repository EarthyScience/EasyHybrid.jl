using Pkg
Pkg.activate("projects/RbQ10")
Pkg.develop(path=pwd())
Pkg.instantiate()

using EasyHybrid
using GLMakie
using AlgebraOfGraphics
using Random
using EasyHybrid.MLUtils
using EasyHybrid.AxisKeys
using EasyHybrid.JLD2
# data
df_o = CSV.read("projects/RbQ10/data/Synthetic4BookChap.csv", DataFrame, normalizenames=true, missingstring = "NA", dateformat="yyyy-mm-ddTHH:MM:SSZ") # /Net/Groups/BGI/scratch/bahrens/DataBookchapter
# some pre-processing
dfall = copy(df_o)

rename!(dfall, :TA => :Temp) 
rename!(dfall, :RECO_syn => :R_soil)  # rename as in hybrid model

cols_to_select = [:Temp, :R_soil, :SW_POT_sm, :SW_POT_sm_diff]

df = select(dfall, cols_to_select...)  # select only the relevant columns
dropmissing!(df)

ds_keyed = to_keyedArray(Float32.(df)) # predictors + forcing

df_pr = df[:, [:Temp, :SW_POT_sm, :SW_POT_sm_diff]]
#rename!(df_pr, :Temp => :Temp_pr) # rename predictors
ds_pr = to_keyedArray(Float32.(df_pr)) # predictors

# TODO check effect of normailization => I had the feeling this was performing worse in retrieving the synthetic Q10. My intuition is that without normalization, we first have to train the Q10 and the NN has to first get to the right order of magnitude which maybe beneficial
#import Flux
#ds_pr = Float32.(Flux.normalise(ds_pr, dims = 1))

df_t_f = df[:, [:R_soil]] # target + forcing                                                                                        
ds_t_f = to_keyedArray(Float32.(df_t_f)) # target + forcing

ds = cat(ds_t_f, ds_pr, dims = 1) # combine target and predictors
axiskeys(ds)
# Define neural network
NN = Lux.Chain(Dense(3, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 1));
# instantiate Hybrid Model
RbQ10 = RespirationRbQ10(NN, (:SW_POT_sm, :SW_POT_sm_diff, :Temp), (:R_soil, ), (:Temp,), 2.5f0) # ? do different initial Q10s
# train model
out = train(RbQ10, ds, (:Q10, ); nepochs=10, batchsize=512, opt=Adam(0.01));

## Plotting results
series(out.ps_history; axis=(; xlabel = "epoch", ylabel=""))

# Test with LBFGS - I had good success with the hybrid example with that optimiser before.

# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ps, st = LuxCore.setup(Random.default_rng(), RbQ10)
ds = ds .|> Float64
ds_p_f, ds_t = EasyHybrid.prepare_data(RbQ10, ds)
ds_t_nan = .!isnan.(ds_t)

dta = (ds_p_f, ds_t, ds_t_nan)

# TODO check if minibatching is doing what is supposed to do - ncycle was used before:
# https://docs.sciml.ai/Optimization/stable/tutorials/minibatch/
(x_train, y_train, nan_train), (x_val, y_val, nan_val) = splitobs(dta; at=0.8, shuffle=false)
dataloader = DataLoader((x_train, y_train, nan_train), batchsize=512, shuffle=true);

# wrap loss function to get arguments as required by Optimization.jl
ls2 = (p, data) -> EasyHybrid.lossfn(RbQ10, data[1], (data[2], data[3]), p, st, LoggingLoss())[1]

# convert to Float64 for optimization
ps_ca = ComponentArray(ps) .|> Float64
ls2(ps_ca, dta)

# Define optimization problem
opt_func = OptimizationFunction(ls2, Optimization.AutoZygote())
opt_prob = OptimizationProblem(opt_func, ps_ca, dataloader)

# TODO check if we should do minibatching here, full-batch at the moment
epochs = 10
n_minibatches = length(dataloader)
Q10s = Float64[]
function callback(state, l)
    push!(Q10s, state.u.Q10[1])
    state.iter % n_minibatches == 0 && println("Epoch: $(state.iter/n_minibatches), Loss: $(l)")
    return l < 1e-8 ## Terminate if loss is small
end

res_adam = solve(opt_prob, Optimisers.Adam(0.01); callback, epochs)
ls2(res_adam.u, dta)

opt_prob = remake(opt_prob; u0=res_adam.u)

# check what maxiters is - is it equivalent to epoch?
opt_prob = remake(opt_prob; u0=res_adam.u, p = dta) # TODO implemnt minibatch - it's fullbatch for now
res_lbfgs = solve(opt_prob, Optimization.LBFGS(); callback, maxiters=200)
ls2(res_lbfgs.u, dta)


lines(Q10s; axis = (; xlabel = "epoch", ylabel="Q10", title="Q10 during training"))

res_lbfgs.u.Q10
