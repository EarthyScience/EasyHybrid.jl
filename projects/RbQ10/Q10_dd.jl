# CC BY-SA 4.0
# activate the project's environment and instantiate dependencies
using Pkg
# Pkg.activate("projects/RbQ10")
# Pkg.develop(path=pwd())
# Pkg.instantiate()

# start using the package
using EasyHybrid

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
RbQ10 = RespirationRbQ10(NN, predictor_names, forcing_names, target_names, 2.5f0)
ds_p_f, ds_t = EasyHybrid.prepare_data(RbQ10, ds_keyed)
ds_t_nan = .!isnan.(ds_t)

using Zygote
ps, st = LuxCore.setup(Random.default_rng(), RbQ10)

l, backtrace = Zygote.pullback(
    (ps) -> EasyHybrid.compute_loss(
        RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st,
        EasyHybrid.LoggingLoss(training_loss = :mse, agg = sum)
    ), ps
)

grads = backtrace(l)[1]

# TODO: test DimArray inputs
using DimensionalData, ChainRulesCore
# mat = Matrix(df)'
mat = Array(Matrix(df)')
da = DimArray(mat, (Dim{:col}(Symbol.(names(df))), Dim{:row}(1:size(df, 1))))

##! new dispatch
ds_p_f = da[col = At(forcing_names ∪ predictor_names)]
ds_t = da[col = At(target_names)]
ds_t_nan = .!isnan.(ds_t)

p = ds_p_f[col = At(RbQ10.predictors)]
x = ds_p_f[col = At(RbQ10.forcing)] # don't propagate names after this

Rb, stQ10 = LuxCore.apply(RbQ10.NN, p, ps.ps, st.st);

R_soil = mRbQ10(Rb, ps.Q10, x, 15.0f0)
targets = RbQ10.targets

# EasyHybrid.get_predictions_targets(RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st, targets)
# ŷ, st_ = RbQ10(ds_p_f, ps, st)

# EasyHybrid._compute_loss(ŷ, ds_t, ds_t_nan, targets, :mse, sum)

# ls = EasyHybrid.compute_loss(RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss())


## ! DimensionalData + ChainRulesCore
# ? test compute_loss
# ps, st = LuxCore.setup(Random.default_rng(), RbQ10)

ls = EasyHybrid.compute_loss(RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss())
ls_logs = EasyHybrid.compute_loss(RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss(train_mode = false))
acc_ = EasyHybrid.evaluate_acc(RbQ10, ds_p_f, ds_t, ds_t_nan, ps, st, [:mse, :r2], :mse, sum)

using Zygote, ChainRulesCore, DimensionalData
using EasyHybrid

l, backtrace = Zygote.pullback(
    (ps) -> EasyHybrid.compute_loss(
        RbQ10, ds_p_f, (ds_t, ds_t_nan), ps, st,
        EasyHybrid.LoggingLoss(training_loss = :mse, agg = sum)
    ), ps
)

grads_dd = backtrace(l)[1]
ar = rand(3, 3)

using EasyHybrid.AxisKeys
ak = KeyedArray(ar; X = [:a, :b, :c], Y = 1:3)
grad_k = Zygote.gradient(x -> sum(x(X = :b)), ak)


# Gradient through single At selector
# ar = rand(3,3)
using Zygote, ChainRulesCore, DimensionalData
using GLMakie
ar = rand(3, 3)
A = DimArray(ar, (Y([:a, :b, :c]), X(1:3)));
grad = Zygote.gradient(x -> sum(x[Y = At(:b)]), A)

xy = EasyHybrid.split_data((ds_p_f, ds_t), 0.8, shuffle = true, rng = Random.default_rng())

EasyHybrid.get_prediction_target_names(RbQ10)

xy1 = EasyHybrid.prepare_data(RbQ10, da)

(x_train, y_train), (x_val, y_val) = EasyHybrid.split_data(da, RbQ10) # ; shuffleobs=false, split_data_at=0.8

out = train(RbQ10, da, (:Q10,); nepochs = 200, batchsize = 512, opt = Adam(0.01));
