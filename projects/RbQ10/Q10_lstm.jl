# activate the project's environment and instantiate dependencies
using Pkg
Pkg.activate("projects/RbQ10")
Pkg.develop(path=pwd())
Pkg.instantiate()

# start using the package
using EasyHybrid
using EasyHybrid.MLUtils
using Random
# for Plotting
using GLMakie
using AlgebraOfGraphics
using EasyHybrid.AxisKeys
function split_into_sequences(xin, y_target; window_size = 8)
    features = size(xin, 1)
    xdata = slidingwindow(xin, size = window_size, stride = 1)
    # get the target values corresponding to the sliding windows,
    # elements of `ydata` correspond to the last sliding window element.
    ydata = y_target[window_size:length(xdata) + window_size - 1]

    xwindowed = zeros(Float32, features, window_size, length(ydata))
    for i in 1:length(ydata)
        xwindowed[:, :, i] = getobs(xdata, i)
    end
    xwindowed = KeyedArray(xwindowed; row=xin.row, window=1:window_size, col=1:length(ydata))
    ydata = KeyedArray(Array(ydata[:,:]'); row=ds_t.row, col=1:length(ydata))
    return xwindowed, ydata
end


script_dir = @__DIR__
include(joinpath(script_dir, "data", "prec_process_data.jl"))

# Common data preprocessing
df = dfall[!, Not(:timesteps)]
ds_keyed = to_keyedArray(Float32.(df))

target_names = [:R_soil]
forcing_names = [:cham_temp_filled]
predictor_names = [:moisture_filled, :rgpot2]

# Define neural network
NN = Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 1))
# NN(rand(Float32, 2,1)) #! needs to be instantiated


# instantiate Hybrid Model
RbQ10 = RespirationRbQ10(NN, predictor_names, target_names, forcing_names, 2.5f0) # ? do different initial Q10s
# train model
out = train(RbQ10, ds_keyed, (:Q10, ); nepochs=200, batchsize=512, opt=Adam(0.01));

##
output_file = joinpath(@__DIR__, "output_tmp/trained_model.jld2")
all_groups = get_all_groups(output_file)

# ? Let's use `Recurrence` to stack LSTM cells and deal with sequences and batching at the same time!

NN_Memory = Chain(
    Recurrence(LSTMCell(2 => 6), return_sequence=true),
    Recurrence(LSTMCell(6 => 2), return_sequence=false),
    Dense(2 => 1)
)

# c_lstm = LSTMCell(4 => 6)
# ps, st = Lux.setup(rng, c_lstm)
# (y, carry), st_lstm = c_lstm(rand(Float32, 4), ps, st)
# m_lstm = Recurrence(LSTMCell(4 => 6), return_sequence=false)

rng = Random.default_rng(1234)
ps, st = Lux.setup(rng, NN_Memory)
mock_data = rand(Float32, 2, 8, 5) #! n_features, n_timesteps (window size), n_samples (batch size)
y_, st_ = NN_Memory(mock_data, ps, st)

RbQ10_Memory = RespirationRbQ10(NN_Memory, predictor_names, target_names, forcing_names, 2.5f0) # ? do different initial Q10s

## legacy
# ? test lossfn
ps, st = LuxCore.setup(Random.default_rng(), RbQ10_Memory)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_p_f, ds_t = EasyHybrid.prepare_data(RbQ10_Memory, ds_keyed)
ds_t_nan = .!isnan.(ds_t)

# 
xwindowed, ydata = split_into_sequences(ds_p_f, ds_t; window_size = 8)
ds_wt = ydata
ds_wt_nan = .!isnan.(ds_wt)
# xdata = slidingwindow(ds_p_f, size = 8, stride = 1)
# getobs(xdata, 1)
# split_test = rand(Float32, 3, 8, 5)
using EasyHybrid.AxisKeys
# A = KeyedArray(split_test; row=ds_p_f.row, window=1:8, col=1:5)

NN_Memory = Chain(
    Recurrence(LSTMCell(2 => 6), return_sequence=true),
    Recurrence(LSTMCell(6 => 2), return_sequence=false),
    Dense(2 => 1)
)

RbQ10_Memory = RespirationRbQ10(NN_Memory, predictor_names, target_names, forcing_names, 2.5f0) # ? do different initial Q10s

#? this sets up initial ps for the hybrid model version
rng = Random.default_rng(1234)
ps, st = Lux.setup(rng, RbQ10_Memory)

ls = EasyHybrid.lossfn(RbQ10_Memory, xwindowed, (ds_wt, ds_wt_nan), ps, st, LoggingLoss())

ls_logs = EasyHybrid.lossfn(RbQ10_Memory, xwindowed, (ds_wt, ds_wt_nan), ps, st, LoggingLoss(train_mode=false))


# p = A(RbQ10_Memory.predictors)
# x = Array(A(RbQ10_Memory.forcing)) # don't propagate names after this
# Rb, st = LuxCore.apply(RbQ10_Memory.NN, p, ps.ps, st.st)

