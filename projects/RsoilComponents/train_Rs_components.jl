using Pkg
Pkg.activate("projects/RsoilComponents")
Pkg.develop(path=pwd())
Pkg.instantiate()

using EasyHybrid

script_dir = @__DIR__
include(joinpath(script_dir, "data", "prec_process_data.jl"))

df = dfall[!, Not(:timesteps)]
ds_keyed = to_keyedArray(Float32.(df))

target_names = [:R_soil, :R_root, :R_myc, :R_het]

NN = Lux.Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 3));
Rsc = Rs_components(NN, (:rgpot, :moisture_filled), target_names, (:cham_temp_filled,), 2.5f0, 2.5f0, 2.5f0)

out = train(Rsc, ds_keyed, (:Q10_het, :Q10_myc, :Q10_root, ); nepochs=100, batchsize=512, opt=Adam(0.01));


# legacy
NN = Lux.Chain(Dense(2, 15, Lux.relu), Dense(15, 15, Lux.relu), Dense(15, 3));
Rsc = Rs_components(NN, (:rgpot, :moisture_filled), target_names, (:cham_temp_filled,), 2.5f0, 2.5f0, 2.5f0)
ds_p_f, ds_t = EasyHybrid.prepare_data(Rsc, ds_keyed)

ps, st = LuxCore.setup(Random.default_rng(), Rsc)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_t_nan = .!isnan.(ds_t)

ls = EasyHybrid.lossfn(Rsc, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss(train_mode=false))

out = train(Rsc, (ds_p_f, ds_t), (:Q10_het, :Q10_myc, :Q10_root, ); nepochs=100, batchsize=512, opt=Adam(0.01));