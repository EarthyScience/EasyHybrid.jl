mutable struct HybridModel13{D,T1,T2,T3,P,T4,T5} <: LuxCore.AbstractLuxContainerLayer{
    (:NN, :predictors, :forcing, :targets, :mech_fun, :params, :nn_names)
}
    NN
    predictors
    forcing
    targets
    mech_fun
    params
    nn_names

    function HybridModel13(
        NN::D,
        predictors::T1,
        forcing::T2,
        targets::T3,
        mech_fun::P,
        params::T4,
        nn_names::T5
    ) where {D,T1,T2,T3,P,T4,T5}
        # canonicalize to Vector{Symbol}
        predictors_v = collect(predictors)
        forcing_v    = collect(forcing)
        targets_v    = collect(targets)
        nn_names_v   = collect(nn_names)

        # validate nn_names ⊆ param keys
        all_names = collect(keys(params.table.axes[1]))
        @assert all(n in all_names for n in nn_names_v) "nn_names must be subset of params keys"

        return new{D,T1,T2,T3,T4,T5,P}(
          NN,
          predictors_v,
          forcing_v,
          targets_v,
          mech_fun,
          params,
          nn_names_v
        )
    end
end


# ───────────────────────────────────────────────────────────────────────────
# 1) New HybridModel that holds FXWParams10 directly
mutable struct HybridModel15
    NN           :: Chain
    predictors   :: Vector{Symbol}
    forcing      :: Vector{Symbol}
    targets      :: Vector{Symbol}
    mech_fun     :: Function
    params       :: FXWParams10          # now hold the full ComponentMatrix
    nn_names     :: Vector{Symbol}       # which rows come from the NN
    global_names :: Vector{Symbol}       # which rows come from the global parameters
end

# 2) Constructor: dispatch on FXWParams10 instead of separate pieces
function constructHybridModel8(
    NN,
    predictors,
    forcing,
    targets,
    mech_fun,
    params,
    nn_names
)
    all_names = collect(keys(params.table.axes[1]))
    @assert all(n in all_names for n in nn_names) "nn_names ⊆ param_names"
    NN = Chain(Dense(length(predictors), 16, relu ), Dense(16, length(nn_names)))
    global_names = [ n for n in all_names if !(n in nn_names) ]
    return HybridModel15(NN, predictors, forcing, targets, mech_fun, params, nn_names, global_names)
end

# ───────────────────────────────────────────────────────────────────────────
# 3) Initial parameters: scalars come from params.table[:, :default]
function LuxCore.initialparameters(rng::AbstractRNG, m::HybridModel15)
    ps_nn, _ = LuxCore.setup(rng, m.NN)
    # start with the NN weights
    nt = (; ps = ps_nn)
    # then append each global parameter (not in nn_names) as a 1‐vector of Float32
    all_names    = collect(keys(m.params.table.axes[1]))
    global_names = setdiff(all_names, m.nn_names)
    for g in global_names
        default_val = m.params.table[g, :default]
        nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
    end
    return nt
end

function LuxCore.initialstates(rng::AbstractRNG, m::HybridModel15)
    _, st_nn = LuxCore.setup(rng, m.NN)
    return (; st = st_nn)
end

# ───────────────────────────────────────────────────────────────────────────
# 4) Forward pass: exactly as before, but drop init_globals
function (m::HybridModel15)(ds_k, ps, st)
    # 1) get features
    p     = ds_k(m.predictors) 
    x     = Array(ds_k(m.forcing))[1, :]

    # 2) run NN → B×P
    nn_out, st = LuxCore.apply(m.NN, p, ps.ps, st.st)

    # 3) split into NamedTuple of NN‐driven parameters
    nn_cols = eachrow(nn_out)
    nn_ps   = NamedTuple(zip(m.nn_names, nn_cols))

    # 4) globals were stored in ps via initialparameters
    # 1) make a concrete tuple of values
    vals = Tuple(ps[g] for g in m.global_names)   # Tuple{Vector{Float32},…}

    # 2) explicitly name the fields and types
    glob_ps = NamedTuple{Tuple(m.global_names), Tuple{typeof.(vals)...}}(vals)

    # 5) gather args
    nn_args   = Tuple(getfield(nn_ps, n) for n in m.nn_names)
    glob_args = Tuple(getfield(glob_ps, g) for g in m.global_names)

    #println(nn_args)
    #println(glob_args)

    all_args = (nn_args..., glob_ps...)
    #println(all_args)           

    # 6) physics
    y_pred = m.mech_fun(x, all_args...)

        out = (;θ = y_pred,)


    return out, (; st)
end

# ───────────────────────────────────────────────────────────────────────────
# 5) Example wiring

# your FXWParams10 from before
p = FXWParams10()  

ca2 = copy(p.table)

targets = [:θ]
forcing = [:h]

# build the hybrid model: here NN does nothing and drives no parameters
hm = constructHybridModel8(
  NN,                    # dummy NN
  predictors,               # predictors
  forcing,                  # forcing
  targets,                   # target
  mFXW_theta,               # your physics function
  p,                         # FXWParams10 holds the bounds/defaults
  [:m]                   # nn_names (empty here)
)

hm.targets

# init
ps = LuxCore.initialparameters(Random.default_rng(), hm)
st = LuxCore.initialstates(Random.default_rng(), hm)

out, st2 = hm(ds_keyed[:,1:10], ps, st)

out.θ

#mFXW_theta(Array(ds_keyed(:h,1:10)), out.θ_s, out.h_r, out.h_0, out.α, out.n, out.m)

typeof(out)

## legacy
# ? test lossfn
ps, st = LuxCore.setup(Random.default_rng(), hm)
# the Tuple `ds_p, ds_t` is later used for batching in the `dataloader`.
ds_p_f, ds_t = EasyHybrid.prepare_data(hm, ds_keyed[:,1:10])
ds_t_nan = .!isnan.(ds_t)

function EasyHybrid.lossfn(HM::HybridModel15, x, (y_t, y_nan), ps, st, logging::LoggingLoss)
    targets = HM.targets
    ŷ, y, y_nan = EasyHybrid.get_predictions_targets(HM, x, (y_t, y_nan), ps, st, targets)
    if logging.train_mode
        return EasyHybrid.compute_loss(ŷ, y, y_nan, targets, logging.training_loss, logging.agg)
    else
        return EasyHybrid.compute_loss(ŷ, y, y_nan, targets, logging.loss_types, logging.agg)
    end
end

ls = EasyHybrid.lossfn(hm, ds_p_f, (ds_t, ds_t_nan), ps, st, LoggingLoss())

tout = train(hm, ds_keyed, (); nepochs=100, batchsize=512, opt=Adam(0.01), file_name = "o_SWRC.jld2")







