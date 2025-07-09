# ───────────────────────────────────────────────────────────────────────────
# 1) New HybridModel that holds FXWParams10 directly
mutable struct HybridModel15
    NN           :: Chain
    predictors   :: Vector{Symbol}
    forcing      :: Vector{Symbol}
    targets      :: Vector{Symbol}
    mech_fun     :: Function
    params       :: FXWParams10          # now hold the full ComponentMatrix
    nn_names     :: Vector{Symbol}       # which parameters come from the NN
    global_names :: Vector{Symbol}       # which parameters are estimated globally
    fixed_names  :: Vector{Symbol}       # which parameters are fixed
end

# 2) Constructor: dispatch on FXWParams10 instead of separate pieces
function constructHybridModel8(
    NN,
    predictors,
    forcing,
    targets,
    mech_fun,
    params,
    nn_names,
    global_names
)
    all_names = collect(keys(params.table.axes[1]))
    @assert all(n in all_names for n in nn_names) "nn_names ⊆ param_names"
    NN = Chain(BatchNorm(length(predictors)), Dense(length(predictors), 32, tanh), Dense(32, length(nn_names), tanh))
    fixed_names = [ n for n in all_names if !(n in [nn_names..., global_names...]) ]
    return HybridModel15(NN, predictors, forcing, targets, mech_fun, params, nn_names, global_names, fixed_names)
end

# ───────────────────────────────────────────────────────────────────────────
# 3) Initial parameters: scalars come from params.table[:, :default]
function LuxCore.initialparameters(rng::AbstractRNG, m::HybridModel15)
    ps_nn, _ = LuxCore.setup(rng, m.NN)
    # start with the NN weights
    nt = (; ps = ps_nn)
    # then append each global parameter as a 1‐vector of Float32
    if !isempty(m.global_names)
        for g in m.global_names
            default_val = m.params.table[g, :default]
            nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
        end
    end
    return nt
end

function LuxCore.initialstates(rng::AbstractRNG, m::HybridModel15)
    _, st_nn = LuxCore.setup(rng, m.NN)
    # start with the NN weights
    nt = (;)
    # then append each global parameter as a 1‐vector of Float32
    if !isempty(m.fixed_names)
        for f in m.fixed_names  
            default_val = m.params.table[f, :default]
            nt = merge(nt, NamedTuple{(f,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
        end
    end
    nt = (; st = st_nn, fixed = nt)
    return nt
end

"""
    scale_single_param(name, raw_val, table)

Scale a single parameter using the sigmoid scaling function.
"""
function scale_single_param(name, raw_val, table)
    ℓ = table[name, :lower]
    u = table[name, :upper]
    return ℓ .+ (u .- ℓ) .* sigmoid.(raw_val)
end


# ───────────────────────────────────────────────────────────────────────────
# 4) Forward pass: exactly as before, but drop init_globals
function (m::HybridModel15)(ds_k, ps, st)
    # 1) get features
    p     = ds_k(m.predictors) 
    x     = Array(ds_k(m.forcing))[1, :]

    # 2) run NN → B×P

    # scale global parameters (handle empty case)
    if !isempty(m.global_names)
        global_vals = Tuple(
                scale_single_param(g, ps[g], m.params.table)
                for g in m.global_names
            )
        glob_ps = NamedTuple{Tuple(m.global_names), Tuple{typeof.(global_vals)...}}(global_vals)
    else
        glob_ps = NamedTuple()
    end

    # scale NN parameters (handle empty case)
    if !isempty(m.nn_names)
        nn_out, st_NN = LuxCore.apply(m.NN, p, ps.ps, st.st)
        nn_cols = eachrow(nn_out)
        nn_ps   = NamedTuple(zip(m.nn_names, nn_cols))
        scaled_nn_vals = Tuple(
            scale_single_param(name, nn_ps[name], m.params.table)
            for name in m.nn_names
        )
        scaled_nn_ps   = NamedTuple(zip(m.nn_names, scaled_nn_vals))
    else
        scaled_nn_ps = NamedTuple()
        st_NN = st.st
    end

    # pick fixed parameters (handle empty case)
    if !isempty(m.fixed_names)
        fixed_vals = Tuple(st.fixed[f] for f in m.fixed_names)
        fixed_ps = NamedTuple{Tuple(m.fixed_names), Tuple{typeof.(fixed_vals)...}}(fixed_vals)
    else
        fixed_ps = NamedTuple()
    end

    a = merge(scaled_nn_ps, glob_ps, fixed_ps)
  
    # 6) physics
    y_pred = m.mech_fun(x, a.θ_s, a.h_r, a.h_0, a.α, a.n, a.m)

    out = (;θ = y_pred, a = a)

    st_new = (; st = st_NN, fixed = st.fixed)

    return out, (; st = st_new)
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
  [:θ_s, :α, :n, :m],                      # nn_names
  []              # global_names
)

hm.targets

# init
ps = LuxCore.initialparameters(Random.default_rng(), hm)
st = LuxCore.initialstates(Random.default_rng(), hm)

out, st2 = hm(ds_keyed, ps, st)

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

tout = train(hm, ds_keyed[:,1:100], (); nepochs=100, batchsize=10, opt=Adam(0.01), file_name = "tour.jld2")

# Plot the NEE predictions as scatter plot
fig_θ = Figure(size=(800, 600))

# Calculate NEE statistics
θ_pred = tout.train_obs_pred[!, Symbol(string(:θ, "_pred"))]
θ_obs = tout.train_obs_pred[!, :θ]
ss_res = sum((θ_obs .- θ_pred).^2)
ss_tot = sum((θ_obs .- mean(θ_obs)).^2)
θ_modelling_efficiency = 1 - ss_res / ss_tot
θ_rmse = sqrt(mean((θ_pred .- θ_obs).^2))

ax_θ = Makie.Axis(fig_θ[1, 1], 
    title="FluxPartModel - θ Predictions vs Observations
    \n Modelling Efficiency: $(round(θ_modelling_efficiency, digits=3)) 
    \n RMSE: $(round(θ_rmse, digits=3)) μmol CO2 m-2 s-1",
    xlabel="Predicted θ", 
    ylabel="Observed θ", aspect=1)

scatter!(ax_θ, θ_pred, θ_obs, color=:purple, alpha=0.6, markersize=8)

# Add 1:1 line
max_val = max(maximum(θ_obs), maximum(θ_pred))
min_val = min(minimum(θ_obs), minimum(θ_pred))
lines!(ax_θ, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=1, label="1:1 line")

axislegend(ax_θ; position=:lt)
fig_θ







