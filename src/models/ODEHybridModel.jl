export ODEHybridModel, constructHybridODE

using Lux: LSTMCell

"""
    ODEHybridModel

Hybrid model that couples an LSTM with a process-based ODE step function,
optionally augmented by static (per-window) neural networks for parameters
like the initial ODE state.

**Two kinds of NN-predicted parameters:**

| Kind | Architecture | Runs | Example |
|------|-------------|------|---------|
| LSTM params | shared LSTM → Dense | per-timestep, inside the loop | `rb` (basal respiration) |
| Static NN params | independent feedforward NNs | once per window, before the loop | `C` (initial carbon pool) |

The LSTM receives `[predictors; ODE_state]` at each step (C feedback).
Static NNs receive the first timestep of their input features and produce
a scalar per sample.  If the ODE `state_name` (e.g. `:C`) is among the
static NN params, its output is used as the initial condition C₀.

# Fields
- `lstm_cell`, `proj`: LSTM + projection for time-varying params
- `static_NNs`: `NamedTuple` of `Chain`s, one per static neural param (empty if none)
- `static_predictors`: `NamedTuple` mapping each static param → its input feature names
- `mechanistic_model`: user function `f(; C, rb, Q10, ...) → (; dC, reco, ...)`
- `parameters`: `ParameterContainer` with bounds for scaling
- `predictors`: LSTM input feature names
- `forcing`, `targets`: same role as in `SingleNNHybridModel`
- `lstm_param_names`: params predicted per-timestep by the LSTM
- `static_nn_param_names`: params predicted per-window by static NNs
- `global_param_names`, `fixed_param_names`: non-neural params
"""
struct ODEHybridModel{LC, P, SNN, F, PM <: AbstractHybridModel} <: LuxCore.AbstractLuxContainerLayer{(:lstm_cell, :proj)}
    lstm_cell::LC
    proj::P
    static_NNs::SNN
    static_predictors::NamedTuple
    mechanistic_model::F
    parameters::PM
    predictors::Vector{Symbol}
    forcing::Vector{Symbol}
    targets::Vector{Symbol}
    lstm_param_names::Vector{Symbol}
    static_nn_param_names::Vector{Symbol}
    global_param_names::Vector{Symbol}
    fixed_param_names::Vector{Symbol}
    scale_nn_outputs::Bool
    start_from_default::Bool
    state_name::Symbol
    deriv_name::Symbol
    n_state::Int
    config::NamedTuple
end

"""
    constructHybridODE(predictors, forcing, targets, mechanistic_model, parameters,
                       lstm_param_names, global_param_names; kwargs...)

Construct an `ODEHybridModel` — the ODE counterpart of `constructHybridModel`.

The user writes the mechanistic model as a plain Julia function with keyword arguments,
exactly like `RbQ10`, but returning a derivative field (e.g. `dC`) in addition to
observable outputs.

# Example — LSTM only (all neural params are time-varying)
```julia
model = constructHybridODE(
    [:sw_pot, :dsw_pot],          # predictors (LSTM input)
    [:SW_IN, :TA],                # forcing
    [:NEE],                       # targets
    mOnePool_step,
    (rb = (3f0, 0f0, 13f0), Q10 = (2f0, 1f0, 4f0), C = (100f0, 10f0, 500f0)),
    [:rb],                        # lstm_param_names
    [:Q10, :C];                   # global_param_names (C₀ trainable scalar)
    hidden_dims = 16,
    state = :C, deriv = :dC,
)
```

# Example — LSTM + static NN for initial C
```julia
model = constructHybridODE(
    [:sw_pot, :dsw_pot],          # LSTM predictors
    [:SW_IN, :TA],                # forcing
    [:NEE],                       # targets
    mOnePool_step,
    (rb = (3f0, 0f0, 13f0), Q10 = (2f0, 1f0, 4f0), C = (100f0, 10f0, 500f0)),
    [:rb],                        # lstm_param_names
    [:Q10];                       # global_param_names
    hidden_dims = 16,
    state = :C, deriv = :dC,
    static_predictors = (; C = [:soil_moisture, :clay_fraction]),
    static_hidden_layers = (; C = [8, 8]),
)
```

# Keyword Arguments
- `hidden_dims::Int = 16`: LSTM hidden state size
- `n_state::Int = 1`: dimensionality of ODE state
- `state::Symbol = :C`: name of the ODE state variable. If this name appears in `parameters`,
  the initial condition is taken from there (trainable if in `global_param_names`, fixed otherwise).
  If it appears in `static_predictors`, a dedicated NN predicts it per window.
- `deriv::Symbol = :dC`: name of the derivative in the step function output
- `scale_nn_outputs::Bool = true`: apply sigmoid scaling to NN outputs
- `start_from_default::Bool = true`: initialize global params at their default values
- `static_predictors::NamedTuple = (;)`: per-param input features for static NNs.
  Keys are parameter names (e.g. `:C`), values are `Vector{Symbol}` of input columns.
- `static_hidden_layers::Union{NamedTuple, Vector{Int}} = [8, 8]`: architecture for
  static NNs.  A `NamedTuple` gives per-NN sizing; a `Vector{Int}` is shared across all.
- `static_activation::Union{NamedTuple, Function} = tanh`: activation for static NNs.
"""
function constructHybridODE(
        predictors::Vector{Symbol},
        forcing::Vector{Symbol},
        targets::Vector{Symbol},
        mechanistic_model,
        parameters,
        lstm_param_names::Vector{Symbol},
        global_param_names::Vector{Symbol};
        hidden_dims::Int = 16,
        n_state::Int = 1,
        state::Symbol = :C,
        deriv::Symbol = :dC,
        scale_nn_outputs::Bool = true,
        start_from_default::Bool = true,
        static_predictors::NamedTuple = (;),
        static_hidden_layers::Union{NamedTuple, Vector{Int}} = [8, 8],
        static_activation::Union{NamedTuple, Function} = tanh,
        kwargs...
    )

    if !isa(parameters, AbstractHybridModel)
        parameters = build_parameters(parameters, mechanistic_model)
    end

    all_names = pnames(parameters)

    static_nn_param_names = Symbol[k for k in keys(static_predictors)]
    all_neural = unique([lstm_param_names..., static_nn_param_names...])
    @assert all(n in all_names for n in all_neural) "all neural param names must be in parameters"

    fixed_param_names = [n for n in all_names if !(n in [all_neural..., global_param_names...])]

    # ---- LSTM + projection ----
    n_pred = length(predictors)
    n_lstm_params = length(lstm_param_names)
    lstm_cell = LSTMCell(n_pred + n_state => hidden_dims)
    proj = Dense(hidden_dims => n_lstm_params)

    # ---- static NNs (one per static param, à la MultiNNHybridModel) ----
    static_NNs = (;)
    for (nn_name, preds) in pairs(static_predictors)
        in_dim = length(preds)
        out_dim = 1
        hl = static_hidden_layers isa NamedTuple ? static_hidden_layers[nn_name] : static_hidden_layers
        act = static_activation isa NamedTuple ? static_activation[nn_name] : static_activation
        nn = prepare_hidden_chain(hl, in_dim, out_dim; activation = act)
        static_NNs = merge(static_NNs, NamedTuple{(nn_name,), Tuple{typeof(nn)}}((nn,)))
    end

    config = (;
        hidden_dims, n_state, state, deriv, scale_nn_outputs, start_from_default,
        static_hidden_layers, static_activation, kwargs...
    )

    return ODEHybridModel(
        lstm_cell, proj, static_NNs, static_predictors,
        mechanistic_model, parameters,
        predictors, forcing, targets,
        lstm_param_names, static_nn_param_names,
        global_param_names, fixed_param_names,
        scale_nn_outputs, start_from_default,
        state, deriv, n_state, config
    )
end

# Keyword-argument overload
function constructHybridODE(;
        predictors, forcing, targets, mechanistic_model, parameters,
        lstm_param_names, global_param_names, kwargs...
    )
    return constructHybridODE(
        predictors, forcing, targets, mechanistic_model, parameters,
        lstm_param_names, global_param_names; kwargs...
    )
end

# ───────────────────────────────────────────────────────────────────────────
# Lux parameter / state initialization

function LuxCore.initialparameters(rng::AbstractRNG, m::ODEHybridModel)
    ps_lstm, _ = LuxCore.setup(rng, m.lstm_cell)
    ps_proj, _ = LuxCore.setup(rng, m.proj)
    nt = (; lstm_cell = ps_lstm, proj = ps_proj)

    # Static NNs
    if !isempty(m.static_nn_param_names)
        snn_ps = (;)
        for (nn_name, nn) in pairs(m.static_NNs)
            ps_nn, _ = LuxCore.setup(rng, nn)
            snn_ps = merge(snn_ps, NamedTuple{(nn_name,), Tuple{typeof(ps_nn)}}((ps_nn,)))
        end
        nt = merge(nt, (; static_NNs = snn_ps))
    end

    # Global scalars
    if !isempty(m.global_param_names)
        if m.start_from_default
            for g in m.global_param_names
                default_val = scale_single_param_minmax(g, m.parameters)
                nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
            end
        else
            for g in m.global_param_names
                random_val = rand(rng, Float32)
                nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([random_val],)))
            end
        end
    end

    return nt
end

function LuxCore.initialstates(rng::AbstractRNG, m::ODEHybridModel)
    _, st_lstm = LuxCore.setup(rng, m.lstm_cell)
    _, st_proj = LuxCore.setup(rng, m.proj)

    # Static NNs
    snn_st = (;)
    if !isempty(m.static_nn_param_names)
        for (nn_name, nn) in pairs(m.static_NNs)
            _, st_nn = LuxCore.setup(rng, nn)
            snn_st = merge(snn_st, NamedTuple{(nn_name,), Tuple{typeof(st_nn)}}((st_nn,)))
        end
    end

    # Fixed params
    fixed = (;)
    if !isempty(m.fixed_param_names)
        for f in m.fixed_param_names
            default_val = default(m.parameters)[f]
            fixed = merge(fixed, NamedTuple{(f,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
        end
    end

    return (; lstm_cell = st_lstm, proj = st_proj, static_NNs = snn_st, fixed = fixed)
end

# ───────────────────────────────────────────────────────────────────────────
# Forward pass — explicit time loop (SpiralClassifier pattern)

function (m::ODEHybridModel)(ds_k::Union{KeyedArray, AbstractDimArray}, ps, st)
    pred_3d = toArray(ds_k, m.predictors)    # (n_pred, T, B)
    T_len = size(pred_3d, 2)
    B = size(pred_3d, 3)
    ET = eltype(pred_3d)

    forc_3d = isempty(m.forcing) ? nothing : toArray(ds_k, m.forcing)

    sn = m.state_name

    # ── static NNs: run once per window, before the time loop ──
    static_kw = (;)
    static_nn_states = st.static_NNs
    if !isempty(m.static_nn_param_names)
        for (nn_name, nn) in pairs(m.static_NNs)
            preds = toArray(ds_k, collect(m.static_predictors[nn_name]))
            nn_input = preds[:, 1, :]   # first timestep → (n_feat, B)
            nn_out, st_nn = LuxCore.apply(nn, nn_input, ps.static_NNs[nn_name], static_nn_states[nn_name])
            static_nn_states = merge(static_nn_states, NamedTuple{(nn_name,), Tuple{typeof(st_nn)}}((st_nn,)))

            nn_val = nn_out[1:1, :]     # (1, B)
            if m.scale_nn_outputs
                nn_val = scale_single_param(nn_name, nn_val, m.parameters)
            end
            static_kw = merge(static_kw, (; zip([nn_name], [nn_val])...))
        end
    end

    # ── initialize ODE state ──
    # Priority: static NN > global param > fixed param > zeros
    if sn in m.static_nn_param_names
        C = static_kw[sn]  # already (1, B) from static NN
    elseif sn in m.global_param_names
        C₀_val = scale_single_param(sn, ps[sn], m.parameters)
        C = C₀_val .+ zeros(ET, m.n_state, B)
    elseif sn in m.fixed_param_names
        C₀_val = st.fixed[sn]
        C = C₀_val .+ zeros(ET, m.n_state, B)
    else
        C = zeros(ET, m.n_state, B)
    end

    # ── global params (excluding state — state is managed by the ODE loop) ──
    global_names = [g for g in m.global_param_names if g != sn]
    if !isempty(global_names)
        global_vals = Tuple(scale_single_param(g, ps[g], m.parameters) for g in global_names)
        global_kw = (; zip(global_names, global_vals)...)
    else
        global_kw = (;)
    end

    # ── fixed params (excluding state) ──
    fixed_names = [f for f in m.fixed_param_names if f != sn]
    if !isempty(fixed_names)
        fixed_vals = Tuple(st.fixed[f] for f in fixed_names)
        fixed_kw = (; zip(fixed_names, fixed_vals)...)
    else
        fixed_kw = (;)
    end

    # ── static NN params that are NOT the ODE state (constant through the loop) ──
    static_non_state_names = [n for n in m.static_nn_param_names if n != sn]
    if !isempty(static_non_state_names)
        static_non_state_kw = (; zip(static_non_state_names, [static_kw[n] for n in static_non_state_names])...)
    else
        static_non_state_kw = (;)
    end

    # ── first timestep (no carry) ──
    st_lstm = st.lstm_cell
    st_proj = st.proj

    pred_1 = pred_3d[:, 1, :]
    lstm_in_1 = vcat(pred_1, C)
    (h, carry), st_lstm = Lux.apply(m.lstm_cell, lstm_in_1, ps.lstm_cell, st_lstm)

    result_1, nn_kw_1, st_proj = _ode_inner_step(
        m, h, C, forc_3d, 1, ps, st_proj, global_kw, fixed_kw, static_non_state_kw
    )
    C = C .+ result_1[m.deriv_name]

    # Accumulate *all* mechanistic outputs (not just targets) via vcat (mutation-free for AD)
    result_names = collect(keys(result_1))
    result_trajs = NamedTuple{Tuple(result_names)}(Tuple(result_1[k] for k in result_names))
    nn_trajs  = NamedTuple{Tuple(m.lstm_param_names)}(Tuple(nn_kw_1[n] for n in m.lstm_param_names))

    # ── remaining timesteps ──
    for t in 2:T_len
        pred_t = pred_3d[:, t, :]
        lstm_in_t = vcat(pred_t, C)
        (h, carry), st_lstm = Lux.apply(m.lstm_cell, (lstm_in_t, carry), ps.lstm_cell, st_lstm)

        result_t, nn_kw_t, st_proj = _ode_inner_step(
            m, h, C, forc_3d, t, ps, st_proj, global_kw, fixed_kw, static_non_state_kw
        )
        C = C .+ result_t[m.deriv_name]

        result_trajs = NamedTuple{Tuple(result_names)}(
            Tuple(vcat(result_trajs[k], result_t[k]) for k in result_names)
        )
        nn_trajs = NamedTuple{Tuple(m.lstm_param_names)}(
            Tuple(vcat(nn_trajs[n], nn_kw_t[n]) for n in m.lstm_param_names)
        )
    end

    # ── output as plain NamedTuple (time subsetting handled by compute_loss) ──
    output = result_trajs

    all_params = merge(nn_trajs, global_kw, fixed_kw, static_kw)
    output = merge(output, (; parameters = all_params))

    st_new = (; lstm_cell = st_lstm, proj = st_proj, static_NNs = static_nn_states, fixed = st.fixed)
    return output, st_new
end

"""
Inner step: project LSTM hidden → per-timestep NN params, merge with static/global/fixed, call mechanistic model.
"""
function _ode_inner_step(m::ODEHybridModel, h, C, forc_3d, t, ps, st_proj, global_kw, fixed_kw, static_non_state_kw)
    nn_raw, st_proj = Lux.apply(m.proj, h, ps.proj, st_proj)

    n_nn = length(m.lstm_param_names)
    if m.scale_nn_outputs
        nn_scaled = ntuple(i -> scale_single_param(m.lstm_param_names[i], nn_raw[i:i, :], m.parameters), n_nn)
    else
        nn_scaled = ntuple(i -> nn_raw[i:i, :], n_nn)
    end
    nn_kw = (; zip(m.lstm_param_names, nn_scaled)...)

    if forc_3d !== nothing
        forc_t = forc_3d[:, t, :]
        forc_kw = (; zip(m.forcing, [forc_t[i:i, :] for i in 1:length(m.forcing)])...)
    else
        forc_kw = (;)
    end

    state_kw = (; zip([m.state_name], [C])...)
    all_kw = merge(nn_kw, global_kw, fixed_kw, static_non_state_kw, forc_kw, state_kw)
    result = m.mechanistic_model(; all_kw...)

    return result, nn_kw, st_proj
end

