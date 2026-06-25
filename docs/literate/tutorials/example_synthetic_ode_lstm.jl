# # ODE-LSTM Hybrid Model with EasyHybrid.jl
#
# This tutorial demonstrates how to couple an LSTM with an ODE using EasyHybrid.
# The LSTM predicts time-varying basal respiration `rb`, while a process-based
# one-pool carbon model evolves the carbon state `C` via `dC = RECO - GPP`.
# The ODE state feeds back into the LSTM at every timestep.
#
# Compare with `example_synthetic_lstm.jl` which uses the same RbQ10 process
# model but without an ODE state variable.
#
# ## 1. Load Packages

using Pkg
Pkg.activate("docs")
Pkg.develop(path = pwd())
Pkg.instantiate()

using EasyHybrid
using AxisKeys
using DimensionalData

# ## 2. Data Loading and Preprocessing

df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc");
df = df[1:1000, :];
first(df, 5);

# ## 3. Define the Process-Based ODE Step Function
#
# The user writes this exactly like a normal EasyHybrid mechanistic model,
# but with an **ODE state** `C` as input and a **derivative** `dC` in the output.
# The LSTM will predict `rb` at each timestep; `Q10` is a global parameter.

"""
    mOnePool_step(; C, rb, Q10, ta, tref=15.0f0)

Single-pool carbon ODE step.  Returns the derivative `dC` and observable `reco`.

- `C`:   carbon pool state [gC/m²]
- `rb`:  basal respiration rate (predicted by LSTM) [µmol/m²/s]
- `Q10`: temperature sensitivity (global parameter) [-]
- `ta`:  air temperature [°C]
"""
function mOnePool_step(; C, rb, Q10, ta, tref = 15.0f0)
    reco = rb .* C .* Q10 .^ (0.1f0 .* (ta .- tref))
    dC   = .- reco
    return (; dC, reco, Q10, rb, C)
end

# If you only need the derivative, you can "subset" the output in a few ways:
#
# - Access the named tuple field directly:
#   `dC = mOnePool_step(; C, rb, Q10, ta).dC`
#
# - Destructure only the field you care about:
#   `(; dC) = mOnePool_step(; C, rb, Q10, ta)`
#
# - Or define a small wrapper (handy when passing a function around):
#
function mOnePool_dC(; C, rb, Q10, ta, tref = 15.0f0)
    return mOnePool_step(; C, rb, Q10, ta, tref).dC
end

function addODEProblem(model, u0, t, p, derivs)
    probm = ODEProblem(fMicrobialModel, um0, tspan, pdefault)
end


# ## 4. Define Model Parameters
#
# Same format as the non-ODE hybrid: `(default, lower_bound, upper_bound)`.
# The initial ODE state `C` is a normal parameter — include it here with bounds.
# Put it in `global_param_names` to make it trainable, or leave it out to
# have it land in `fixed_param_names` (frozen at its default).

parameters = (
    rb  = (3.0f0, 0.0f0, 13.0f0),
    Q10 = (2.0f0, 1.0f0, 4.0f0),
    C   = (100.0f0, 10.0f0, 500.0f0),
)

# ## 5. Configure Model Components

forcing    = [:ta]
predictors = [:sw_pot, :dsw_pot]
target     = [:reco]

global_param_names = [:Q10]
lstm_param_names = Vector{Symbol}()

# ## 6. Construct the ODE-LSTM Hybrid Model
#
# `constructHybridODE` is the ODE counterpart of `constructHybridModel`.
# The only new arguments are `state` / `deriv` (which fields in the step output
# are the ODE state and its derivative) and `hidden_dims` for the LSTM.

hode = constructHybridODE(
    predictors,
    forcing,
    target,
    mOnePool_step,
    parameters,
    lstm_param_names,
    global_param_names;
    hidden_dims = 16,
    state = :C,
    deriv = :dC,
    scale_nn_outputs = true,
)

# ## 7. Data Preparation (under the hood)
#
# The data pipeline is identical to the LSTM case — `prepare_data` +
# `split_into_sequences` produce 3D tensors `(features, time, batch)`.

pref_array_type = :DimArray
input_window  = 10
output_window = 1
output_shift  = 1

sdf = split_data(
    df, hode;
    sequence_kwargs = (;
        input_window  = input_window,
        output_window = output_window,
        output_shift  = output_shift,
        lead_time     = 0,
    ),
    array_type = pref_array_type,
);

(x_train, y_train), (x_val, y_val) = sdf;
x_train

# Quick sanity check: run the model forward once.
ps, st = Lux.setup(Random.default_rng(), hode);
train_dl = EasyHybrid.DataLoader((x_train, y_train); batchsize = 32);
x_first = first(train_dl)[1]
frun = hode(x_first, ps, st);
frun[1].reco
frun[1].C
frun[1].dC
frun[1].Q10
frun[1].rb
# ## 8. Train the ODE-LSTM Hybrid Model
#
# Uses the same `train` function and configuration objects as every other
# EasyHybrid model. The only difference is `DataConfig.sequence_length` which
# triggers the windowing pipeline.

out_ode = train(
    hode,
    df;
    train_cfg = EasyHybrid.TrainConfig(
        nepochs       = 2,
        batchsize     = 128,
        opt           = RMSProp(0.01),
        training_loss = :nseLoss,
        loss_types    = [:nse],
        plotting      = false,
        show_progress = false,
    ),
    data_cfg = EasyHybrid.DataConfig(
        sequence_length       = input_window,
        sequence_output_window = output_window,
        sequence_output_shift  = output_shift,
        sequence_lead_time     = 0,
        array_type             = pref_array_type,
    ),
);

out_ode.val_obs_pred

# ## 9. Static NN for Initial Conditions
#
# Instead of making `C₀` a single trainable scalar (`global_param_names`), you
# can let a dedicated feedforward neural network predict the initial carbon pool
# from site/window features.  This is useful when the initial condition should
# vary across sites or depend on auxiliary features like soil moisture.
#
# The `static_predictors` keyword tells `constructHybridODE` which parameters
# get their own per-window NN, and which input columns those NNs see.  Parameters
# listed in `static_predictors` are automatically removed from `global_param_names`
# (you should not list them there) and are predicted *before* the time loop.

hode_static = constructHybridODE(
    predictors,                         # LSTM inputs (unchanged)
    forcing,                            # forcing (unchanged)
    target,                             # targets (unchanged)
    mOnePool_step,
    parameters,
    [:rb],                              # LSTM-predicted params
    [:Q10];                             # global_param_names (C no longer here!)
    hidden_dims = 16,
    state = :C,
    deriv = :dC,
    scale_nn_outputs = true,
    static_predictors = (; C = [:sw_pot, :dsw_pot]),  # static NN for C₀
    static_hidden_layers = (; C = [8, 8]),
)
hode_static

# Quick sanity check — the model should run exactly like before.
ps2, st2 = Lux.setup(Random.default_rng(), hode_static);
frun2 = hode_static(x_first, ps2, st2);
frun2[1].reco

# Train with the static-NN variant
out_ode_static = train(
    hode_static,
    df;
    train_cfg = EasyHybrid.TrainConfig(
        nepochs       = 100,
        batchsize     = 128,
        opt           = RMSProp(0.01),
        training_loss = :nseLoss,
        loss_types    = [:nse],
        plotting      = false,
        show_progress = false,
        model_name    = "mOnePool_ode_lstm_static_C0",
    ),
    data_cfg = EasyHybrid.DataConfig(
        sequence_length       = input_window,
        sequence_output_window = output_window,
        sequence_output_shift  = output_shift,
        sequence_lead_time     = 0,
        array_type             = pref_array_type,
    ),
);

out_ode.best_loss
out_ode_static.best_loss