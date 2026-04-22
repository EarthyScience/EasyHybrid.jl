using EasyHybrid

ds = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc")
ds = ds[1:20000, :]  # Use subset for faster execution
first(ds, 5)

function RbQ10(;ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end
parameters = (
    rb  = (3.0f0, 0.0f0, 13.0f0),  # Basal respiration [μmol/m²/s]
    Q10 = (2.0f0, 1.0f0, 4.0f0),   # Temperature sensitivity - describes factor by which respiration is increased for 10 K increase in temperature [-]
)

forcing = [:ta]                    # Forcing variables (temperature)
predictors = [:sw_pot, :dsw_pot]   # Predictor variables (solar radiation)
target = [:reco]                   # Target variable (respiration)

global_param_names = [:Q10]        # Global parameters (same for all samples)
neural_param_names = [:rb]         # Neural network predicted parameters

hybrid_model = constructHybridModel(
    predictors,               # Input features
    forcing,                  # Forcing variables
    target,                   # Target variables
    RbQ10,                    # Process-based model function
    parameters,               # Parameter definitions
    neural_param_names,       # NN-predicted parameters
    global_param_names,       # Global parameters
    hidden_layers = [16, 16], # Neural network architecture
    activation = swish,       # Activation function
    scale_nn_outputs = true,  # Scale neural network outputs
    input_batchnorm = true    # Apply batch normalization to inputs
)

using ProfileView

# @time train(
#     hybrid_model,
#     ds,
#     ();
#     nepochs = 1,               # Number of training epochs
#     batchsize = 512,             # Batch size for training
#     opt = RMSProp(0.001),        # Optimizer and learning rate
#     monitor_names = [:rb, :Q10], # Parameters to monitor during training
#     yscale = identity,           # Scaling for outputs
#     patience = 30,               # Early stopping patience
#     show_progress=false,
#     model_name="RbQ10",
#     save_training=false
# );

using Random

Random.seed!(124)
data_cfg = EasyHybrid.DataConfig()
train_cfg = EasyHybrid.TrainConfig(;batchsize=512)

((x_train, forcings_train), y_train), ((x_val, forcings_val), y_val) = prepare_splits(ds, hybrid_model, data_cfg)

mask_train, _ = EasyHybrid.valid_mask(y_train)
mask_val, _ = EasyHybrid.valid_mask(y_val)
loader = EasyHybrid.build_loader(x_train, forcings_train, y_train, mask_train, train_cfg)
ps, st, train_state = EasyHybrid.init_model_state(hybrid_model, train_cfg)

@time EasyHybrid.init_model_state(hybrid_model, train_cfg);

init = EasyHybrid.compute_initial_state(hybrid_model, x_train, forcings_train, y_train, mask_train, x_val, forcings_val, y_val, mask_val, ps, st, train_cfg)
stopper = EasyHybrid.EarlyStopping(init.l_val, ps, st, train_cfg)

ps = ps |> train_cfg.gdev
st = st |> train_cfg.gdev
train_state = train_state |> train_cfg.gdev


loss_fn = EasyHybrid.build_loss_fn(hybrid_model, train_cfg)
x, y = first(loader)

(x_col, y_col) = EasyHybrid.collect_dim_data(x, y, train_cfg);

@time hybrid_model(x_col, ps, LuxCore.testmode(st));

@code_warntype hybrid_model(x_col, ps, LuxCore.testmode(st));


using Lux

@time gs, loss, stats, train_state = Lux.Training.single_train_step!(
    train_cfg.autodiff_backend,
    loss_fn,
    (x_col, y_col),
    train_state;
    return_gradients = true
    );

using Metal

ps = ps |> gpu_device();
st = st |> gpu_device();
train_state = train_state |> gpu_device();

x_col = x_col |> gpu_device();
y_col = y_col |> gpu_device();

@time gs, loss, stats, train_state = Lux.Training.single_train_step!(
    train_cfg.autodiff_backend,
    loss_fn,
    (x_col, y_col),
    train_state;
    return_gradients = true
    );

@time hybrid_model(x_col, ps, LuxCore.testmode(st));

@code_warntype hybrid_model(x_col, ps, LuxCore.testmode(st));

