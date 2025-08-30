```@raw html
---
authors:
  - name: Bernhard Ahrens
    avatar: https://raw.githubusercontent.com/EarthyScience/EasyHybrid.jl/72c2fa9df829d46d25df15352a4b728d2dbe94ed/docs/src/assets/Bernhard_Ahrens.png
    link: https://www.bgc-jena.mpg.de/en/bgi/miss
  - name: Lazaro Alonso
    avatar: https://avatars.githubusercontent.com/u/19525261?v=4
    platform: github
    link: https://lazarusa.github.io

---

<Authors />
```

# Getting Started


### 1. Setup and Data Loading

Load package and synthetic dataset

```@example hyperparameter_tuning
using EasyHybrid
using CairoMakie
using Hyperopt
```

```@example hyperparameter_tuning
ds = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc")
ds = ds[1:20000, :]  # Use subset for faster execution
first(ds, 5)
```

### 2. Define the Process-based Model

RbQ10 model: Respiration model with Q10 temperature sensitivity

```@example hyperparameter_tuning
function RbQ10(;ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end
```

### 3. Configure Model Parameters

Parameter specification: (default, lower_bound, upper_bound)

```@example hyperparameter_tuning
parameters = (
    rb  = (3.0f0, 0.0f0, 13.0f0),  # Basal respiration [μmol/m²/s]
    Q10 = (2.0f0, 1.0f0, 4.0f0),   # Temperature sensitivity - describes factor by which respiration is increased for 10 K increase in temperature [-]
)
```

### 4. Construct the Hybrid Model

Define input variables

```@example hyperparameter_tuning
forcing = [:ta]                    # Forcing variables (temperature)
predictors = [:sw_pot, :dsw_pot]   # Predictor variables (solar radiation)
target = [:reco]                   # Target variable (respiration)
```

Parameter classification as global, neural or fixed (difference between global and neural)

```@example hyperparameter_tuning
global_param_names = [:Q10]        # Global parameters (same for all samples)
neural_param_names = [:rb]         # Neural network predicted parameters
```

Construct hybrid model

```@example hyperparameter_tuning
hybrid_model = constructHybridModel(
    predictors,               # Input features
    forcing,                  # Forcing variables
    target,                   # Target variables
    RbQ10,                    # Process-based model function
    parameters,               # Parameter definitions
    neural_param_names,       # NN-predicted parameters
    global_param_names,       # Global parameters
    hidden_layers = [16, 16], # Neural network architecture
    activation = relu,       # Activation function
    scale_nn_outputs = true,  # Scale neural network outputs
    input_batchnorm = false    # Apply batch normalization to inputs
)
```

### 5. Train the Model

```@example hyperparameter_tuning
out = train(
    hybrid_model, 
    ds, 
    (); 
    nepochs = 100,               # Number of training epochs
    batchsize = 512,             # Batch size for training
    opt = AdamW(0.001),        # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,           # Scaling for outputs
    patience = 30,               # Early stopping patience
    show_progress=false,
    hybrid_name="before"
)
```

```@raw html
<video src="../training_history_before.mp4" controls="controls" autoplay="autoplay"></video>
```

### 6. Check Results

Evolution of train and validation loss

```@example hyperparameter_tuning
EasyHybrid.plot_loss(out, yscale = identity)
```

Check results - what do you think - is it the true Q10 used to generate the synthetic dataset?

```@example hyperparameter_tuning
out.train_diffs.Q10
``` 

Quick scatterplot - dispatches on the output of train

```@example hyperparameter_tuning
EasyHybrid.poplot(out)
```

## Hyperparameter Tuning

EasyHybrid provides built-in hyperparameter tuning capabilities to optimize your model configuration. This is especially useful for finding the best neural network architecture, optimizer settings, and other hyperparameters.

### Basic Hyperparameter Tuning

You can use the `tune` function to automatically search for optimal hyperparameters:

```@example hyperparameter_tuning
# Create empty model specification for tuning
mspempty = ModelSpec()

# Define hyperparameter search space
nhyper = 4
ho = @thyperopt for i=nhyper,
    opt = [AdamW(0.01), AdamW(0.1), RMSProp(0.001), RMSProp(0.01)],
    input_batchnorm = [true, false]
    
    hyper_parameters = (;opt, input_batchnorm)
    println("Hyperparameter run: ", i, " of ", nhyper, " with hyperparameters: ", hyper_parameters)
    
    # Run tuning with current hyperparameters
    out = EasyHybrid.tune(
        hybrid_model, 
        ds, 
        mspempty; 
        hyper_parameters..., 
        nepochs = 10, 
        plotting = false, 
        show_progress = false, 
        file_name = "test$i.jld2"
    )
    
    out.best_loss
end

# Get the best hyperparameters
ho.minimizer
printmin(ho)

# Train the model with the best hyperparameters
best_hyperp = best_hyperparams(ho)

```

### Train model with the best hyperparameters

```@example hyperparameter_tuning
# Run tuning with specific hyperparameters
out_tuned = EasyHybrid.tune(
    hybrid_model, 
    ds, 
    mspempty; 
    best_hyperp...,
    nepochs = 100,
    monitor_names = [:rb, :Q10],
    hybrid_name="after"
)

# Check the tuned model performance
out_tuned.best_loss
```

```@raw html
<video src="../training_history_after.mp4" controls="controls" autoplay="autoplay"></video>
```

### Key Hyperparameters to Tune

When tuning your hybrid model, consider these important hyperparameters:

- **Optimizer and Learning Rate**: Try different optimizers (AdamW, RMSProp, Adam) with various learning rates
- **Neural Network Architecture**: Experiment with different `hidden_layers` configurations
- **Activation Functions**: Test different activation functions (relu, sigmoid, tanh)
- **Batch Normalization**: Enable/disable `input_batchnorm` and other normalization options
- **Batch Size**: Adjust `batchsize` for optimal training performance

### Tips for Hyperparameter Tuning

- **Start with a small search space** to get a baseline understanding
- **Monitor for overfitting** by tracking validation loss  
- **Consider computational cost** - more hyperparameters and epochs increase training time

## More Examples

Check out the `projects/` directory for additional examples and use cases. Each project demonstrates different aspects of hybrid modeling with EasyHybrid.