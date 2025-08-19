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
[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

# Getting Started

This page demonstrates how to use `EasyHybrid` to create a hybrid model for ecosystem respiration. You will become familiar with the following concepts:

::: tip Key concepts

1. **Process-based Model**: The `RbQ10` function represents a classical Q10 model for respiration with base respiration `rb` and `Q10` which describes the factor by respiration is increased for a 10 K change in temperature.
2. **Neural Network**: Learns to predict the basal respiration parameter `rb` from environmental conditions.
3. **Hybrid Integration**: Combines the neural network predictions with the process-based model to produce final outputs.
4. **Parameter Learning**: Some parameters (like `Q10`) can be learned globally, while others (like `rb`) are predicted per sample.

:::

The framework automatically handles the integration between neural networks and mechanistic models, making it easy to leverage both data-driven learning and domain knowledge.

## Installation

Install [Julia v1.10](https://julialang.org/downloads/) or above. `EasyHybrid.jl` is available through the Julia package manager. You can enter it by pressing `]` in the `REPL` and then typing `add EasyHybrid`. Alternatively, you can also do

```julia
import Pkg
Pkg.add("EasyHybrid")
```

## Quickstart

### 1. Setup and Data Loading

Load package and synthetic dataset

```@example quick_start_complete
using EasyHybrid

ds = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc")
ds = ds[1:20000, :]  # Use subset for faster execution
first(ds, 5)
```

### 2. Define the Process-based Model

RbQ10 model: Respiration model with Q10 temperature sensitivity

```@example quick_start_complete
function RbQ10(;ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end
```

### 3. Configure Model Parameters

Parameter specification: (default, lower_bound, upper_bound)

```@example quick_start_complete
parameters = (
    rb  = (3.0f0, 0.0f0, 13.0f0),  # Basal respiration [μmol/m²/s]
    Q10 = (2.0f0, 1.0f0, 4.0f0),   # Temperature sensitivity - describes factor by which respiration is increased for 10 K increase in temperature [-]
)
```

### 4. Construct the Hybrid Model

Define input variables

```@example quick_start_complete
forcing = [:ta]                    # Forcing variables (temperature)
predictors = [:sw_pot, :dsw_pot]   # Predictor variables (solar radiation)
target = [:reco]                   # Target variable (respiration)
```

Parameter classification as global, neural or fixed (difference between global and neural)

```@example quick_start_complete
global_param_names = [:Q10]        # Global parameters (same for all samples)
neural_param_names = [:rb]         # Neural network predicted parameters
```

Construct hybrid model

```@example quick_start_complete
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
```

### 5. Train the Model

```@example quick_start_complete
out = train(
    hybrid_model, 
    ds, 
    (); 
    nepochs = 100,               # Number of training epochs
    batchsize = 512,             # Batch size for training
    opt = RMSProp(0.001),        # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,           # Scaling for outputs
    patience = 30,               # Early stopping patience
    show_progress=false,
)
```

### 6. Check Results

Evolution of train and validation loss

```@example quick_start_complete
using CairoMakie
EasyHybrid.plot_loss(out, yscale = identity)
```

Check results - what do you think - is it the true Q10 used to generate the synthetic dataset?

```@example quick_start_complete
out.train_diffs.Q10
``` 

Quick scatterplot - dispatches on the output of train

```@example quick_start_complete
EasyHybrid.poplot(out)
```

## More Examples

Check out the `projects/` directory for additional examples and use cases. Each project demonstrates different aspects of hybrid modeling with EasyHybrid.