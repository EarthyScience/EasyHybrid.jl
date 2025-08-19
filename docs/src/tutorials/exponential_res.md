```@raw html
---
authors:
  - name: Markus Reichstein
    avatar: https://raw.githubusercontent.com/EarthyScience/EasyHybrid.jl/72c2fa9df829d46d25df15352a4b728d2dbe94ed/docs/src/assets/markus_reichstein.png
    link: https://www.bgc-jena.mpg.de/en/bgi/home
  - name: Lazaro Alonso
    avatar: https://avatars.githubusercontent.com/u/19525261?v=4
    platform: github
    link: https://lazarusa.github.io
  - name: Bernhard Ahrens
    avatar: https://raw.githubusercontent.com/EarthyScience/EasyHybrid.jl/72c2fa9df829d46d25df15352a4b728d2dbe94ed/docs/src/assets/Bernhard_Ahrens.png
    link: https://www.bgc-jena.mpg.de/en/bgi/miss
---

<Authors />
```

[![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

# Soil respiration

```@example expo
using EasyHybrid
using CairoMakie
```

## Create synthetic data

Exponential temperature relationship

```math
Resp = Resp0 * \exp{(k*T)};
```

```math
Resp0 = f(SM)
```

```@example expo
using Random
Random.seed!(2314)

T = rand(500) .* 40 .- 10      # Random temperature
SM = rand(500) .* 0.8 .+ 0.1   # Random soil moisture
SM_fac = exp.(-8.0*(SM .- 0.6) .^ 2)
Resp0 = 1.1 .* SM_fac # Base respiration dependent on soil moisture
Resp = Resp0 .* exp.(0.07 .* T)
Resp_obs = Resp .+ randn(length(Resp)) .* 0.05 .* mean(Resp);  # Add some noise
nothing # hide
```

```@example expo
df = DataFrame(; T, SM, SM_fac, Resp0, Resp, Resp_obs)
first(df, 5)
```

## Define the Process-based Model

```@example expo
function Expo_resp_model(;T, Resp0, k)
    Resp_obs = Resp0 .* exp.(k .* T)
    return (; Resp_obs, Resp0, k)
end;
nothing # hide
```

:::  tip about Expo_resp_model

Mechanistic model for soil respiration based on exponential temperature response

Mathematical Model: `Resp = Resp0 * exp(k * T)`. Implements Arrhenius-type temperature dependence for biological processes (true connection?).

**Arguments**

- `T` : Air or soil temperature [°C]
- `Resp0` : Basal respiration rate - which is equal to Resp at T=0°C [μmol CO₂ m⁻² s⁻¹]
- `k` : Temperature sensitivity parameter [°C⁻¹], typical range 0.05-0.15

**Returns**

- `Resp_obs` : Predicted soil respiration [μmol CO₂ m⁻² s⁻¹]
- `Resp0` : Basal respiration (passed through)
- `k` : Temperature sensitivity (passed through)

**Scientific Context**

Models temperature dependence of soil CO₂ efflux for carbon cycle studies.
Exponential relationship reflects enzyme kinetics in microbial decomposition.

:::

## Configure Model Parameters

```@example expo
parameters = (
    # name: (default, lower_bound, upper_bound) # Description
    k     = (0.01f0, 0.0f0, 0.2f0),  # Exponent
    Resp0 = (2.0f0,  0.0f0, 8.0f0),  # Basal respiration [μmol/m²/s]
);
nothing # hide
```

## Construct the Hybrid Model

```@example expo
targets = [:Resp_obs]
forcings = [:T]
predictors = (Resp0=[:SM],);
nothing # hide
```
 
Define global parameters (none for this model, Q10 is fixed)

```@example expo
global_param_names = [:k]
```

```@example expo
hybrid_model = constructHybridModel(
    predictors,
    forcings,
    targets,
    Expo_resp_model,
    parameters,
    global_param_names,
    scale_nn_outputs=false, # TODO `true` also works with good lower and upper bounds
    hidden_layers = [16, 16],
    activation = sigmoid,
    input_batchnorm = true
)
```

## Train the Model

```@example expo
out =  train(hybrid_model, df, (:k,); nepochs=300, batchsize=64,
    opt=AdamW(0.01, (0.9, 0.999), 0.01), loss_types=[:mse, :nse],
    training_loss=:nse, random_seed=123, yscale = identity,
    monitor_names=[:Resp0, :k],
    show_progress=false,
    hybrid_name="expo_response"
    );
nothing # hide
```

```@raw html
<video src="../training_history_expo_response.mp4" controls="controls" autoplay="autoplay"></video>
```