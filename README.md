# EasyHybrid.jl
<img src="docs/src/assets/logo.png" align="right" width="30%"></img>
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://earthyscience.github.io/EasyHybrid.jl/stable/)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://earthyscience.github.io/EasyHybrid.jl/dev/)
[![Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Fmonthly_downloads%2FEasyHybrid&query=total_requests&suffix=%2Fmonth&label=Downloads)](https://juliapkgstats.com/pkg/EasyHybrid)
[![CI](https://github.com/EarthyScience/EasyHybrid.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EarthyScience/EasyHybrid.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/EarthyScience/EasyHybrid.jl/blob/main/LICENSE)



> [!CAUTION]
> Work in progress

`EasyHybrid.jl` provides a simple and flexible framework for hybrid modeling, enabling the integration of neural networks with process-based models. This approach can be expressed as:

$$
\hat{y} = \mathcal{M}(h(x;\theta), z; \phi)
$$

where $\hat{y}$ denotes the predicted output of the hybrid model, $h(x;\theta)$ is a neural network with inputs $x$ and learnable parameters $\theta$, $z$ denotes forcing passed directly to the mechanistic model $\mathcal{M}(\cdot, z;\, \phi)$, which is parameterized by $\phi$. The parameters $\phi$ may be known, learned from data or fixed.


## Installation

Since `EasyHybrid.jl` is registered in the Julia General registry, it is available through the Julia package manager. You can enter it by pressing `]` in the `REPL` and then typing `add EasyHybrid`. Alternatively, you can also do

```julia
julia> using Pkg
julia> Pkg.add("EasyHybrid")
```

Start using the package:

```julia
using EasyHybrid
```

If you want to use the latest unreleased version then do

```julia
using Pkg
Pkg.add(url="https://github.com/EarthyScience/EasyHybrid.jl.git")
```

## Developing EasyHybrid

<details>
  <summary><span style="color:red"> 🛠️ ⚙️ 🚀  Click for more! 🛠️ ⚙️ 🚀 </span></summary>

Clone the repository

```sh
git clone https://github.com/EarthyScience/EasyHybrid.jl.git
```

and start using it by opening one of the `env` in `projects`, i.e. `Q10.jl`. There executing the first 4 lines should get you all needed dependencies: `shift + enter`.


Or if you are already working in a project and want to add EasyHybrid in dev mode then do

```julia
# local will clone the repository at your current directory
]dev --local https://github.com/EarthyScience/EasyHybrid.jl.git
```

</details>

## Quick Start Example

Here's a complete example demonstrating how to use EasyHybrid to create a hybrid model for ecosystem respiration. This example demonstrates the key concepts of EasyHybrid:

1. **Process-based Model**: The `RbQ10` function represents a classical Q10 model for respiration with base respiration `rb` and `Q10` which describes the factor by respiration is increased for a 10 K change in temperature
2. **Neural Network**: Learns to predict the basal respiration parameter `rb` from environmental covariates
3. **Hybrid Integration**: Combines the neural network predictions with the process-based model to produce final outputs
4. **Parameter Learning**: Some parameters, like `Q10` corresponding to $\phi$, can be learned globally, while others, like `rb` corresponding to $\theta$, are predicted per sample

The framework automatically handles the integration between neural networks and mechanistic models, making it easy to leverage both data-driven learning and domain knowledge.


### 1. Setup and Data Loading

```julia
using EasyHybrid

# Load synthetic dataset
ds = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc")
ds = ds[1:20000, :]  # Use subset for faster execution
```

### 2. Define the Process-based Model

```julia
# RbQ10 model: Respiration model with Q10 temperature sensitivity
function RbQ10(;ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end
```

### 3. Configure Model Parameters

```julia
# Parameter specification: (default, lower_bound, upper_bound)
parameters = (
    rb  = (3.0f0, 0.0f0, 13.0f0),  # Basal respiration [μmol/m²/s]
    Q10 = (2.0f0, 1.0f0, 4.0f0),   # Temperature sensitivity - describes factor by which respiration is increased for 10 K increase in temperature [-]
)
```

### 4. Construct the Hybrid Model

```julia
# Define input variables
forcing = [:ta]                    # Forcing variables (temperature)
predictors = [:sw_pot, :dsw_pot]   # Predictor variables (solar radiation)
target = [:reco]                   # Target variable (respiration)

# Parameter classification
global_param_names = [:Q10]        # Global parameters (same for all samples)
neural_param_names = [:rb]         # Neural network predicted parameters

# Construct hybrid model
hybrid_model = constructHybridModel(
    predictors,              # Input features
    forcing,                 # Forcing variables
    target,                  # Target variables
    RbQ10,                  # Process-based model function
    parameters,              # Parameter definitions
    neural_param_names,      # NN-predicted parameters
    global_param_names,      # Global parameters
    hidden_layers = [16, 16], # Neural network architecture
    activation = swish,      # Activation function
    scale_nn_outputs = true, # Scale neural network outputs
    input_batchnorm = true   # Apply batch normalization to inputs
)
```

### 5. Train the model
```julia
using WGLMakie # to see an interactive and automatically updated train_board figure
out = train(
    hybrid_model, 
    ds, 
    (); 
    nepochs = 100,           # Number of training epochs
    batchsize = 512,         # Batch size for training
    opt = RMSProp(0.001),   # Optimizer and learning rate
    monitor_names = [:rb, :Q10], # Parameters to monitor during training
    yscale = identity,       # Scaling for outputs
    patience = 30            # Early stopping patience
)

# Check results
out.train_diffs.Q10
```

## More Examples

Check out the `projects/` directory for additional examples and use cases. Each project demonstrates different aspects of hybrid modeling with EasyHybrid.

## Acknowledgments & Funding

<div align="center">
<table style="border-collapse: collapse; border: none; white-space: nowrap;">
<tr>
<td style="text-align: center; border: none;"><img src="https://erc.europa.eu/sites/default/files/2023-06/LOGO_ERC-FLAG_FP.png" height="120" /></td>
<td style="text-align: center; border: none;"><a href="https://ai4soilhealth.eu" target="_blank"><img src="https://ai4soilhealth.eu/wp-content/uploads/2023/06/ai4soilhealth_4f.png" height="50" /></a></td>
<td style="text-align: center; border: none;"><a href="https://www.usmile-erc.eu" target="_blank"><img src="https://www.usmile-erc.eu/wp-content/uploads/sites/9/2020/04/USMILE-Logo-H-pos.jpg" height="80" /></td>
</tr>
</table>
</div>

- This work is part of the **[AI4SoilHealth](https://AI4SoilHealth.eu)** project, funded by the **European Union's Horizon Europe Research and Innovation Programme** under **Grant Agreement [No. 101086179](https://cordis.europa.eu/project/id/101086179)**.

- Supported also by the European Research Council (ERC) Synergy Grant Understanding and modeling the Earth System with Machine Learning **[USMILE](https://www.usmile-erc.eu)** under the Horizon 2020 research and innovation programme **(Grant Agreement No. 855187)**.

*Funded by the European Union. The views expressed are those of the authors and do not necessarily reflect those of the European Union or the European Research Executive Agency.*

---
