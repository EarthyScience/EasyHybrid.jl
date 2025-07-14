# EasyHybrid.jl
<img src="docs/src/assets/logo.png" align="right" width="30%"></img>
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://earthyscience.github.io/EasyHybrid.jl/dev/)
[![CI](https://github.com/EarthyScience/EasyHybrid.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/EarthyScience/EasyHybrid.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/EarthyScience/EasyHybrid.jl/blob/main/LICENSE)



> [!CAUTION]
> Work in progress

`EasyHybrid.jl` provides a simple and flexible framework for hybrid modeling, enabling the integration of neural networks with mechanistic (physics-based) models. This approach can be expressed as:

$$
\hat{y} = \mathcal{M}(h(x;\theta), z; \phi)
$$

where $\hat{y}$ denotes the predicted output of the hybrid model, $h(x;\theta)$ is a neural network with inputs $x$ and learnable parameters $\theta$, $z$ denotes additional inputs passed directly to the mechanistic model $\mathcal{M}(\cdot, z;\, \phi)$, which is parameterized by $\phi$. The parameters $\phi$ may be known from first principles or learned from data.


## Installation
Clone the repository

```sh
git clone https://github.com/EarthyScience/EasyHybrid.jl.git
```

and start using it by opening one of the `env` in `projects`, i.e. Q10.jl. There executing the first 4 lines should get you all needed dependencies. `shift + enter`.

### If you want to start adding new functionality then do

```sh
EasyHybrid $ julia # call julia in the EasyHybrid directory
```

```sh
julia> ] # ']' should be pressed, this is the pkg mode
```

```sh
pkg > activate . # activate this project
```

### install dependencies

```sh
pkg > instantiate
```

and now you are good to go!

```julia
using EasyHybrid
```