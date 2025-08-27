```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "EasyHybrid.jl"
  tagline: Integrates neural networks with mechanistic models.
  image:
    src: /logo.png
    alt: EasyHybrid
  actions:
    - theme: brand
      text: Get Started
      link: /get_started
    - theme: alt
      text: View on Github
      link: https://github.com/EarthyScience/EasyHybrid.jl
    - theme: alt
      text: API
      link: /api
features:
  - title: Powered by Lux.jl
    details: Built for speed and flexibility in pure Julia. Native GPU acceleration across CUDA, AMDGPU, Metal, and Intel platforms enables seamless scaling from prototyping to production.
    link: https://lux.csail.mit.edu/stable/
  - title: Seamless Data Handling
    details: Built for efficient data manipulation in pure Julia. <a href = "https://github.com/JuliaData/DataFrames.jl" class="highlight-link">DataFrames.jl</a> handles tabular data and <a href="https://github.com/mcabbott/AxisKeys.jl" class="highlight-link">AxisKeys.jl</a> provides multi-dimensional named arrays with automatic differentiation support.
  - title: Feature Research
    details: Using EasyHybrid in your research? Share your work with us through a pull request or drop us a line, and we'll showcase it here alongside other innovative applications.
    link: /research/overview
---
```

## How to Install EasyHybrid.jl?

Since `EasyHybrid.jl` is registered in the Julia General registry, it is available through the Julia package manager. You can enter it by pressing `]` in the `REPL` and then typing `add EasyHybrid`. Alternatively, you can also do

```julia
julia> using Pkg
julia> Pkg.add("EasyHybrid")
```

If you want to use the latest unreleased version of `EasyHybrid.jl` you can run the following command: (in most cases the released version will be same as the version on github)

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/EarthyScience/EasyHybrid.jl")
```