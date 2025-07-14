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
  - title: RbQ10
    details: What is this about?
    link: /research/RbQ10_results
  - title: BulkDensitySOC
    details: What's this about?
    link: /research/BulkDensitySOC_results
---
```

## How to Install EasyHybrid.jl?

Get the latest unreleased version of `EasyHybrid.jl` by run the following command: (in most cases the released version will be same as the version on github)

```julia
julia> using Pkg
julia> Pkg.add(url="https://github.com/EarthyScience/EasyHybrid.jl")
```