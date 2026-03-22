````@raw html
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
    details: Efficient data manipulation with <a href="https://github.com/JuliaData/DataFrames.jl" class="highlight-link" target="_blank" rel="noopener noreferrer">DataFrames.jl</a> for tabular data, and <a href="https://github.com/rafaqz/DimensionalData.jl" class="highlight-link" target="_blank" rel="noopener noreferrer">DimensionalData.jl</a> and <a href="https://github.com/mcabbott/AxisKeys.jl" class="highlight-link" target="_blank" rel="noopener noreferrer">AxisKeys.jl</a> for named multidimensional arrays with <a href="https://juliadiff.org" class="highlight-link" target="_blank" rel="noopener noreferrer">automatic differentiation</a> support.
  - title: Feature Research
    details: Using EasyHybrid in your research? Share your work with us through a pull request or drop us a line, and we'll showcase it here alongside other innovative applications.
    link: /research/overview
---

<div class="feature-showcase">

<div class="feature-row">
  <div class="feature-text">
    <h3>Install via Julia Package Manager</h3>
    <p>EasyHybrid.jl is registered in the Julia General registry. Enter the package manager by pressing <code>]</code> in the REPL and add it with a single command.</p>
    <div class="feature-links">
      <a href="./get_started#Installation">Learn more</a>
    </div>
  </div>
  <div class="feature-code">

```julia
julia> ]
pkg> add EasyHybrid
```

  </div>
</div>

<div class="feature-row feature-row-reverse">
  <div class="feature-text">
    <h3>Install via Pkg.add</h3>
    <p>Use Julia's <code>Pkg</code> module directly to add EasyHybrid.jl programmatically, great for reproducible project setups.</p>
    <div class="feature-links">
      <a href="./get_started#Installation">Learn more</a>
    </div>
  </div>
  <div class="feature-code">

```julia
using Pkg
Pkg.add("EasyHybrid")
```

  </div>
</div>

<div class="feature-row">
  <div class="feature-text">
    <h3>Install Latest from GitHub</h3>
    <p>Install directly from GitHub to get the latest unreleased version. In most cases this will match the released version.</p>
    <div class="feature-links">
      <a href="https://github.com/EarthyScience/EasyHybrid.jl" target="_blank">GitHub repo</a>
    </div>
  </div>
  <div class="feature-code">

```julia
using Pkg
Pkg.add(url="https://github.com/EarthyScience/EasyHybrid.jl")
# or in the REPL
julia> ]
pkg> add EasyHybrid#main
```

  </div>
</div>

<div class="feature-row feature-row-reverse">
    <div class="feature-text">
      <h3>Load EasyHybrid in Your Session</h3>
      <p>Once installed, add this line at the beginning of any Julia script or session to load EasyHybrid.jl and access all its features.</p>
      <div class="feature-links">
        <a href="./get_started#Quickstart">Learn more</a>
      </div>
    </div>
    <div class="feature-code">

```julia
using EasyHybrid
```

  </div>
</div>

</div>

````