using LuxCore, Random

export HybridModel
export constructHybridModel

"""
    HybridModel(NN, predictors, forcing, target, mech_fun, param_names, nn_names; init_globals...)

A generic hybrid container layer:

- `NN`: neural network mapping `predictors` -> outputs of size `length(nn_names)`
- `predictors`: Vector{Symbol} keys for NN inputs
- `forcing`: Symbol key for mechanistic input (e.g., suction `h`)
- `target`: Symbol for the output prediction key
- `mech_fun`: function `f(h, p1, p2, ..., pN)` implementing the physics model
- `param_names`: Vector{Symbol} names of all mechanistic parameters in the order they must be passed to `mech_fun`
- `nn_names`: Vector{Symbol} ⊆ `param_names`; these parameters come from the NN outputs
- `init_globals...`: keyword args naming the remaining `param_names` (the global parameters) with their initial Float values

During training, the NN weights + all global scalars are learned together.
"""
struct HybridModel4{D,P<:Function}
    NN           :: D
    predictors   :: Vector{Symbol}
    forcing      :: Symbol
    target       :: Symbol
    mech_fun     :: P
    param_names  :: Vector{Symbol}
    nn_names     :: Vector{Symbol}
    init_globals :: NamedTuple
end

function constructHybridModel(
    NN, predictors, forcing, target,
    mech_fun,
    param_names, nn_names, init_globals
)
    # Validation
    @assert all(n in param_names for n in nn_names) "nn_names must be subset of param_names"
    global_names = [n for n in param_names if !(n in nn_names)]
    #@assert Set(global_names) == Set(keys(init_globals)) "init_globals must cover exactly the global names"

    return HybridModel4(
      NN,
      predictors,
      forcing,
      target,
      mech_fun,
      param_names,
      nn_names,
      init_globals
    )
end

#–– Parameter setup: network + global scalars ––#
function LuxCore.initialparameters(::AbstractRNG, m::HybridModel4)
    ps_nn, _ = LuxCore.setup(Random.default_rng(), m.NN)
    # Build NamedTuple: ps (NN params) + each global as 1‑vector
    nt = (; ps = ps_nn)
    for g in filter(n-> !(n in m.nn_names), m.param_names)
        nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([Float32(m.init_globals[g])] ,)))
    end
    return nt
end

function LuxCore.initialstates(::AbstractRNG, m::HybridModel4)
    _, st_nn = LuxCore.setup(Random.default_rng(), m.NN)
    return (; st = st_nn)
end

#–– Forward pass ––#
function (m::HybridModel4)(ds_k, ps, st)
    # 1) NN features -> p
    p = ds_k(m.predictors)
    # 2) Mechanistic input vector (e.g. h)
    x = Array(ds_k(m.forcing))[1, :]

    # 3) NN output -> B×P matrix
    nn_out, st = LuxCore.apply(m.NN, p, ps.ps, st.st)

    # 4) Split into NamedTuple of NN-driven parameters
    nn_cols = eachrow(nn_out)
    nn_pairs = zip(m.nn_names, nn_cols)
    nn_ps = NamedTuple(nn_pairs)

    # 5) Extract global scalars (1‑vector) from ps
    global_names = filter(n-> !(n in m.nn_names), m.param_names)
    global_pairs = ((g => ps[g]) for g in global_names)
    glob_ps = NamedTuple(global_pairs)

    # 6) Build ordered tuple of args for mech_fun
    nn_args     = (getfield(nn_ps, n) for n in m.nn_names)
    glob_args   = (getfield(glob_ps, g) for g in global_names)
    all_args    = (nn_args..., glob_args...)
    # 7) Call physics model
    y_pred = m.mech_fun(x, all_args...)

    # 8) Package outputs (predicted + any extras)
    out = (; 
      m.target => y_pred,   # primary prediction
      (m.nn_names .=> nn_cols)..., 
      (global_names .=> getfield.(Ref(ps), global_names))...
    )

    return out, (; st)
end
""
