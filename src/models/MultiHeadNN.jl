export MultiHeadNN
using Lux, LuxCore, Random

"""
    MultiHeadSOC(trunk, head_bd, head_soc, head_cf, predictors, targets)

mimic `BulkDensitySOC`
a shared `trunk`, 3 task-specific heads
`predictors` and `targets` are name lists to pick values from keyed arrays
"""
struct MultiHeadNN{Tr,HB,HS,HC,TP,TT} <:
        LuxCore.AbstractLuxContainerLayer{
            (:trunk, :head_bd, :head_soc, :head_cf, :predictors, :targets)
        }
    trunk     :: Tr
    head_bd   :: HB
    head_soc  :: HS
    head_cf   :: HC
    predictors:: TP
    targets   :: TT
    function MultiHeadNN(trunk, head_bd, head_soc, head_cf,
                          predictors::TP, targets::TT) where {TP,TT}
        new{typeof(trunk),typeof(head_bd),typeof(head_soc),typeof(head_cf),TP,TT}(
            trunk, head_bd, head_soc, head_cf,
            collect(predictors), collect(targets))
    end
end

function LuxCore.initialparameters(rng::AbstractRNG, mh::MultiHeadNN)
    ps_trunk, _   = LuxCore.setup(rng, mh.trunk)
    ps_bd, _      = LuxCore.setup(rng, mh.head_bd)
    ps_soc, _     = LuxCore.setup(rng, mh.head_soc)
    ps_cf, _      = LuxCore.setup(rng, mh.head_cf)
    return (; trunk = ps_trunk,
            head_bd  = ps_bd,
            head_soc = ps_soc,
            head_cf  = ps_cf)
end

function LuxCore.initialstates(rng::AbstractRNG, mh::MultiHeadNN)
    _, st_trunk = LuxCore.setup(rng, mh.trunk)
    _, st_bd    = LuxCore.setup(rng, mh.head_bd)
    _, st_soc   = LuxCore.setup(rng, mh.head_soc)
    _, st_cf    = LuxCore.setup(rng, mh.head_cf)
    return (; trunk = st_trunk,
            head_bd  = st_bd,
            head_soc = st_soc,
            head_cf  = st_cf)
end

function (mh::MultiHeadNN)(ds_p, ps, st)
    x   = ds_p(mh.predictors)    # covs
    z,  st_trunk = LuxCore.apply(mh.trunk, x,  ps.trunk,  st.trunk) # trunk
    # heads
    y_bd,  st_bd  = LuxCore.apply(mh.head_bd,  z, ps.head_bd,  st.head_bd)
    y_soc, st_soc = LuxCore.apply(mh.head_soc, z, ps.head_soc, st.head_soc)
    y_cf,  st_cf  = LuxCore.apply(mh.head_cf,  z, ps.head_cf,  st.head_cf)

    ŷ = vcat(y_bd, y_soc, y_cf) # put together for loss
    new_st = (; trunk = st_trunk,
              head_bd  = st_bd,
              head_soc = st_soc,
              head_cf  = st_cf)
    return ŷ, new_st
end
