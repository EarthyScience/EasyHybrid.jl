struct RMSNorm{W} <: Lux.AbstractLuxLayer
    dim::Int
    eps::Float32
end
RMSNorm(dim::Int; eps::Float32 = 1.0f-5) = RMSNorm{Nothing}(dim, eps)

Lux.initialparameters(rng::AbstractRNG, m::RMSNorm) = (weight = ones(Float32, m.dim),)
Lux.initialstates(rng::AbstractRNG, m::RMSNorm) = NamedTuple()

function (m::RMSNorm)(x, ps, st)
    ms = mean(abs2, x; dims = 1)
    x_normed = x ./ sqrt.(ms .+ m.eps)
    return reshape(ps.weight, m.dim, 1, 1) .* x_normed, st
end

"""
    FeedForward(dim::Int; multiple_of::Int=256, ffn_dim_multiplier::Union{Float64, Nothing}=nothing)

Implements a SwiGLU FeedForward network.
This block projects the input to a higher dimension, applies a Swish (SiLU) gating mechanism, 
and projects it back to the original dimension, offering better gradient flow than standard GELU MLPs.
"""
struct FeedForward{W1, W2, W3, D} <: LuxCore.AbstractLuxContainerLayer{(:w1, :w2, :w3, :drop)}
    w1::W1   # gate
    w2::W2   # down
    w3::W3   # up
    drop::D
end

function FeedForward(dim::Int; multiple_of::Int = 256, ffn_dim_multiplier::Union{Float64, Nothing} = nothing, dropout_rate::Float32 = 0.0f0)
    hidden = 2 * (4 * dim) ÷ 3
    hidden = ffn_dim_multiplier === nothing ? hidden : floor(Int, hidden * ffn_dim_multiplier)
    hidden = multiple_of * cld(hidden, multiple_of)
    return FeedForward(
        Dense(dim => hidden; use_bias = false),
        Dense(hidden => dim; use_bias = false),
        Dense(dim => hidden; use_bias = false),
        Dropout(dropout_rate)
    )
end

function (m::FeedForward)(x, ps, st)
    g, st_w1 = m.w1(x, ps.w1, st.w1)
    u, st_w3 = m.w3(x, ps.w3, st.w3)
    h = NNlib.swish.(g) .* u
    y, st_w2 = m.w2(h, ps.w2, st.w2)
    y, st_drop = m.drop(y, ps.drop, st.drop)
    return y, (w1 = st_w1, w2 = st_w2, w3 = st_w3, drop = st_drop)
end

"""
    TransformerStack(layers::Union{Vector, Tuple})

A sequential container for Transformer blocks. 
While `Lux.Chain` passes inputs through a sequence of layers, `TransformerStack` 
is specifically designed to correctly propagate arbitrary keyword arguments 
(such as `mask`, `cache`, and RoPE frequencies) down to each individual block, 
which is required for advanced attention mechanisms.
"""
struct TransformerStack{L} <: LuxCore.AbstractLuxContainerLayer{(:layers,)}
    layers::L
end

function TransformerStack(layers::Union{Vector, Tuple})
    names = Tuple(Symbol(:layer_, i) for i in 1:length(layers))
    return TransformerStack(NamedTuple{names}(Tuple(layers)))
end

function (m::TransformerStack)(x, ps, st; kwargs...)
    st_new = st.layers
    for name in keys(m.layers)
        x, st_i = m.layers[name](x, ps.layers[name], st.layers[name]; kwargs...)
        st_new = merge(st_new, (; name => st_i))
    end
    return x, (layers = st_new,)
end

struct TransformerBlock{A, N1, CA, CN, F, N2} <: LuxCore.AbstractLuxContainerLayer{(:attention, :attn_norm, :cross_attention, :norm_cross, :feed_forward, :ffn_norm)}
    attention::A
    attn_norm::N1
    cross_attention::CA
    norm_cross::CN
    feed_forward::F
    ffn_norm::N2
end

"""
    TransformerBlock(dim, n_heads, n_kv_heads; norm_eps=1.0f-5, cross_attention=false)

Creates a State-of-the-Art Transformer Block using RMSNorm, SwiGLU FeedForward, and GroupedQueryAttention.
If `cross_attention` is true, an additional MultiHeadSelfAttention block is added for Encoder-Decoder architectures.
"""
function TransformerBlock(dim, n_heads, n_kv_heads; norm_eps = 1.0f-5, cross_attention = false, dropout_rate::Float32 = 0.0f0)
    return TransformerBlock(
        GroupedQueryAttention(dim, n_heads, n_kv_heads; dropout_rate = dropout_rate),
        RMSNorm(dim; eps = norm_eps),
        cross_attention ? MultiHeadSelfAttention(dim, n_heads; dropout_rate = dropout_rate) : NoOpLayer(),
        cross_attention ? RMSNorm(dim; eps = norm_eps) : NoOpLayer(),
        FeedForward(dim; dropout_rate = dropout_rate),
        RMSNorm(dim; eps = norm_eps),
    )
end

function _cross_attn(m::TransformerBlock, x, ps, st, context)
    y, st_cn = m.norm_cross(x, ps.norm_cross, st.norm_cross)
    y, st_ca = m.cross_attention(y, ps.cross_attention, st.cross_attention; context)
    return x .+ y, st_ca, st_cn
end

function _cross_attn(m::TransformerBlock{A, N, NoOpLayer, NoOpLayer}, x, ps, st, context) where {A, N}
    return x, st.cross_attention, st.norm_cross
end

function (m::TransformerBlock)(x, ps, st; cache = nothing, start_pos = nothing, cosf = nothing, sinf = nothing, context = nothing, mask = nothing)
    y, st_n1 = m.attn_norm(x, ps.attn_norm, st.attn_norm)

    if m.attention isa GroupedQueryAttention && !isnothing(cache) && !isnothing(start_pos) && !isnothing(cosf) && !isnothing(sinf)
        y, st_attn = m.attention(y, cache, start_pos, cosf, sinf, ps.attention, st.attention)
    else
        # Fallback if cache is not provided (e.g. standard training without autoregressive generation)
        y, st_attn = m.attention(y, ps.attention, st.attention; context = nothing, mask = mask)
    end

    x = x .+ y

    x, st_ca, st_cn = _cross_attn(m, x, ps, st, context)

    y, st_n2 = m.ffn_norm(x, ps.ffn_norm, st.ffn_norm)
    y, st_ff = m.feed_forward(y, ps.feed_forward, st.feed_forward)
    x = x .+ y

    return x, (
            attention = st_attn,
            attn_norm = st_n1,
            cross_attention = st_ca,
            norm_cross = st_cn,
            feed_forward = st_ff,
            ffn_norm = st_n2,
        )
end
