# struct KVCache{K, V}
#     k::K   # (head_dim, n_kv_heads, max_seq_len, max_batch)
#     v::V
# end
#
# function KVCache(head_dim::Int, n_kv_heads::Int, max_seq_len::Int, max_batch::Int)
#     return KVCache(
#         zeros(Float32, head_dim, n_kv_heads, max_seq_len, max_batch),
#         zeros(Float32, head_dim, n_kv_heads, max_seq_len, max_batch),
#     )
# end

"""
    GroupedQueryAttention(dim::Int, n_heads::Int, n_kv_heads::Int; dropout_rate=0.0f0)

Implements Grouped Query Attention (GQA).
GQA reduces memory and computational overhead during autoregressive generation 
by sharing key and value projections across multiple query heads.
"""
struct GroupedQueryAttention{Q, K, V, O, D} <: LuxCore.AbstractLuxContainerLayer{(:wq, :wk, :wv, :wo, :drop)}
    wq::Q
    wk::K
    wv::V
    wo::O
    drop::D
    n_heads::Int
    n_kv_heads::Int
    head_dim::Int
end

"""
    GroupedQueryAttention(dim::Int, n_heads::Int, n_kv_heads::Int; dropout_rate=0.0f0)

Implements Grouped Query Attention (GQA).
GQA reduces memory and computational overhead during autoregressive generation 
by sharing key and value projections across multiple query heads.

# Arguments
- `dim`: Dimensionality of the model's hidden states (`d_model`)
- `n_heads`: Number of query attention heads
- `n_kv_heads`: Number of key/value attention heads
- `dropout_rate`: Dropout probability applied to the final attention projection

# Returns
- A `GroupedQueryAttention` container layer
"""
function GroupedQueryAttention(dim::Int, n_heads::Int, n_kv_heads::Int; dropout_rate = 0.0f0)
    head_dim = dim ÷ n_heads
    return GroupedQueryAttention(
        Dense(dim => n_heads * head_dim; use_bias = false),
        Dense(dim => n_kv_heads * head_dim; use_bias = false),
        Dense(dim => n_kv_heads * head_dim; use_bias = false),
        Dense(n_heads * head_dim => dim; use_bias = false),
        Dropout(dropout_rate),
        n_heads, n_kv_heads, head_dim,
    )
end

function repeat_kv(x::AbstractArray{T, 4}, n_rep::Int) where {T}
    n_rep == 1 && return x
    head_dim, n_kv, seq_len, batch = size(x)
    x = reshape(x, head_dim, 1, n_kv, seq_len, batch)
    x = repeat(x, 1, n_rep, 1, 1, 1)
    return reshape(x, head_dim, n_rep * n_kv, seq_len, batch)
end

function causal_mask_offset(x, seq_len::Int, kv_len::Int)
    offset = kv_len - seq_len
    mask_cpu = [k <= offset + q for k in 1:kv_len, q in 1:seq_len]
    return copyto!(similar(x, Bool, kv_len, seq_len), mask_cpu)
end

# """
#     (m::GroupedQueryAttention)(x, cache::KVCache, start_pos::Int, cosf, sinf, ps, st)
#
# Autoregressive forward pass for Grouped Query Attention using a Key-Value cache.
# Used during inference/generation.
#
# # Arguments
# - `m`: The `GroupedQueryAttention` layer
# - `x`: Input sequence data of shape `(dim, seq_len, batch)`
# - `cache`: The `KVCache` object to store and retrieve past keys/values
# - `start_pos`: The starting position in the cache for the current sequence
# - `cosf`: Precomputed cosine frequencies for RoPE
# - `sinf`: Precomputed sine frequencies for RoPE
# - `ps`: Model parameters
# - `st`: Model state
#
# # Returns
# - `(out, st_out)`: Attended sequence and updated state
# """
# function (m::GroupedQueryAttention)(x, cache::KVCache, start_pos::Int, cosf, sinf, ps, st)
#     dim, seq_len, batch = size(x)
#     n_rep = m.n_heads ÷ m.n_kv_heads
#
#     q, st_q = m.wq(x, ps.wq, st.wq)
#     k, st_k = m.wk(x, ps.wk, st.wk)
#     v, st_v = m.wv(x, ps.wv, st.wv)
#
#     q = reshape(q, m.head_dim, m.n_heads, seq_len, batch)
#     k = reshape(k, m.head_dim, m.n_kv_heads, seq_len, batch)
#     v = reshape(v, m.head_dim, m.n_kv_heads, seq_len, batch)
#
#     _cosf = @view cosf[:, start_pos:(start_pos + seq_len - 1)]
#     _sinf = @view sinf[:, start_pos:(start_pos + seq_len - 1)]
#
#     q = apply_rotary_embeddings(q, _cosf, _sinf)
#     k = apply_rotary_embeddings(k, _cosf, _sinf)
#
#     # Write into cache at [start_pos, start_pos+seq_len-1]
#     cache.k[:, :, start_pos:(start_pos + seq_len - 1), 1:batch] .= k
#     cache.v[:, :, start_pos:(start_pos + seq_len - 1), 1:batch] .= v
#
#     kv_len = start_pos + seq_len - 1
#     k_full = @view cache.k[:, :, 1:kv_len, 1:batch]
#     v_full = @view cache.v[:, :, 1:kv_len, 1:batch]
#
#     k_rep = repeat_kv(k_full, n_rep)
#     v_rep = repeat_kv(v_full, n_rep)
#
#     q2 = reshape(permutedims(q, (1, 3, 2, 4)), m.head_dim, seq_len, :)
#     k2 = reshape(permutedims(k_rep, (1, 3, 2, 4)), m.head_dim, kv_len, :)
#     v2 = reshape(permutedims(v_rep, (1, 3, 2, 4)), m.head_dim, kv_len, :)
#
#     mask = seq_len > 1 ? causal_mask_offset(q2, seq_len, kv_len) : nothing
#
#     y, _ = dot_product_attention(q2, k2, v2; mask, nheads = 1)
#
#     y = reshape(y, m.head_dim, seq_len, m.n_heads, batch)
#     y = permutedims(y, (1, 3, 2, 4))
#     y = reshape(y, m.n_heads * m.head_dim, seq_len, batch)
#
#     out, st_o = m.wo(y, ps.wo, st.wo)
#     out, st_d = m.drop(out, ps.drop, st.drop)
#     return out, (wq = st_q, wk = st_k, wv = st_v, wo = st_o, drop = st_d)
# end

"""
    (m::GroupedQueryAttention)(x, ps, st; context=nothing, mask=nothing)

Standard forward pass for Grouped Query Attention without KVCache (e.g. for ViT or standard training).

# Arguments
- `m`: The `GroupedQueryAttention` layer
- `x`: Input sequence data of shape `(dim, seq_len, batch)`
- `ps`: Model parameters
- `st`: Model state
- `context`: Optional context sequence for cross-attention
- `mask`: Optional attention mask

# Returns
- `(out, st_out)`: Attended sequence and updated state
"""
function (m::GroupedQueryAttention)(x, ps, st; context = nothing, mask = nothing, cosf = nothing, sinf = nothing)
    # Forward pass without KVCache (standard training / ViT)
    # x: (dim, seq_len, batch)
    dim, seq_len, batch = size(x)
    n_rep = m.n_heads ÷ m.n_kv_heads

    q, st_q = m.wq(x, ps.wq, st.wq)
    k, st_k = m.wk(x, ps.wk, st.wk)
    v, st_v = m.wv(x, ps.wv, st.wv)

    q = reshape(q, m.head_dim, m.n_heads, seq_len, batch)
    k = reshape(k, m.head_dim, m.n_kv_heads, seq_len, batch)
    v = reshape(v, m.head_dim, m.n_kv_heads, seq_len, batch)

    if !isnothing(cosf) && !isnothing(sinf)
        q = apply_rotary_embeddings(q, cosf, sinf)
        k = apply_rotary_embeddings(k, cosf, sinf)
    end

    # In non-autoregressive setting without RoPE provided, we just do attention
    k_rep = repeat_kv(k, n_rep)
    v_rep = repeat_kv(v, n_rep)

    q2 = reshape(permutedims(q, (1, 3, 2, 4)), m.head_dim, seq_len, :)
    k2 = reshape(permutedims(k_rep, (1, 3, 2, 4)), m.head_dim, seq_len, :)
    v2 = reshape(permutedims(v_rep, (1, 3, 2, 4)), m.head_dim, seq_len, :)

    y, _ = dot_product_attention(q2, k2, v2; mask, nheads = 1)

    y = reshape(y, m.head_dim, seq_len, m.n_heads, batch)
    y = permutedims(y, (1, 3, 2, 4))
    y = reshape(y, m.n_heads * m.head_dim, seq_len, batch)

    out, st_o = m.wo(y, ps.wo, st.wo)
    out, st_d = m.drop(out, ps.drop, st.drop)
    return out, (wq = st_q, wk = st_k, wv = st_v, wo = st_o, drop = st_d)
end

"""
    MultiHeadSelfAttention(d_model, n_heads; dropout_rate=0.0f0)

Standard Multi-Head Self Attention, primarily used for Cross-Attention in Encoder-Decoder architectures.

# Arguments
- `d_model`: Dimensionality of the model's hidden states
- `n_heads`: Number of attention heads
- `dropout_rate`: Dropout probability applied to the final attention projection

# Returns
- A `MultiHeadSelfAttention` container layer
"""
struct MultiHeadSelfAttention{Q, K, V, O, D} <: LuxCore.AbstractLuxContainerLayer{(:query, :key, :value, :out, :drop)}
    query::Q
    key::K
    value::V
    out::O
    drop::D
    n_heads::Int
end

function MultiHeadSelfAttention(d_model, n_heads; dropout_rate = 0.0f0)
    return MultiHeadSelfAttention(
        Dense(d_model => d_model),
        Dense(d_model => d_model; use_bias = false),
        Dense(d_model => d_model),
        Dense(d_model => d_model),
        Dropout(dropout_rate),
        n_heads
    )
end

"""
    (m::MultiHeadSelfAttention)(x, ps, st; context=nothing, mask=nothing)

Forward pass for standard Multi-Head Self Attention.

# Arguments
- `m`: The `MultiHeadSelfAttention` layer
- `x`: Input sequence data of shape `(d_model, seq_len, batch)` (used for Queries, and Keys/Values if `context` is `nothing`)
- `ps`: Model parameters
- `st`: Model state
- `context`: Optional context sequence of shape `(d_model, context_len, batch)` for cross-attention (Keys/Values)
- `mask`: Optional attention mask

# Returns
- `(out, st_out)`: Attended sequence and updated state
"""
function (m::MultiHeadSelfAttention)(x, ps, st; context = nothing, mask = nothing)
    src = isnothing(context) ? x : context

    q, st_q = m.query(x, ps.query, st.query)
    k, st_k = m.key(src, ps.key, st.key)
    v, st_v = m.value(src, ps.value, st.value)

    y, _ = dot_product_attention(q, k, v; mask, nheads = m.n_heads)

    out, st_out = m.out(y, ps.out, st.out)
    out, st_d = m.drop(out, ps.drop, st.drop)

    return out, (query = st_q, key = st_k, value = st_v, out = st_out, drop = st_d)
end
