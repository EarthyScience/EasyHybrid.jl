# EXPERIMENTAL / WORK IN PROGRESS
# This file contains an experimental ALiBi implementation.
# It is planned for future integration but is currently not used in production.
#
# Reference:
# "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation"
# (Press, Smith, Lewis - ICLR 2022)

using Lux
using ChainRulesCore

"""
    get_alibi_slopes(n_heads::Int)

Generates the geometrically decaying slopes for each attention head as defined 
in the ALiBi paper (Attention with Linear Biases).

# Arguments
- `n_heads::Int`: Number of attention heads.

# Returns
- A `Vector{Float32}` containing the slope for each head.
"""
function get_alibi_slopes(n_heads::Int)
    closest_power_of_2 = 2^floor(Int, log2(n_heads))
    base = 2.0^(-8.0 / closest_power_of_2)

    slopes = Float32[base^i for i in 1:closest_power_of_2]
    if closest_power_of_2 < n_heads
        base = 2.0^(-4.0 / closest_power_of_2)
        append!(slopes, Float32[base^i for i in 1:2:(2 * (n_heads - closest_power_of_2))])
    end
    return slopes
end

"""
    alibi_attention(q, k, v, n_heads; mask=nothing)

Computes Attention with Linear Biases (ALiBi).
Because ALiBi modifies the attention scores before the softmax, this implements 
the attention steps manually instead of using `dot_product_attention`.

# Arguments
- `q`: Query tensor of shape `(head_dim, n_heads, seq_len, batch)`.
- `k`: Key tensor of shape `(head_dim, n_heads, kv_len, batch)`.
- `v`: Value tensor of shape `(head_dim, n_heads, kv_len, batch)`.
- `n_heads::Int`: Number of attention heads.
- `mask`: Optional boolean mask to apply to the attention scores. `false` elements are masked out.

# Returns
- The attended sequence tensor of shape `(head_dim, n_heads, seq_len, batch)`.
"""
function alibi_attention(q, k, v, n_heads; mask = nothing)
    # Assuming q, k, v are (head_dim, heads, seq_len, batch)
    head_dim, heads, seq_len, batch = size(q)
    kv_len = size(k, 3)

    # 1. Reshape to merge heads and batch for efficient batched_mul
    q_b = reshape(q, head_dim, seq_len, heads * batch)
    k_b = reshape(k, head_dim, kv_len, heads * batch)
    v_b = reshape(v, head_dim, kv_len, heads * batch)

    # 2. Compute Raw Attention Scores: (seq_len, kv_len, heads * batch)
    scores = batched_mul(batched_adjoint(k_b), q_b) ./ Float32(sqrt(head_dim))

    # Reshape scores to separate heads: (seq_len, kv_len, heads, batch)
    scores = reshape(scores, seq_len, kv_len, heads, batch)

    # 3. Generate ALiBi Biases
    # Note: Creating this dense bias matrix consumes memory (O(N^2)).
    # For a memory-free version, a custom kernel is required.
    slopes = get_alibi_slopes(heads)

    bias = @ignore_derivatives begin
        # 1. Create indices on the correct device
        i_idx = similar(q, Float32, seq_len)
        i_idx .= 1:seq_len
        j_idx = similar(q, Float32, kv_len)
        j_idx .= 1:kv_len

        # 2. Calculate distance matrix: j - i
        i_mat = reshape(i_idx, :, 1)
        j_mat = reshape(j_idx, 1, :)
        dist = j_mat .- i_mat

        # 3. Move slopes to the device
        slopes_dev = similar(q, Float32, heads)
        slopes_dev .= slopes
        slopes_mat = reshape(slopes_dev, 1, 1, heads, 1)

        # 4. Calculate final bias: slope * dist where dist <= 0, else -Inf
        # The result shape will be (seq_len, kv_len, heads, 1) via broadcasting
        ifelse.(dist .<= 0, slopes_mat .* dist, Float32(-Inf))
    end

    # 4. Add Bias and Mask
    scores = scores .+ bias
    if !isnothing(mask)
        # Assuming mask is a boolean matrix, we set false to -Inf
        scores = scores .+ ifelse.(mask, 0.0f0, -Inf32)
    end

    # 5. Softmax
    attn_weights = NNlib.softmax(scores, dims = 2)

    # 6. Multiply by V
    attn_weights_b = reshape(attn_weights, seq_len, kv_len, heads * batch)
    out = batched_mul(v_b, attn_weights_b)

    return reshape(out, head_dim, heads, seq_len, batch)
end
