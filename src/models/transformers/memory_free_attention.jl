# EXPERIMENTAL / WORK IN PROGRESS
# This file contains experimental memory-free attention implementations.
# It is planned for future integration but is currently not used in production.
#
# Reference:
# "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
# (Dao, Fu, Ermon, Rudra, Ré - NeurIPS 2022)

using Lux
using NNlib
using ChainRulesCore

struct LazyCausalMask end

struct LazyLocalMask
    window_size::Int
end

"""
    smart_local_mask(x, seq_len::Int, window_size::Int; threshold=4096)

Generates a mask for local sliding-window attention. 
If `seq_len` exceeds `threshold`, it returns a memory-free `LazyLocalMask` object, 
triggering block-chunked attention. Otherwise, it returns a standard dense boolean matrix.

# Arguments
- `x`: Reference array for device placement.
- `seq_len::Int`: Sequence length.
- `window_size::Int`: Sliding window size.
- `threshold::Int`: Sequence length above which a memory-free approach is used.
"""
function smart_local_mask(x, seq_len::Int, window_size::Int; threshold = 4096)
    if seq_len > threshold
        return LazyLocalMask(window_size)
    else
        return make_local_mask(x, seq_len, window_size) # from local_mask.jl
    end
end

"""
    smart_causal_mask(x, seq_len::Int; threshold=4096)

Generates a causal mask. Returns a `LazyCausalMask` if sequence length exceeds threshold.
"""
function smart_causal_mask(x, seq_len::Int; threshold = 4096)
    if seq_len > threshold
        return LazyCausalMask()
    else
        return make_causal_mask(x, seq_len) # from transformer.jl
    end
end

"""
    compute_attention(q, k, v, mask)

Standard fallback for dense masks (or no mask). Uses the monolithic, highly optimized 
kernel from NNlib.
"""
function compute_attention(q, k, v, mask::Union{AbstractArray, Nothing})
    y, _ = NNlib.dot_product_attention(q, k, v; mask = mask)
    return y
end

"""
    compute_attention(q, k, v, mask::LazyLocalMask; chunk_size=1024)

Memory-free (chunked) attention for Local/Sliding Window Attention.
Instead of computing the entire N x N mask and score matrix, it splits the query 
sequence into smaller blocks. For each block, it only loads the relevant Key/Value 
blocks within the window.

This bounds VRAM usage strictly to O(chunk_size * window_size) regardless of seq_len!
"""
function compute_attention(q, k, v, mask::LazyLocalMask; chunk_size = 1024)
    # Assuming q, k, v shapes are (head_dim, seq_len, ...batched_dims)
    seq_len = size(q, 2)
    out = similar(q)

    # Process Q in manageable chunks
    for q_start in 1:chunk_size:seq_len
        q_end = min(q_start + chunk_size - 1, seq_len)
        q_chunk = selectdim(q, 2, q_start:q_end)

        # Find the valid key range for this specific query chunk
        k_start = max(1, q_start - mask.window_size)
        k_end = min(seq_len, q_end + mask.window_size)

        k_chunk = selectdim(k, 2, k_start:k_end)
        v_chunk = selectdim(v, 2, k_start:k_end)

        # Generate the dense mask ONLY for this small localized block
        # The size is at most `chunk_size x (chunk_size + 2*window_size)`
        q_idx = similar(q, Int, q_end - q_start + 1)
        q_idx .= q_start:q_end

        k_idx = similar(q, Int, k_end - k_start + 1)
        k_idx .= k_start:k_end

        q_mat = reshape(q_idx, 1, :)
        k_mat = reshape(k_idx, :, 1)

        local_mask = abs.(q_mat .- k_mat) .<= mask.window_size

        # Compute attention for this chunk
        chunk_out, _ = NNlib.dot_product_attention(q_chunk, k_chunk, v_chunk, mask = local_mask)

        # Assign to output
        selectdim(out, 2, q_start:q_end) .= chunk_out
    end

    return out
end

"""
    compute_attention(q, k, v, mask::LazyCausalMask; chunk_size=1024)

Memory-free (chunked) causal attention.
A simplistic pure-Julia implementation of the FlashAttention algorithm (online softmax tiling).
"""
function compute_attention(q, k, v, ::LazyCausalMask; chunk_size = 1024)
    head_dim, seq_len, batch = size(q)
    out = zeros(eltype(q), size(q))

    # Online softmax states
    m = fill!(similar(q, seq_len, 1, batch), -Inf32)
    l = zeros(eltype(q), seq_len, 1, batch)

    scale = 1.0f0 / sqrt(Float32(head_dim))

    for q_start in 1:chunk_size:seq_len
        q_end = min(q_start + chunk_size - 1, seq_len)
        q_chunk = selectdim(q, 2, q_start:q_end)

        m_chunk = selectdim(m, 1, q_start:q_end)
        l_chunk = selectdim(l, 1, q_start:q_end)
        out_chunk = selectdim(out, 2, q_start:q_end)

        for k_start in 1:chunk_size:q_end
            k_end = min(k_start + chunk_size - 1, seq_len)
            k_chunk = selectdim(k, 2, k_start:k_end)
            v_chunk = selectdim(v, 2, k_start:k_end)

            # 1. Compute raw scores Q * K^T: (q_len, k_len, batch)
            scores = batched_mul(batched_adjoint(q_chunk), k_chunk) .* scale

            # 2. Apply causal mask if blocks overlap
            if q_start <= k_end
                q_idx = similar(q, Int, q_end - q_start + 1)
                q_idx .= q_start:q_end
                k_idx = similar(q, Int, k_end - k_start + 1)
                k_idx .= k_start:k_end

                causal_mask = reshape(k_idx, 1, :) .> reshape(q_idx, :, 1)
                scores = scores .+ ifelse.(causal_mask, Float32(-Inf), 0.0f0)
            end

            # 3. Online Softmax Update (FlashAttention math)
            m_tilde = maximum(scores, dims = 2)
            m_new = max.(m_chunk, m_tilde)

            exp_scores = exp.(scores .- m_new)
            exp_m_diff = exp.(m_chunk .- m_new)

            l_new = exp_m_diff .* l_chunk .+ sum(exp_scores, dims = 2)

            # Align dims for V multiplication: out_chunk is (head_dim, q_len, batch)
            # exp_m_diff is (q_len, 1, batch) -> reshape to (1, q_len, batch)
            exp_m_diff_aligned = reshape(exp_m_diff, 1, size(q_chunk, 2), batch)
            out_chunk .= exp_m_diff_aligned .* out_chunk .+ batched_mul(v_chunk, batched_adjoint(exp_scores))

            m_chunk .= m_new
            l_chunk .= l_new
        end

        # 4. Finalize block by dividing by L
        l_chunk_aligned = reshape(l_chunk, 1, size(q_chunk, 2), batch)
        out_chunk ./= l_chunk_aligned
    end
    return out
end

# 4. Memory-Free Training (FlashAttention Algorithm Backward Pass)
function ChainRulesCore.rrule(config::RuleConfig, ::typeof(compute_attention), q, k, v, mask::LazyCausalMask; chunk_size = 1024)
    # 1. FORWARD PASS (Saving intermediate LogSumExp states)
    head_dim, seq_len, batch = size(q)
    out = zeros(eltype(q), size(q))

    m = fill!(similar(q, seq_len, 1, batch), -Inf32)
    l = zeros(eltype(q), seq_len, 1, batch)
    scale = 1.0f0 / sqrt(Float32(head_dim))

    for q_start in 1:chunk_size:seq_len
        q_end = min(q_start + chunk_size - 1, seq_len)
        q_chunk = selectdim(q, 2, q_start:q_end)

        m_chunk = selectdim(m, 1, q_start:q_end)
        l_chunk = selectdim(l, 1, q_start:q_end)
        out_chunk = selectdim(out, 2, q_start:q_end)

        for k_start in 1:chunk_size:q_end
            k_end = min(k_start + chunk_size - 1, seq_len)
            k_chunk = selectdim(k, 2, k_start:k_end)
            v_chunk = selectdim(v, 2, k_start:k_end)

            scores = batched_mul(batched_adjoint(q_chunk), k_chunk) .* scale

            if q_start <= k_end
                q_idx = similar(q, Int, q_end - q_start + 1)
                q_idx .= q_start:q_end
                k_idx = similar(q, Int, k_end - k_start + 1)
                k_idx .= k_start:k_end

                causal_mask = reshape(k_idx, 1, :) .> reshape(q_idx, :, 1)
                scores = scores .+ ifelse.(causal_mask, Float32(-Inf), 0.0f0)
            end

            m_tilde = maximum(scores, dims = 2)
            m_new = max.(m_chunk, m_tilde)

            exp_scores = exp.(scores .- m_new)
            exp_m_diff = exp.(m_chunk .- m_new)

            l_new = exp_m_diff .* l_chunk .+ sum(exp_scores, dims = 2)

            exp_m_diff_aligned = reshape(exp_m_diff, 1, size(q_chunk, 2), batch)
            out_chunk .= exp_m_diff_aligned .* out_chunk .+ batched_mul(v_chunk, batched_adjoint(exp_scores))

            m_chunk .= m_new
            l_chunk .= l_new
        end

        l_chunk_aligned = reshape(l_chunk, 1, size(q_chunk, 2), batch)
        out_chunk ./= l_chunk_aligned
    end

    # 2. BACKWARD PASS (Recomputing blocks memory-free)
    function compute_attention_pullback(Ybar)
        dO = unthunk(Ybar)
        dQ = zeros(eltype(q), size(q))
        dK = zeros(eltype(k), size(k))
        dV = zeros(eltype(v), size(v))

        # D_i = sum(dO_i * O_i) across head_dim
        # size(D) = (seq_len, 1, batch)
        D = reshape(sum(dO .* out, dims = 1), seq_len, 1, batch)

        for q_start in 1:chunk_size:seq_len
            q_end = min(q_start + chunk_size - 1, seq_len)
            q_chunk = selectdim(q, 2, q_start:q_end)
            dO_chunk = selectdim(dO, 2, q_start:q_end)
            dQ_chunk = selectdim(dQ, 2, q_start:q_end)

            m_chunk = selectdim(m, 1, q_start:q_end)
            l_chunk = selectdim(l, 1, q_start:q_end)
            D_chunk = selectdim(D, 1, q_start:q_end)

            for k_start in 1:chunk_size:q_end
                k_end = min(k_start + chunk_size - 1, seq_len)
                k_chunk = selectdim(k, 2, k_start:k_end)
                v_chunk = selectdim(v, 2, k_start:k_end)
                dK_chunk = selectdim(dK, 2, k_start:k_end)
                dV_chunk = selectdim(dV, 2, k_start:k_end)

                # Recompute scores & P_ij
                scores = batched_mul(batched_adjoint(q_chunk), k_chunk) .* scale
                if q_start <= k_end
                    q_idx = similar(q, Int, q_end - q_start + 1)
                    q_idx .= q_start:q_end
                    k_idx = similar(q, Int, k_end - k_start + 1)
                    k_idx .= k_start:k_end
                    causal_mask = reshape(k_idx, 1, :) .> reshape(q_idx, :, 1)
                    scores = scores .+ ifelse.(causal_mask, Float32(-Inf), 0.0f0)
                end

                P = exp.(scores .- m_chunk) ./ l_chunk # (q_len, k_len, batch)

                # dV_j += dO_i * P_ij^T -> batched_mul(dO_i, P_ij)
                dV_chunk .+= batched_mul(dO_chunk, P)

                # dP_ij = dO_i^T * V_j -> batched_mul(batched_adjoint(dO_i), V_j)
                dP = batched_mul(batched_adjoint(dO_chunk), v_chunk)

                # dS_ij = P_ij * (dP_ij - D_i) * scale
                dS = P .* (dP .- D_chunk) .* scale

                # dQ_i += dS_ij * K_j^T -> batched_mul(K_j, dS_ij^T) -> batched_mul(K, batched_adjoint(dS))
                dQ_chunk .+= batched_mul(k_chunk, batched_adjoint(dS))

                # dK_j += dS_ij^T * Q_i -> batched_mul(Q_i, dS_ij)
                dK_chunk .+= batched_mul(q_chunk, dS)
            end
        end

        return (NoTangent(), dQ, dK, dV, NoTangent())
    end

    return out, compute_attention_pullback
end
