# EXPERIMENTAL / WORK IN PROGRESS
# This file contains an experimental Local Sliding Window Mask implementation.
# It is planned for future integration but is currently not used in production.
#
# References:
# "Longformer: The Long-Document Transformer" (Beltagy, Peters, Cohan - 2020)
# "Mistral 7B" (Jiang et al. - 2023)

using Lux

"""
    make_local_mask(x, seq_len::Int, window_size::Int)

Generates a sliding window (local) mask for attention.
Elements outside the `window_size` (both past and future) are masked out.

# Arguments
- `x`: A reference array used to determine the device (e.g., CPU or GPU) for the mask.
- `seq_len::Int`: The length of the sequence.
- `window_size::Int`: The maximum allowed distance between a query and a key.

# Returns
- A dense boolean matrix of size `(seq_len, seq_len)` where `true` indicates allowed attention.
"""
function make_local_mask(x, seq_len::Int, window_size::Int)
    return @ignore_derivatives begin
        # Create row and column indices
        q_idx = similar(x, Int, seq_len)
        q_idx .= 1:seq_len
        k_idx = similar(x, Int, seq_len)
        k_idx .= 1:seq_len

        # Calculate distance between query and key
        q_mat = reshape(q_idx, :, 1) # (seq_len, 1)
        k_mat = reshape(k_idx, 1, :) # (1, seq_len)

        dist = abs.(q_mat .- k_mat)

        # Mask is true where distance is within the window_size
        # For a causal sliding window, combine with the standard causal check:
        # mask = (k_mat .<= q_mat) .& (dist .<= window_size)

        mask = dist .<= window_size
        mask
    end
end

"""
    local_mask_offset(x, seq_len::Int, kv_len::Int, window_size::Int)

Generates a sliding window (local) causal mask offset for autoregressive generation 
with a KV cache. It only generates the mask for the current query position(s) 
relative to the cached key positions.

# Arguments
- `x`: A reference array used to determine the device for the mask.
- `seq_len::Int`: The length of the current query sequence.
- `kv_len::Int`: The length of the cached key sequence.
- `window_size::Int`: The maximum allowed distance between a query and a key.

# Returns
- A dense boolean matrix of size `(seq_len, kv_len)` where `true` indicates allowed attention.
"""
function local_mask_offset(x, seq_len::Int, kv_len::Int, window_size::Int)
    offset = kv_len - seq_len
    return @ignore_derivatives begin
        k = similar(x, Int, kv_len)
        k .= 1:kv_len
        q = similar(x, Int, seq_len)
        q .= 1:seq_len

        q_mat = reshape(q, :, 1) .+ offset
        k_mat = reshape(k, 1, :)

        # Causal and within sliding window
        (k_mat .<= q_mat) .& (abs.(q_mat .- k_mat) .<= window_size)
    end
end
