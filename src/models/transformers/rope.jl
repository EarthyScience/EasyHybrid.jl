"""
    precompute_rope_freqs(head_dim::Int, max_seq_len::Int; theta::Float32 = 10_000f0)

Precomputes the trigonometric frequencies (cosine and sine) needed for Rotary Positional Embeddings (RoPE). 

**Why RoPE matters:**
Unlike standard additive positional embeddings (which encode absolute positions as fixed, learned vectors),
RoPE encodes *relative* positional information directly into the attention mechanism's Queries and Keys
using complex rotations. This allows the model to:
1. Extrapolate to sequence lengths (or spatial grids) longer than those seen during training.
2. Intuitively understand the relative distance between tokens, making it incredibly robust for 
   spatial-temporal grids and continuous time-series where boundaries might shift.

# Arguments
- `head_dim`: The dimensionality of each attention head
- `max_seq_len`: The maximum sequence length to precompute frequencies for
- `theta`: The base for the inverse frequency computation (default `10_000f0`)

# Returns
- `(cosf, sinf)`: A tuple of cosine and sine frequency matrices, each of shape `(head_dim/2, max_seq_len)`
"""
function precompute_rope_freqs(head_dim::Int, max_seq_len::Int; theta::Float32 = 10_000f0)
    @assert iseven(head_dim) "head_dim must be even"
    inv_freq = 1.0f0 ./ (theta .^ (Float32.(0:2:(head_dim - 1)) ./ head_dim))  # (head_dim/2,)
    pos = Float32.(0:(max_seq_len - 1))                                     # (max_seq_len,)
    freqs = inv_freq * pos'                                                 # (head_dim/2, max_seq_len)
    return cos.(freqs), sin.(freqs)   # each (head_dim/2, max_seq_len)
end

"""
    apply_rotary_embeddings(x::AbstractArray{T, 4}, cosf, sinf) where {T}

Applies the precomputed rotary positional embeddings to a 4D tensor `x`.

**How it works:**
It splits the `head_dim` (feature dimension) in half and applies a 2D rotation. 
Because it directly rotates the hidden representations in the complex plane, the dot product 
between any Query and Key during attention will naturally decay based on their relative distance 
in the sequence. This injects highly effective, shift-invariant positional context without 
adding any learned parameters to the network.

# Arguments
- `x`: Input tensor of shape `(head_dim, n_heads, seq_len, batch)` (either Queries or Keys)
- `cosf`: Precomputed cosine frequencies from `precompute_rope_freqs`
- `sinf`: Precomputed sine frequencies from `precompute_rope_freqs`

# Returns
- A rotated tensor of the same shape as `x`
"""
function apply_rotary_embeddings(x::AbstractArray{T, 4}, cosf, sinf) where {T}
    head_dim, n_heads, seq_len, batch = size(x)
    half = head_dim ÷ 2

    x1 = @view x[1:half, :, :, :]
    x2 = @view x[(half + 1):end, :, :, :]

    # Expand cosf and sinf to match dimensions for broadcasting
    c = reshape(cosf, half, 1, seq_len, 1)
    s = reshape(sinf, half, 1, seq_len, 1)

    rotated1 = x1 .* c .- x2 .* s
    rotated2 = x2 .* c .+ x1 .* s

    return vcat(rotated1, rotated2)
end
