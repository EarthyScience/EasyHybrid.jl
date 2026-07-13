"""
    FeatureEmbedding(in_features::Int, d_model::Int)

Creates a FeatureEmbedding layer for continuous/multivariate sequences.
Unlike standard TokenEmbeddings (which use dictionaries for discrete tokens), 
this projects continuous features into the hidden `d_model` dimension via a Dense layer.

# Arguments
- `in_features`: Number of input features/covariates per timestep
- `d_model`: Dimensionality of the model's hidden states

# Returns
- A `FeatureEmbedding` container layer
"""
struct FeatureEmbedding{D} <: LuxCore.AbstractLuxContainerLayer{(:dense,)}
    dense::D
end

function FeatureEmbedding(in_features::Int, d_model::Int)
    return FeatureEmbedding(Dense(in_features => d_model))
end

"""
    (m::FeatureEmbedding)(x, ps, st)

Forward pass for the continuous FeatureEmbedding.

# Arguments
- `m`: The `FeatureEmbedding` layer
- `x`: Input sequence data of shape `(in_features, seq_len, batch)`
- `ps`: Model parameters
- `st`: Model state

# Returns
- `(y, st_out)`: Projected features of shape `(d_model, seq_len, batch)` and updated state
"""
function (m::FeatureEmbedding)(x, ps, st)
    y, st_dense = m.dense(x, ps.dense, st.dense)
    return y, (dense = st_dense,)
end

"""
    PrefixTokens(embed_dim::Int; use_cls_token::Bool=true, n_register_tokens::Int=0)

Creates a layer that maintains learned [CLS] and [REGISTER] tokens.
During the forward pass, these tokens are repeated for the batch size and prepended to the sequence.

# Arguments
- `embed_dim`: Dimensionality of the model's hidden states
- `use_cls_token`: If true, includes a single [CLS] token
- `n_register_tokens`: Number of [REGISTER] (storage) tokens to include

# Returns
- A `PrefixTokens` layer
"""
struct PrefixTokens <: Lux.AbstractLuxLayer
    embed_dim::Int
    use_cls_token::Bool
    n_register_tokens::Int
end

function PrefixTokens(embed_dim::Int; use_cls_token::Bool = true, n_register_tokens::Int = 0)
    return PrefixTokens(embed_dim, use_cls_token, n_register_tokens)
end

function Lux.initialparameters(rng::AbstractRNG, m::PrefixTokens)
    ps = NamedTuple()
    if m.use_cls_token
        ps = merge(ps, (cls_token = randn(rng, Float32, m.embed_dim, 1, 1) .* 0.02f0,))
    end
    if m.n_register_tokens > 0
        ps = merge(ps, (register_tokens = randn(rng, Float32, m.embed_dim, m.n_register_tokens, 1) .* 0.02f0,))
    end
    return ps
end

Lux.initialstates(rng::AbstractRNG, m::PrefixTokens) = NamedTuple()

function (m::PrefixTokens)(x, ps, st)
    batch_size = size(x)[end]
    if m.use_cls_token && m.n_register_tokens > 0
        cls = repeat(ps.cls_token, 1, 1, batch_size)
        reg = repeat(ps.register_tokens, 1, 1, batch_size)
        prefix = cat(cls, reg; dims = 2)
        return cat(prefix, x; dims = 2), st
    elseif m.use_cls_token
        cls = repeat(ps.cls_token, 1, 1, batch_size)
        return cat(cls, x; dims = 2), st
    elseif m.n_register_tokens > 0
        reg = repeat(ps.register_tokens, 1, 1, batch_size)
        return cat(reg, x; dims = 2), st
    else
        return x, st
    end
end

"""
    PositionEmbedding(max_positions::Integer, d_model::Integer; dim::Int=2)

Creates a learned additive Positional Embedding layer.
Typically used when RoPE is disabled.

# Arguments
- `max_positions`: The maximum sequence length supported
- `d_model`: Dimensionality of the model's hidden states
- `dim`: The dimension along which to add the embeddings (default `2` for `seq_len`)

# Returns
- A `PositionEmbedding` layer
"""
struct PositionEmbedding{E} <: LuxCore.AbstractLuxContainerLayer{(:embedding,)}
    embedding::E
    dim::Int
end

function PositionEmbedding(max_positions::Int, d_model::Int; dim::Int = 2)
    emb = Embedding(max_positions => d_model)
    return PositionEmbedding(emb, dim)
end

"""
    (m::PositionEmbedding)(x, ps, st)

Forward pass for the additive PositionEmbedding.

# Arguments
- `m`: The `PositionEmbedding` layer
- `x`: Hidden states of shape `(d_model, seq_len, batch)`
- `ps`: Model parameters
- `st`: Model state

# Returns
- `(y, st)`: Hidden states with positional embeddings added, and updated state
"""
function (m::PositionEmbedding)(x, ps, st)
    pos = @ignore_derivatives begin
        p = similar(x, Int, size(x, m.dim))
        p .= 1:size(x, m.dim)
        p
    end
    emb, st_emb = m.embedding(pos, ps.embedding, st.embedding)
    return x .+ reshape(emb, size(emb, 1), size(emb, 2), 1), (embedding = st_emb,)
end

"""
    PatchEmbedding(patch_size::Tuple, in_channels::Int, d_model::Int; ndims::Int=2)

Creates a PatchEmbedding for Vision Transformers using a Convolutional layer.

# Arguments
- `patch_size`: Tuple defining the patch dimensions
- `in_channels`: Number of input channels
- `d_model`: Dimensionality of the model's hidden states
- `ndims`: 2 for spatial grids (images), 3 for spatio-temporal grids (video/climate)

# Returns
- A `PatchEmbedding` container layer
"""
struct PatchEmbedding{C} <: LuxCore.AbstractLuxContainerLayer{(:conv,)}
    conv::C
end

function PatchEmbedding(patch_size::Tuple, in_channels::Int, d_model::Int; ndims::Int = 2)
    # Stride is equal to patch size for non-overlapping patches.
    # We use cross_correlation=true to match PyTorch's Conv behavior (useful if loading PyTorch weights).
    conv = Conv(patch_size, in_channels => d_model, stride = patch_size, cross_correlation = true)
    return PatchEmbedding(conv)
end

"""
    (m::PatchEmbedding)(x, ps, st)

Forward pass for PatchEmbedding.

# Arguments
- `m`: The `PatchEmbedding` layer
- `x`: Input data of shape `(W, H, C, B)` for 2D or `(W, H, T, C, B)` for 3D
- `ps`: Model parameters
- `st`: Model state

# Returns
- `(y, st_out)`: Flattened sequence of patches of shape `(d_model, seq_len, batch)` and updated state
"""
function (m::PatchEmbedding)(x, ps, st)
    # x: (W, H, C, B) for 2D or (W, H, T, C, B) for 3D
    y, st_conv = m.conv(x, ps.conv, st.conv)

    # Flatten spatial/temporal dimensions into sequence dimension
    d_model = size(y)[end - 1]
    batch = size(y)[end]

    y = reshape(y, :, d_model, batch)
    y = permutedims(y, (2, 1, 3)) # (d_model, seq_len, batch)

    return y, (conv = st_conv,)
end

"""
    PatchUnEmbedding(patch_size::Tuple, d_model::Int, out_channels::Int, grid_size::Tuple; ndims::Int=2)

Creates a PatchUnEmbedding layer to reconstruct spatial or spatio-temporal grids 
from a sequence of patches. This is the reverse of `PatchEmbedding`.

# Arguments
- `patch_size`: Tuple defining the patch dimensions
- `d_model`: Dimensionality of the model's hidden states
- `out_channels`: Number of output channels for the reconstructed grid
- `grid_size`: The number of patches in each spatial/temporal dimension (e.g., `(W', H')` for 2D).
- `ndims`: 2 for spatial grids (images), 3 for spatio-temporal grids (video/climate)

# Returns
- A `PatchUnEmbedding` container layer
"""
struct PatchUnEmbedding{C} <: LuxCore.AbstractLuxContainerLayer{(:conv_transpose,)}
    conv_transpose::C
    grid_size::Tuple
end

function PatchUnEmbedding(patch_size::Tuple, d_model::Int, out_channels::Int, grid_size::Tuple; ndims::Int = 2)
    # We use cross_correlation=true to match PyTorch's ConvTranspose behavior.
    conv_t = ConvTranspose(patch_size, d_model => out_channels, stride = patch_size, cross_correlation = true)
    return PatchUnEmbedding(conv_t, grid_size)
end

"""
    (m::PatchUnEmbedding)(x, ps, st)

Forward pass for PatchUnEmbedding.

# Arguments
- `m`: The `PatchUnEmbedding` layer
- `x`: Input sequence of patches of shape `(d_model, seq_len, batch)`
- `ps`: Model parameters
- `st`: Model state

# Returns
- `(out, st_out)`: Reconstructed grid of shape `(W, H, C, B)` for 2D or `(W, H, T, C, B)` for 3D and updated state
"""
function (m::PatchUnEmbedding)(x, ps, st)
    batch = size(x, 3)
    d_model = size(x, 1)

    # 1. Permute back to (seq_len, d_model, batch)
    y = permutedims(x, (2, 1, 3))

    # 2. Reshape to spatial grid of patches: (W', H', d_model, batch)
    y = reshape(y, m.grid_size..., d_model, batch)

    # 3. Apply Transposed Convolution to upsample patches to pixels
    out, st_conv = m.conv_transpose(y, ps.conv_transpose, st.conv_transpose)

    return out, (conv_transpose = st_conv,)
end
