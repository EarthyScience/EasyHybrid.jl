struct FeatureEmbedding{D} <: LuxCore.AbstractLuxContainerLayer{(:dense,)}
    dense::D
end

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

struct PositionEmbedding{E} <: LuxCore.AbstractLuxContainerLayer{(:embedding,)}
    embedding::E
    dim::Int
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
function PositionEmbedding(max_positions::Integer, d_model::Integer; dim::Int = 2)
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
    pos = 1:size(x, m.dim)
    emb, st_emb = m.embedding(pos, ps.embedding, st.embedding)
    return x .+ reshape(emb, size(emb, 1), size(emb, 2), 1), (embedding = st_emb,)
end

struct PatchEmbedding{C} <: LuxCore.AbstractLuxContainerLayer{(:conv,)}
    conv::C
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
function PatchEmbedding(patch_size::Tuple, in_channels::Int, d_model::Int; ndims::Int = 2)
    # Stride is equal to patch size for non-overlapping patches
    conv = Conv(patch_size, in_channels => d_model, stride = patch_size)
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
