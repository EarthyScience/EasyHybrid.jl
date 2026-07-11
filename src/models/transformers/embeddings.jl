struct FeatureEmbedding{D} <: LuxCore.AbstractLuxContainerLayer{(:dense,)}
    dense::D
    function FeatureEmbedding{D}(dense::D) where {D}
        return new{D}(dense)
    end
end

function FeatureEmbedding(in_features::Integer, d_model::Integer)
    dense = Dense(in_features => d_model)
    return FeatureEmbedding{typeof(dense)}(dense)
end

function (m::FeatureEmbedding)(x, ps, st)
    emb, st_dense = m.dense(x, ps.dense, st.dense)
    return emb, (dense = st_dense,)
end

struct PositionEmbedding{E} <: LuxCore.AbstractLuxContainerLayer{(:embedding,)}
    embedding::E
    dim::Int
    function PositionEmbedding{E}(embedding::E, dim::Int) where {E}
        return new{E}(embedding, dim)
    end
end

function PositionEmbedding(n_positions::Integer, d_model::Integer; dim::Int = 1)
    emb = Embedding(n_positions => d_model)
    return PositionEmbedding{typeof(emb)}(emb, dim)
end

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
`ndims` allows choosing 2D spatial patches or 3D spatial-temporal patches.
"""
function PatchEmbedding(patch_size::Tuple, in_channels::Int, d_model::Int; ndims::Int=2)
    # Stride is equal to patch size for non-overlapping patches
    conv = Conv(patch_size, in_channels => d_model, stride=patch_size)
    return PatchEmbedding(conv)
end

function (m::PatchEmbedding)(x, ps, st)
    # x: (W, H, C, B) for 2D or (W, H, T, C, B) for 3D
    y, st_conv = m.conv(x, ps.conv, st.conv)
    
    # Flatten spatial/temporal dimensions into sequence dimension
    d_model = size(y)[end-1]
    batch = size(y)[end]
    
    y = reshape(y, :, d_model, batch)
    y = permutedims(y, (2, 1, 3)) # (d_model, seq_len, batch)
    
    return y, (conv = st_conv,)
end
