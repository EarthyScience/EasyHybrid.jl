struct VisionTransformer{P, E, L, N, O} <: LuxCore.AbstractLuxContainerLayer{(:patch_embedding, :position_embedding, :blocks, :norm, :head)}
    patch_embedding::P
    position_embedding::E
    blocks::L
    norm::N
    head::O
end

"""
    VisionTransformer(; patch_size, in_channels, d_model, n_layers, n_heads, n_kv_heads, max_positions, num_classes, use_rope=false, ndims=2)

Creates a Vision Transformer (ViT) that supports spatial-temporal training if `ndims=3`.
Uses State-of-the-Art components like GroupedQueryAttention and RMSNorm.
"""
function VisionTransformer(;
        patch_size, in_channels, d_model, n_layers, n_heads, n_kv_heads = n_heads,
        max_positions, num_classes, use_rope=false, ndims=2, norm_eps = 1.0f-5
    )
    
    patch_embed = PatchEmbedding(patch_size, in_channels, d_model; ndims=ndims)
    pos_embed = use_rope ? NoOpLayer() : PositionEmbedding(max_positions, d_model; dim=2)
    
    blocks = TransformerStack(Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps) for _ in 1:n_layers))
    
    norm = RMSNorm(d_model; eps = norm_eps)
    head = Dense(d_model => num_classes)
    
    return VisionTransformer(patch_embed, pos_embed, blocks, norm, head)
end

function (m::VisionTransformer)(x, ps, st)
    y, st_pe = m.patch_embedding(x, ps.patch_embedding, st.patch_embedding)
    
    if !(m.position_embedding isa NoOpLayer)
        y, st_pos = m.position_embedding(y, ps.position_embedding, st.position_embedding)
    else
        st_pos = st.position_embedding
    end
    
    # y shape is (d_model, seq_len, batch)
    # Forward through transformer blocks
    y, st_blocks = m.blocks(y, ps.blocks, st.blocks)
    
    y, st_n = m.norm(y, ps.norm, st.norm)
    
    # Global average pooling over sequence length for classification
    y_pool = dropdims(mean(y, dims=2), dims=2)
    
    logits, st_h = m.head(y_pool, ps.head, st.head)
    
    return logits, (
        patch_embedding = st_pe,
        position_embedding = st_pos,
        blocks = st_blocks,
        norm = st_n,
        head = st_h
    )
end
