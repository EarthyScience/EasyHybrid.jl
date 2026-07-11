"""
    VisionTransformer(; patch_size, in_channels, d_model, n_layers, n_heads, n_kv_heads=n_heads, max_positions, num_classes, ndims=2, use_rope=false, dropout_rate=0.0f0, stem=nothing, norm_eps=1.0f-5)

Creates a Vision Transformer. If `stem` is provided, it acts as a Hybrid feature extractor before the `PatchEmbedding`.

# Arguments
- `patch_size`: Tuple defining the spatial/spatio-temporal dimensions of each patch
- `in_channels`: Number of input channels (e.g., 1 for single variable, 3 for RGB)
- `d_model`: Dimensionality of the model's hidden states
- `n_layers`: Number of Transformer blocks
- `n_heads`: Number of query attention heads
- `n_kv_heads`: Number of key/value attention heads (defaults to `n_heads` for standard attention, less for GQA)
- `max_positions`: Maximum number of positions for additive positional embeddings
- `num_classes`: Dimensionality of the final output (e.g., number of classes for classification)
- `ndims`: Number of spatial/spatio-temporal dimensions (2 for spatial, 3 for spatio-temporal)
- `use_rope`: If true, uses Rotary Positional Embeddings instead of additive embeddings
- `dropout_rate`: Dropout probability applied to attention and feedforward layers
- `stem`: Optional Lux layer to apply before patch embedding (e.g., a CNN for Hybrid ViT)
- `norm_eps`: Epsilon value for RMSNorm stability

# Returns
- A `VisionTransformer` container layer
"""
struct VisionTransformer{ST, E, PE, S, N, O} <: LuxCore.AbstractLuxContainerLayer{(:stem, :patch_embedding, :pos_embedding, :blocks, :norm, :output)}
    stem::ST
    patch_embedding::E
    pos_embedding::PE
    blocks::S
    norm::N
    output::O
    use_rope::Bool
end

function VisionTransformer(;
        patch_size, in_channels, d_model, n_layers, n_heads, n_kv_heads = n_heads,
        max_positions, num_classes, ndims = 2, use_rope = false,
        dropout_rate::Float32 = 0.0f0, stem = nothing, norm_eps = 1.0f-5
    )

    decoder_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, dropout_rate) for _ in 1:n_layers)

    return VisionTransformer(
        stem === nothing ? NoOpLayer() : stem,
        PatchEmbedding(patch_size, in_channels, d_model; ndims = ndims),
        PositionEmbedding(d_model, max_positions),
        TransformerStack(decoder_blocks),
        RMSNorm(d_model; eps = norm_eps),
        Dense(d_model => num_classes; use_bias = false),
        use_rope
    )
end

"""
    (m::VisionTransformer)(x, ps, st)

Forward pass for the Vision Transformer model.

# Arguments
- `m`: The `VisionTransformer` model
- `x`: Input data of shape `(W, H, C, B)` for 2D or `(W, H, T, C, B)` for 3D
- `ps`: Model parameters
- `st`: Model state

# Returns
- `(y, st_out)`: A tuple containing the model predictions (logits) and updated state
"""
function (m::VisionTransformer)(x, ps, st)
    x, st_stem = m.stem(x, ps.stem, st.stem)

    y, st_emb = m.patch_embedding(x, ps.patch_embedding, st.patch_embedding)

    if !m.use_rope
        y, st_pos = m.pos_embedding(y, ps.pos_embedding, st.pos_embedding)
    else
        st_pos = st.pos_embedding
    end

    y, st_blocks = m.blocks(y, ps.blocks, st.blocks)

    # Global Average Pooling (GAP) instead of [CLS] token
    y = dropdims(mean(y; dims = 2); dims = 2)

    y, st_n = m.norm(y, ps.norm, st.norm)
    y, st_out = m.output(y, ps.output, st.output)

    return y, (stem = st_stem, patch_embedding = st_emb, pos_embedding = st_pos, blocks = st_blocks, norm = st_n, output = st_out)
end
