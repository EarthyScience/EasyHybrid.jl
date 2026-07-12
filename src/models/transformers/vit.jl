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
- `use_cls_token`: If true, prepends a [CLS] token and uses it for the final output instead of GAP.
- `n_register_tokens`: Number of [REGISTER] tokens to prepend.
- `layer_scale_init`: Initial value for LayerScale (e.g., 1e-5). If nothing, LayerScale is not used.

# Returns
- A `VisionTransformer` container layer
"""
struct VisionTransformer{ST, E, PE, PT, S, N, O} <: LuxCore.AbstractLuxContainerLayer{(:stem, :patch_embedding, :pos_embedding, :prefix_tokens, :blocks, :norm, :output)}
    stem::ST
    patch_embedding::E
    pos_embedding::PE
    prefix_tokens::PT
    blocks::S
    norm::N
    output::O
    use_rope::Bool
    use_cls_token::Bool
    n_register_tokens::Int
end

function VisionTransformer(;
        patch_size, in_channels, d_model, n_layers, n_heads, n_kv_heads = n_heads,
        max_positions, num_classes, ndims = 2, use_rope = false,
        dropout_rate = 0.0f0, stem = nothing, norm_eps = 1.0f-5,
        use_cls_token = false, n_register_tokens = 0, layer_scale_init = nothing
    )

    decoder_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, dropout_rate, layer_scale_init) for _ in 1:n_layers)

    return VisionTransformer(
        stem === nothing ? NoOpLayer() : stem,
        PatchEmbedding(patch_size, in_channels, d_model; ndims = ndims),
        use_rope ? NoOpLayer() : PositionEmbedding(max_positions, d_model),
        PrefixTokens(d_model; use_cls_token = use_cls_token, n_register_tokens = n_register_tokens),
        TransformerStack(decoder_blocks),
        RMSNorm(d_model; eps = norm_eps),
        Dense(d_model => num_classes; use_bias = false),
        use_rope,
        use_cls_token,
        n_register_tokens
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

    y, st_pre = m.prefix_tokens(y, ps.prefix_tokens, st.prefix_tokens)

    y, st_blocks = m.blocks(y, ps.blocks, st.blocks)

    if m.use_cls_token
        # Extract the [CLS] token (it's the first token in the sequence)
        y = y[:, 1, :]
    else
        # Global Average Pooling over the patches
        # We must discard any register tokens before pooling!
        y_patches = y[:, (m.n_register_tokens + 1):end, :]
        y = dropdims(mean(y_patches; dims = 2); dims = 2)
    end

    y, st_n = m.norm(y, ps.norm, st.norm)
    y, st_out = m.output(y, ps.output, st.output)

    return y, (stem = st_stem, patch_embedding = st_emb, pos_embedding = st_pos, prefix_tokens = st_pre, blocks = st_blocks, norm = st_n, output = st_out)
end

"""
    VisionToVisionModel(; patch_size, grid_size, in_channels, out_channels, d_model, n_layers, n_heads, n_kv_heads=n_heads, max_positions, ndims=2, use_rope=false, dropout_rate=0.0f0, stem=nothing, norm_eps=1.0f-5)

Creates a Vision-to-Vision Transformer (e.g. for Image-to-Image regression or Grid-to-Grid forecasting). 
The input grid is processed via `PatchEmbedding` and the output sequence is reconstructed back into a grid via `PatchUnEmbedding`.

# Arguments
- `patch_size`: Tuple defining the spatial/spatio-temporal dimensions of each patch
- `grid_size`: Tuple defining the number of patches in each dimension (e.g., `(W', H')`)
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels for the reconstructed grid
- `d_model`: Dimensionality of the model's hidden states
- `n_layers`: Number of Transformer blocks
- `n_heads`: Number of query attention heads
- `n_kv_heads`: Number of key/value attention heads (defaults to `n_heads`)
- `max_positions`: Maximum number of positions for additive positional embeddings
- `ndims`: Number of spatial/spatio-temporal dimensions (2 for spatial, 3 for spatio-temporal)
- `use_rope`: If true, uses Rotary Positional Embeddings instead of additive embeddings
- `dropout_rate`: Dropout probability
- `stem`: Optional Lux layer to apply before patch embedding
- `norm_eps`: Epsilon value for RMSNorm stability
- `use_cls_token`: If true, prepends a [CLS] token.
- `n_register_tokens`: Number of [REGISTER] tokens to prepend.
- `layer_scale_init`: Initial value for LayerScale (e.g., 1e-5). If nothing, LayerScale is not used.

# Returns
- A `VisionToVisionModel` container layer
"""
struct VisionToVisionModel{ST, E, PE, PT, S, N, O} <: LuxCore.AbstractLuxContainerLayer{(:stem, :patch_embedding, :pos_embedding, :prefix_tokens, :blocks, :norm, :output)}
    stem::ST
    patch_embedding::E
    pos_embedding::PE
    prefix_tokens::PT
    blocks::S
    norm::N
    output::O
    use_rope::Bool
    use_cls_token::Bool
    n_register_tokens::Int
end

function VisionToVisionModel(;
        patch_size, grid_size, in_channels, out_channels, d_model, n_layers, n_heads, n_kv_heads = n_heads,
        max_positions, ndims = 2, use_rope = false,
        dropout_rate = 0.0f0, stem = nothing, norm_eps = 1.0f-5,
        use_cls_token = false, n_register_tokens = 0, layer_scale_init = nothing
    )

    decoder_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, dropout_rate, layer_scale_init) for _ in 1:n_layers)

    return VisionToVisionModel(
        stem === nothing ? NoOpLayer() : stem,
        PatchEmbedding(patch_size, in_channels, d_model; ndims = ndims),
        use_rope ? NoOpLayer() : PositionEmbedding(max_positions, d_model),
        PrefixTokens(d_model; use_cls_token = use_cls_token, n_register_tokens = n_register_tokens),
        TransformerStack(decoder_blocks),
        RMSNorm(d_model; eps = norm_eps),
        PatchUnEmbedding(patch_size, d_model, out_channels, grid_size; ndims = ndims),
        use_rope,
        use_cls_token,
        n_register_tokens
    )
end

"""
    (m::VisionToVisionModel)(x, ps, st)

Forward pass for the Vision-to-Vision Transformer model.

# Arguments
- `m`: The `VisionToVisionModel` model
- `x`: Input grid of shape `(W, H, C, B)` for 2D or `(W, H, T, C, B)` for 3D
- `ps`: Model parameters
- `st`: Model state

# Returns
- `(y, st_out)`: Predicted output grid of shape `(W, H, out_channels, B)` (or 3D equivalent) and updated state
"""
function (m::VisionToVisionModel)(x, ps, st)
    x, st_stem = m.stem(x, ps.stem, st.stem)

    y, st_emb = m.patch_embedding(x, ps.patch_embedding, st.patch_embedding)

    if !m.use_rope
        y, st_pos = m.pos_embedding(y, ps.pos_embedding, st.pos_embedding)
    else
        st_pos = st.pos_embedding
    end

    y, st_pre = m.prefix_tokens(y, ps.prefix_tokens, st.prefix_tokens)

    y, st_blocks = m.blocks(y, ps.blocks, st.blocks)

    # Discard prefix tokens before un-embedding
    n_prefix = (m.use_cls_token ? 1 : 0) + m.n_register_tokens
    if n_prefix > 0
        y = y[:, (n_prefix + 1):end, :]
    end

    y, st_n = m.norm(y, ps.norm, st.norm)
    y, st_out = m.output(y, ps.output, st.output)

    return y, (stem = st_stem, patch_embedding = st_emb, pos_embedding = st_pos, prefix_tokens = st_pre, blocks = st_blocks, norm = st_n, output = st_out)
end

"""
    extract_features(m::Union{VisionTransformer, VisionToVisionModel}, x, ps, st; n_blocks::Union{Int, Nothing} = 1, blocks::Union{AbstractVector{Int}, Nothing} = nothing)

Extracts the spatial/spatio-temporal features from intermediate blocks of the VisionTransformer.
This is particularly useful when using a pretrained LingBot-Vision or DINOv2 model as a frozen 
feature extractor for downstream dense prediction tasks (like depth estimation or segmentation).

# Arguments
- `m`: The `VisionTransformer` or `VisionToVisionModel` layer
- `x`: Input data of shape `(W, H, C, B)` for 2D or `(W, H, T, C, B)` for 3D
- `ps`: Model parameters
- `st`: Model state
- `n_blocks`: The number of intermediate blocks to extract from the end of the transformer (default: 1).
- `blocks`: Specific block indices to extract (e.g. `[1, 3, 5]`). If provided, `n_blocks` is ignored.

# Returns
- A tuple of tensors, one for each extracted block, reshaped back into spatial grids 
  of shape `(W', H', d_model, B)` for 2D or `(W', H', T', d_model, B)` for 3D.
"""
function extract_features(m::Union{VisionTransformer, VisionToVisionModel}, x, ps, st; n_blocks::Union{Int, Nothing} = 1, blocks::Union{AbstractVector{Int}, Nothing} = nothing)
    # We can infer the grid size by running the patch embedding's convolution directly
    # This matches the shape before flattening in PatchEmbedding
    y_conv, _ = m.patch_embedding.conv(x, ps.patch_embedding.conv, st.patch_embedding.conv)
    grid_size = size(y_conv)[1:(end - 2)]
    d_model = size(y_conv)[end - 1]
    batch = size(y_conv)[end]

    # Full PatchEmbedding forward pass
    y, st_emb = m.patch_embedding(x, ps.patch_embedding, st.patch_embedding)

    if !m.use_rope
        y, st_pos = m.pos_embedding(y, ps.pos_embedding, st.pos_embedding)
    end

    y, st_pre = m.prefix_tokens(y, ps.prefix_tokens, st.prefix_tokens)

    num_blocks = length(m.blocks.layers)
    if blocks !== nothing
        blocks_to_take = blocks
    else
        n_b = n_blocks === nothing ? 1 : n_blocks
        blocks_to_take = (num_blocks - n_b + 1):num_blocks
    end

    outputs = []
    st_b = st.blocks.layers

    for (i, name) in enumerate(keys(m.blocks.layers))
        y, st_i = m.blocks.layers[name](y, ps.blocks.layers[name], st_b[name])
        if i in blocks_to_take
            # We must discard any prefix tokens!
            n_prefix = (m.use_cls_token ? 1 : 0) + m.n_register_tokens
            y_patches = n_prefix > 0 ? y[:, (n_prefix + 1):end, :] : y

            # y_patches is (d_model, seq_len, batch)
            # To reverse the flattening done in PatchEmbedding:
            y_spatial = permutedims(y_patches, (2, 1, 3)) # (seq_len, d_model, batch)
            y_spatial = reshape(y_spatial, grid_size..., d_model, batch)

            push!(outputs, y_spatial)
        end
    end

    return tuple(outputs...)
end
