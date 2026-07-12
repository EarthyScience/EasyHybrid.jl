"""
    EncoderDecoderModel(; in_features, dec_features, d_model, enc_layers, dec_layers, n_heads, n_kv_heads=n_heads, out_features, norm_eps=1.0f-5, dropout_rate=0.0f0, stem=nothing)

Creates a sequence-to-sequence Encoder-Decoder Transformer using GroupedQueryAttention for self-attention
and MultiHeadSelfAttention for cross-attention.

# Arguments
- `in_features`: Number of input features for the encoder
- `dec_features`: Number of input features for the decoder (shifted targets or covariates)
- `d_model`: Dimensionality of the model's hidden states
- `enc_layers`: Number of Transformer blocks in the encoder
- `dec_layers`: Number of Transformer blocks in the decoder
- `n_heads`: Number of query attention heads
- `n_kv_heads`: Number of key/value attention heads (defaults to `n_heads`)
- `out_features`: Dimensionality of the final output projection
- `norm_eps`: Epsilon value for RMSNorm stability
- `dropout_rate`: Dropout probability applied to attention and feedforward layers
- `stem`: Optional Lux layer to apply as a feature extractor before encoder embedding

# Returns
- An `EncoderDecoderModel` container layer
"""
struct EncoderDecoderModel{ST, EE, ES, EN, DE, DS, DN, O} <: LuxCore.AbstractLuxContainerLayer{(:stem, :enc_embedding, :enc_blocks, :enc_norm, :dec_embedding, :dec_blocks, :dec_norm, :output)}
    stem::ST
    enc_embedding::EE
    enc_blocks::ES
    enc_norm::EN
    dec_embedding::DE
    dec_blocks::DS
    dec_norm::DN
    output::O
end

function EncoderDecoderModel(;
        in_features, dec_features, d_model,
        enc_layers, dec_layers, n_heads, n_kv_heads = n_heads,
        out_features, norm_eps = 1.0f-5, dropout_rate = 0.0f0, stem = nothing, layer_scale_init = nothing
    )

    enc_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, cross_attention = false, dropout_rate, layer_scale_init) for _ in 1:enc_layers)
    dec_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, cross_attention = true, dropout_rate, layer_scale_init) for _ in 1:dec_layers)

    return EncoderDecoderModel(
        stem === nothing ? NoOpLayer() : stem,
        FeatureEmbedding(in_features, d_model),
        TransformerStack(enc_blocks),
        RMSNorm(d_model; eps = norm_eps),
        FeatureEmbedding(dec_features, d_model),
        TransformerStack(dec_blocks),
        RMSNorm(d_model; eps = norm_eps),
        Dense(d_model => out_features; use_bias = false)
    )
end

"""
    (m::EncoderDecoderModel)(enc_x, dec_x, ps, st; enc_causal=false, dec_causal=true)

Forward pass for the sequence-to-sequence EncoderDecoderModel.

# Arguments
- `m`: The `EncoderDecoderModel`
- `enc_x`: Encoder input sequence data
- `dec_x`: Decoder input sequence data (e.g. shifted targets)
- `ps`: Model parameters
- `st`: Model state
- `enc_causal`: Boolean kwarg (default `false`). If `true`, applies causal masking to the encoder.
- `dec_causal`: Boolean kwarg (default `true`). If `true`, applies causal masking to the decoder self-attention.

# Returns
- `(out, new_st)`: A tuple containing the model predictions and updated state
"""
function (m::EncoderDecoderModel)(enc_x, dec_x, ps, st; enc_causal = false, dec_causal = true)
    enc_x, st_stem = m.stem(enc_x, ps.stem, st.stem)

    # 1. Encoder Forward
    enc_y, st_ee = m.enc_embedding(enc_x, ps.enc_embedding, st.enc_embedding)
    enc_seq_len = size(enc_y, 2)
    enc_mask = enc_causal ? make_causal_mask(enc_y, enc_seq_len) : nothing
    memory, st_eb = m.enc_blocks(enc_y, ps.enc_blocks, st.enc_blocks; mask = enc_mask)
    memory, st_en = m.enc_norm(memory, ps.enc_norm, st.enc_norm)

    # 2. Decoder Forward
    dec_y, st_de = m.dec_embedding(dec_x, ps.dec_embedding, st.dec_embedding)
    dec_seq_len = size(dec_y, 2)
    dec_mask = dec_causal ? make_causal_mask(dec_y, dec_seq_len) : nothing

    # Pass `context=memory` for cross-attention
    dec_y, st_db = m.dec_blocks(dec_y, ps.dec_blocks, st.dec_blocks; mask = dec_mask, context = memory)
    dec_y, st_dn = m.dec_norm(dec_y, ps.dec_norm, st.dec_norm)

    # 3. Output Projection
    out, st_out = m.output(dec_y, ps.output, st.output)

    new_st = (
        stem = st_stem, enc_embedding = st_ee, enc_blocks = st_eb, enc_norm = st_en,
        dec_embedding = st_de, dec_blocks = st_db, dec_norm = st_dn, output = st_out,
    )
    return out, new_st
end

"""
    VisionEncoderDecoderModel(; patch_size, in_channels, dec_features, d_model, enc_layers, dec_layers, n_heads, n_kv_heads=n_heads, out_features, ndims=2, norm_eps=1.0f-5, dropout_rate=0.0f0, stem=nothing)

Creates a sequence-to-sequence Encoder-Decoder Transformer where the Encoder processes
spatial or spatio-temporal data (via PatchEmbedding) and the Decoder processes 
continuous sequential covariates (via FeatureEmbedding).

# Arguments
- `patch_size`: Tuple defining the spatial/spatio-temporal dimensions of each encoder patch
- `in_channels`: Number of input channels for the encoder grid
- `dec_features`: Number of input features for the decoder (shifted targets or covariates)
- `d_model`: Dimensionality of the model's hidden states
- `enc_layers`: Number of Transformer blocks in the encoder
- `dec_layers`: Number of Transformer blocks in the decoder
- `n_heads`: Number of query attention heads
- `n_kv_heads`: Number of key/value attention heads (defaults to `n_heads`)
- `out_features`: Dimensionality of the final output projection
- `ndims`: Number of spatial/spatio-temporal dimensions for the encoder (2 or 3)
- `norm_eps`: Epsilon value for RMSNorm stability
- `dropout_rate`: Dropout probability applied to attention and feedforward layers
- `stem`: Optional Lux layer to apply as a feature extractor before patch embedding

# Returns
- A `EncoderDecoderModel` configured for vision-to-sequence tasks
"""
function VisionEncoderDecoderModel(;
        patch_size, in_channels, dec_features, d_model,
        enc_layers, dec_layers, n_heads, n_kv_heads = n_heads,
        out_features, ndims = 2, norm_eps = 1.0f-5, dropout_rate = 0.0f0, stem = nothing, layer_scale_init = nothing
    )

    enc_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, cross_attention = false, dropout_rate, layer_scale_init) for _ in 1:enc_layers)
    dec_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, cross_attention = true, dropout_rate, layer_scale_init) for _ in 1:dec_layers)

    return EncoderDecoderModel(
        stem === nothing ? NoOpLayer() : stem,
        PatchEmbedding(patch_size, in_channels, d_model; ndims = ndims),
        TransformerStack(enc_blocks),
        RMSNorm(d_model; eps = norm_eps),
        FeatureEmbedding(dec_features, d_model),
        TransformerStack(dec_blocks),
        RMSNorm(d_model; eps = norm_eps),
        Dense(d_model => out_features; use_bias = false)
    )
end

"""
    VisionToVisionEncoderDecoderModel(; patch_size, grid_size, in_channels, dec_channels, out_channels, d_model, enc_layers, dec_layers, n_heads, n_kv_heads=n_heads, ndims=2, norm_eps=1.0f-5, dropout_rate=0.0f0, stem=nothing)

Creates a sequence-to-sequence Encoder-Decoder Transformer where BOTH inputs and outputs are spatial or spatio-temporal grids.
The Encoder processes the historical maps, the Decoder processes known future covariate maps, and the Output predicts target future maps.

# Arguments
- `patch_size`: Tuple defining the spatial/spatio-temporal dimensions of each patch
- `grid_size`: Tuple defining the number of patches in each dimension (e.g., `(W', H')`)
- `in_channels`: Number of input channels for the encoder grid
- `dec_channels`: Number of input channels for the decoder grid (covariates)
- `out_channels`: Number of output channels for the reconstructed prediction grid
- `d_model`: Dimensionality of the model's hidden states
- `enc_layers`: Number of Transformer blocks in the encoder
- `dec_layers`: Number of Transformer blocks in the decoder
- `n_heads`: Number of query attention heads
- `n_kv_heads`: Number of key/value attention heads
- `ndims`: Number of spatial/spatio-temporal dimensions (2 or 3)
- `norm_eps`: Epsilon value for RMSNorm stability
- `dropout_rate`: Dropout probability
- `stem`: Optional Lux layer to apply as a feature extractor before patch embedding

# Returns
- A `EncoderDecoderModel` configured for grid-to-grid forecasting tasks
"""
function VisionToVisionEncoderDecoderModel(;
        patch_size, grid_size, in_channels, dec_channels, out_channels, d_model,
        enc_layers, dec_layers, n_heads, n_kv_heads = n_heads,
        ndims = 2, norm_eps = 1.0f-5, dropout_rate = 0.0f0, stem = nothing, layer_scale_init = nothing
    )

    enc_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, cross_attention = false, dropout_rate, layer_scale_init) for _ in 1:enc_layers)
    dec_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, cross_attention = true, dropout_rate, layer_scale_init) for _ in 1:dec_layers)

    return EncoderDecoderModel(
        stem === nothing ? NoOpLayer() : stem,
        PatchEmbedding(patch_size, in_channels, d_model; ndims = ndims),
        TransformerStack(enc_blocks),
        RMSNorm(d_model; eps = norm_eps),
        PatchEmbedding(patch_size, dec_channels, d_model; ndims = ndims),
        TransformerStack(dec_blocks),
        RMSNorm(d_model; eps = norm_eps),
        PatchUnEmbedding(patch_size, d_model, out_channels, grid_size; ndims = ndims)
    )
end
