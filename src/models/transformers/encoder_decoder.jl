struct EncoderDecoderModel{EE, ES, EN, DE, DS, DN, O} <: LuxCore.AbstractLuxContainerLayer{(:enc_embedding, :enc_blocks, :enc_norm, :dec_embedding, :dec_blocks, :dec_norm, :output)}
    enc_embedding::EE
    enc_blocks::ES
    enc_norm::EN
    dec_embedding::DE
    dec_blocks::DS
    dec_norm::DN
    output::O
end

"""
    EncoderDecoderModel(; in_features, dec_features, d_model, enc_layers, dec_layers, n_heads, n_kv_heads=n_heads, out_features)

Creates a sequence-to-sequence Encoder-Decoder Transformer using GroupedQueryAttention for self-attention
and MultiHeadSelfAttention for cross-attention.
"""
function EncoderDecoderModel(;
        in_features, dec_features, d_model, 
        enc_layers, dec_layers, n_heads, n_kv_heads = n_heads,
        out_features, norm_eps = 1.0f-5
    )
    
    enc_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, cross_attention=false) for _ in 1:enc_layers)
    dec_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, cross_attention=true) for _ in 1:dec_layers)
    
    return EncoderDecoderModel(
        FeatureEmbedding(in_features, d_model),
        TransformerStack(enc_blocks),
        RMSNorm(d_model; eps = norm_eps),
        FeatureEmbedding(dec_features, d_model),
        TransformerStack(dec_blocks),
        RMSNorm(d_model; eps = norm_eps),
        Dense(d_model => out_features; use_bias = false)
    )
end

function (m::EncoderDecoderModel)(enc_x, dec_x, ps, st; enc_causal=false, dec_causal=true)
    enc_seq_len = size(enc_x, 2)
    dec_seq_len = size(dec_x, 2)
    
    # 1. Encoder Forward
    enc_y, st_ee = m.enc_embedding(enc_x, ps.enc_embedding, st.enc_embedding)
    enc_mask = enc_causal ? make_causal_mask(enc_seq_len) : nothing
    memory, st_eb = m.enc_blocks(enc_y, ps.enc_blocks, st.enc_blocks; mask=enc_mask)
    memory, st_en = m.enc_norm(memory, ps.enc_norm, st.enc_norm)
    
    # 2. Decoder Forward
    dec_y, st_de = m.dec_embedding(dec_x, ps.dec_embedding, st.dec_embedding)
    dec_mask = dec_causal ? make_causal_mask(dec_seq_len) : nothing
    
    # Pass `context=memory` for cross-attention
    dec_y, st_db = m.dec_blocks(dec_y, ps.dec_blocks, st.dec_blocks; mask=dec_mask, context=memory)
    dec_y, st_dn = m.dec_norm(dec_y, ps.dec_norm, st.dec_norm)
    
    # 3. Output Projection
    out, st_out = m.output(dec_y, ps.output, st.output)
    
    new_st = (
        enc_embedding = st_ee, enc_blocks = st_eb, enc_norm = st_en,
        dec_embedding = st_de, dec_blocks = st_db, dec_norm = st_dn, output = st_out
    )
    return out, new_st
end

"""
    VisionEncoderDecoderModel(; patch_size, in_channels, dec_features, d_model, enc_layers, dec_layers, n_heads, n_kv_heads=n_heads, out_features, ndims=2)

Creates a sequence-to-sequence Encoder-Decoder Transformer where the Encoder processes
spatial or spatio-temporal data (via PatchEmbedding) and the Decoder processes 
continuous sequential covariates (via FeatureEmbedding).
"""
function VisionEncoderDecoderModel(;
        patch_size, in_channels, dec_features, d_model, 
        enc_layers, dec_layers, n_heads, n_kv_heads = n_heads,
        out_features, ndims = 2, norm_eps = 1.0f-5
    )
    
    enc_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, cross_attention=false) for _ in 1:enc_layers)
    dec_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, cross_attention=true) for _ in 1:dec_layers)
    
    return EncoderDecoderModel(
        PatchEmbedding(patch_size, in_channels, d_model; ndims=ndims),
        TransformerStack(enc_blocks),
        RMSNorm(d_model; eps = norm_eps),
        FeatureEmbedding(dec_features, d_model),
        TransformerStack(dec_blocks),
        RMSNorm(d_model; eps = norm_eps),
        Dense(d_model => out_features; use_bias = false)
    )
end
