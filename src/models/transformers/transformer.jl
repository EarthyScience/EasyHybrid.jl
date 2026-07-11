struct TransformerModel{E, S, N, O} <: LuxCore.AbstractLuxContainerLayer{(:embedding, :blocks, :norm, :output)}
    embedding::E
    blocks::S
    norm::N
    output::O
end

"""
    TransformerModel(; in_features, d_model, n_layers, n_heads, n_kv_heads=n_heads, max_positions, out_features)

Creates a continuous sequence TransformerModel using GroupedQueryAttention and RMSNorm.
"""
function TransformerModel(;
        in_features, d_model, n_layers, n_heads, n_kv_heads = n_heads,
        max_positions, out_features, norm_eps = 1.0f-5
    )

    decoder_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps) for _ in 1:n_layers)

    return TransformerModel(
        FeatureEmbedding(in_features, d_model),
        TransformerStack(decoder_blocks),
        RMSNorm(d_model; eps = norm_eps),
        Dense(d_model => out_features; use_bias = false)
    )
end

function make_causal_mask(seq_len::Int)
    # k_idx <= q_idx -> upper triangular matrix
    return triu(ones(Bool, seq_len, seq_len))
end

function (m::TransformerModel)(x, ps, st; causal = false)
    # x: (in_features, seq_len, batch)
    seq_len = size(x, 2)

    y, st_emb = m.embedding(x, ps.embedding, st.embedding)

    mask = causal ? make_causal_mask(seq_len) : nothing

    y, st_blocks = m.blocks(y, ps.blocks, st.blocks; mask = mask)

    y, st_n = m.norm(y, ps.norm, st.norm)
    y, st_out = m.output(y, ps.output, st.output)

    return y, (embedding = st_emb, blocks = st_blocks, norm = st_n, output = st_out)
end
