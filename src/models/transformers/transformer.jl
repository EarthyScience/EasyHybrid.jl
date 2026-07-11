"""
    TransformerModel(; in_features, d_model, n_layers, n_heads, n_kv_heads=n_heads, out_features, dropout_rate=0.0f0, stem=nothing, max_positions=nothing, norm_eps=1.0f-5)

Creates a continuous sequence TransformerModel using GroupedQueryAttention and RMSNorm.
Optionally accepts a `stem` (e.g. a CNN or LSTM) to act as a feature extractor before embedding.

# Arguments
- `in_features`: Number of input features/covariates per timestep
- `d_model`: Dimensionality of the model's hidden states
- `n_layers`: Number of Transformer blocks
- `n_heads`: Number of query attention heads
- `n_kv_heads`: Number of key/value attention heads (defaults to `n_heads`)
- `out_features`: Dimensionality of the final output projection
- `dropout_rate`: Dropout probability applied to attention and feedforward layers
- `stem`: Optional Lux layer to apply as a feature extractor before embedding
- `max_positions`: Unused placeholder for additive embeddings if needed later
- `norm_eps`: Epsilon value for RMSNorm stability

# Returns
- A `TransformerModel` container layer
"""
struct TransformerModel{ST, E, S, N, O} <: LuxCore.AbstractLuxContainerLayer{(:stem, :embedding, :blocks, :norm, :output)}
    stem::ST
    embedding::E
    blocks::S
    norm::N
    output::O
end

function TransformerModel(;
        in_features, d_model, n_layers, n_heads, n_kv_heads = n_heads,
        max_positions = nothing, out_features, norm_eps = 1.0f-5,
        dropout_rate::Float32 = 0.0f0, stem = nothing
    )

    decoder_blocks = Tuple(TransformerBlock(d_model, n_heads, n_kv_heads; norm_eps, dropout_rate) for _ in 1:n_layers)

    return TransformerModel(
        stem === nothing ? NoOpLayer() : stem,
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

"""
    (m::TransformerModel)(x, ps, st; causal=false)

Forward pass for the continuous sequence TransformerModel.

# Arguments
- `m`: The `TransformerModel` model
- `x`: Input sequence data of shape `(in_features, seq_len, batch)` or `(spatial..., batch)` if stem is used
- `ps`: Model parameters
- `st`: Model state
- `causal`: Boolean kwarg (default `false`). If `true`, applies a causal upper-triangular mask to prevent peeking into the future.

# Returns
- `(y, st_out)`: A tuple containing the model predictions and updated state
"""
function (m::TransformerModel)(x, ps, st; causal = false)
    # x: (in_features, seq_len, batch) or (spatial..., batch) if stem is used
    x, st_stem = m.stem(x, ps.stem, st.stem)

    seq_len = size(x, 2)

    y, st_emb = m.embedding(x, ps.embedding, st.embedding)

    mask = causal ? make_causal_mask(seq_len) : nothing

    y, st_blocks = m.blocks(y, ps.blocks, st.blocks; mask = mask)

    y, st_n = m.norm(y, ps.norm, st.norm)
    y, st_out = m.output(y, ps.output, st.output)

    return y, (stem = st_stem, embedding = st_emb, blocks = st_blocks, norm = st_n, output = st_out)
end
