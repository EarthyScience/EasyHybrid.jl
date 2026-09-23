# Transformer Mathematics and Architecture

::: warning

**Experimental & Untested Module**
The Transformer submodule (`EasyHybrid.Transformers`) is currently experimental and under active development. APIs, layer specifications, and training workflows may change in the future.

:::

This document outlines the fundamental mathematical and tensor operations occurring within the `EasyHybrid.jl` Transformer architecture, mapping the math directly to the source code files.

## 1. Embeddings (`embeddings.jl`)

### FeatureEmbedding (Time Series)
Given an input sequence ``X \in \mathbb{R}^{F \times T \times B}`` (``F``: features, ``T``: time, ``B``: batch), a linear layer maps it to the hidden dimension ``D``:
```math
H = W_{emb} X + b_{emb} \quad \in \mathbb{R}^{D \times T \times B}
```

### PatchEmbedding (Conv2D for Spatio-Temporal Maps)
For spatial grids ``X \in \mathbb{R}^{W \times H \times C \times B}``, a 2D convolution extracts local patches of size ``P \times P``. The number of patches is ``N = (W/P) \times (H/P)``. By setting `stride = patch_size`, the convolution acts as a linear projection for each patch.
For an output feature ``d`` at spatial patch index ``(i, j)`` on the coarse grid:
```math
H_{spatial}^{(i, j, d)} = b_d + \sum_{c=1}^C \sum_{m=1}^P \sum_{n=1}^P W_{m, n, c, d} \cdot X_{(i-1)P + m, \ (j-1)P + n, \ c}
```
Flattening the spatial dimensions ``(i, j) \to N`` yields the sequence:
```math
H \in \mathbb{R}^{D \times N \times B}
```

### PatchUnEmbedding (ConvTranspose)
To invert the sequence back to spatial grids, `PatchUnEmbedding` reshapes the sequence back to ``\frac{W}{P} \times \frac{H}{P} \times D`` and applies a `ConvTranspose` operation to upscale the spatial dimensions by ``P``.
For a pixel ``(x, y)`` in the high-resolution output grid, it maps to coarse grid cell ``i = \lfloor (x-1)/P \rfloor + 1, \ j = \lfloor (y-1)/P \rfloor + 1`` with internal patch offsets ``m = ((x-1) \bmod P) + 1, \ n = ((y-1) \bmod P) + 1``:
```math
\hat{X}_{x, y, c_{out}} = b_{c_{out}} + \sum_{d=1}^D W_{m, n, d, c_{out}} \cdot H_{spatial}^{(i, j, d)}
```

---

## 2. Attention Mechanisms (`attention.jl`)

Let sequence length be ``L``. Input ``H \in \mathbb{R}^{D \times L}``.

### Self-Attention & Grouped Query Attention (GQA)
Self-Attention allows every token to interact with every other token in the sequence. For ``h`` heads (each of dimension ``d = D/h``), we explicitly define the operation over ``H \in \mathbb{R}^{D \times L}``:
```math
\text{SelfAttention}(H) = W_O \cdot \text{Concat}_{i=1}^h \left[ A_i \right] \quad \in \mathbb{R}^{D \times L}
```
Where the output of each head ``A_i`` is calculated using linearly projected Queries, Keys, and Values:
```math
A_i = (W_{V,i} H) \times \text{softmax}\left( \frac{(H^T W_{Q,i}^T) \times (W_{K,i} H)}{\sqrt{d}} + M \right)^T \quad \in \mathbb{R}^{d \times L}
```
*Note: In GQA, multiple query heads share the same ``W_K`` and ``W_V`` projections to reduce memory footprint. ``M`` is an optional causal mask matrix.*

### Cross-Attention
In Cross-Attention, the Queries come from the target sequence (``X_{dec} \in \mathbb{R}^{D \times L_{dec}}``), while the Keys and Values are explicitly drawn from the Encoder's memory (``M_{enc} \in \mathbb{R}^{D \times L_{enc}}``):
```math
\text{CrossAttention}(X_{dec}, M_{enc}) = W_O \cdot \text{Concat}_{i=1}^h \left[ (W_{V,i} M_{enc}) \times \text{softmax}\left( \frac{(X_{dec}^T W_{Q,i}^T) \times (W_{K,i} M_{enc})}{\sqrt{d}} \right)^T \right]
```

### Rotary Position Embedding (RoPE)
Instead of adding absolute positional embeddings to ``H``, RoPE rotates the query and key vectors in the complex plane based on sequence position ``m``.
For a pair of features in ``q_m \in \mathbb{R}^d``, it rotates by angle ``m \theta_k`` where ``\theta_k = 10000^{-2k/d}``:
```math
\begin{pmatrix} q'_{m, 2k} \\ q'_{m, 2k+1} \end{pmatrix} = 
\begin{pmatrix} \cos(m \theta_k) & -\sin(m \theta_k) \\ \sin(m \theta_k) & \cos(m \theta_k) \end{pmatrix}
\begin{pmatrix} q_{m, 2k} \\ q_{m, 2k+1} \end{pmatrix}
```
Because ``R_m^T R_n = R_{n-m}``, the dot product ``Q^T K`` intrinsically encodes the relative distance between tokens.

---

## 3. Architecture Blocks (`blocks.jl`)

The transformer blocks combine Attention and Feed-Forward Networks using Residual Connections and Pre-Normalization (specifically RMSNorm).

### Root Mean Square Normalization (RMSNorm)
Before entering any sub-layer, the feature vectors are normalized to stabilize training. For a feature vector ``x \in \mathbb{R}^{D}``:
```math
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}} \odot \gamma
```
where ``\gamma \in \mathbb{R}^D`` is a learnable scaling parameter.

### Feed Forward Network (SwiGLU FFN)
The FFN acts on each sequence position independently, introducing non-linearity. Following modern architectures (e.g. LLaMA, PaLM; [Shazeer 2020](https://arxiv.org/abs/2002.05202)), `EasyHybrid` implements a **SwiGLU** gated activation block with 3 linear projections without bias:
- Gate projection: ``W_1 \in \mathbb{R}^{d_{ff} \times D}``
- Up projection: ``W_3 \in \mathbb{R}^{d_{ff} \times D}``
- Down projection: ``W_2 \in \mathbb{R}^{D \times d_{ff}}``

Given input ``H \in \mathbb{R}^{D \times L}``:
```math
\text{FFN}(H) = W_2 \left( \text{Swish}(W_1 H) \odot (W_3 H) \right)
```
Where ``\text{Swish}(x) = x \cdot \sigma(x)``, ``\odot`` is the Hadamard (element-wise) product, and the hidden dimension is scaled as ``d_{ff} = \text{multiple\_of} \times \lceil \lfloor \frac{8}{3} D \rfloor / \text{multiple\_of} \rceil`` (default `multiple_of = 256`).

### TransformerBlock (Encoder)
The fundamental building block for encoding sequences. It explicitly combines Pre-RMSNorm, Self-Attention, and SwiGLU FFN with residual (skip) connections:
```math
H' = H + \text{SelfAttention}(\text{RMSNorm}_1(H))
```
```math
H_{out} = H' + \text{FFN}(\text{RMSNorm}_2(H'))
```

### CrossAttentionBlock (Decoder)
The building block for decoders. It includes a third explicit sub-layer specifically to query the Encoder's memory (``M_{enc} \in \mathbb{R}^{D \times L_{enc}}``):
```math
H' = H + \text{SelfAttention}(\text{RMSNorm}_1(H))
```
```math
H'' = H' + \text{CrossAttention}(\text{RMSNorm}_2(H'), M_{enc})
```
```math
H_{out} = H'' + \text{FFN}(\text{RMSNorm}_3(H''))
```

---

## 4. Sequence Models (`transformer.jl` & `encoder_decoder.jl`)

These modules stack the fundamental blocks to create full architectural loops.

### TransformerModel (Encoder-Only)
Given a raw sequence ``X \in \mathbb{R}^{F \times L}``, it is embedded and positional information is added to form the initial layer input ``H^{(0)}``:
```math
H^{(0)} = W_{emb} X + P_{emb} \quad \in \mathbb{R}^{D \times L}
```
The data is then processed sequentially through ``N`` identical `TransformerBlock`s:
```math
H^{(l)} = \text{TransformerBlock}^{(l)}(H^{(l-1)}) \quad \text{for } l = 1 \dots N
```
Finally, a layer normalization and linear projection output the desired target features ``C_{out}``:
```math
Y = W_{out} \cdot \text{RMSNorm}(H^{(N)}) \quad \in \mathbb{R}^{C_{out} \times L}
```

### EncoderDecoderModel (Seq2Seq)
This model splits processing into two distinct streams.
**1. Encoder**: Processes the historical sequence ``X_{past}`` (length ``L_{past}``) through ``N_{enc}`` `TransformerBlock`s to produce the latent memory:
```math
M_{enc} = H^{(N_{enc})}_{past} \quad \in \mathbb{R}^{D \times L_{past}}
```

**2. Decoder**: Processes the concurrent/future forcings ``X_{future}`` (length ``L_{future}``) through ``N_{dec}`` `CrossAttentionBlock`s. The cross-attention layers use ``M_{enc}`` for Keys and Values:
```math
Z^{(0)} = W_{emb\_dec} X_{future} + P_{emb\_dec}
```
```math
Z^{(l)} = \text{CrossAttentionBlock}^{(l)}(Z^{(l-1)}, M_{enc}) \quad \text{for } l = 1 \dots N_{dec}
```
The final decoder sequence is projected to the output:
```math
Y_{target} = W_{out} \cdot \text{RMSNorm}(Z^{(N_{dec})}) \quad \in \mathbb{R}^{C_{out} \times L_{future}}
```

---

## 5. Vision Models (`vit.jl`)

### VisionTransformer (Scalar / Classification)
Takes a spatial map ``X \in \mathbb{R}^{W \times H \times C}`` and maps it to a single global vector ``Y \in \mathbb{R}^{C_{out}}``.
1. **Patch Extraction**: ``H^{(0)} = \text{PatchEmbedding}(X) \in \mathbb{R}^{D \times N}``, where ``N`` is the number of patches.
2. **Transformer Stack**: The sequence passes through ``N`` standard `TransformerBlock`s yielding ``H^{(N)}``.
3. **Global Average Pooling**: Instead of keeping the full sequence, we average across all ``N`` spatial patches to form a single vector representation:
```math
z = \frac{1}{N} \sum_{i=1}^N H^{(N)}_i \quad \in \mathbb{R}^{D \times 1}
```
4. **Projection**: ``Y = W_{head} \cdot \text{RMSNorm}(z)``.
**This is used exclusively for global scalar regression or classification**.

### VisionToVisionModel (Map-to-Map Regression)
**Used for map-to-map regression or spatial forecasting.** It retains full spatial topology without pooling.
1. **Patch Extraction**: ``H^{(0)} = \text{PatchEmbedding}(X) \in \mathbb{R}^{D \times N}``.
2. **Transformer Stack**: The sequence is processed, allowing spatial patches to globally attend to one another, yielding ``H^{(N)}``.
3. **Unflattening**: The sequence ``H^{(N)} \in \mathbb{R}^{D \times N}`` is reshaped back into a spatial patch grid ``H_{grid} \in \mathbb{R}^{\frac{W}{P} \times \frac{H}{P} \times D}``.
4. **PatchUnEmbedding**: A `ConvTranspose` layer expands the spatial dimensions by the patch size ``P``, mapping back to the original grid scale with the target channels:
```math
Y_{map} = \text{ConvTranspose}_{P}( \text{RMSNorm}(H_{grid}) ) \quad \in \mathbb{R}^{W \times H \times C_{out}}
```

---

## 6. Pedagogical Toy Example: The Full Pipeline

Let's walk through an entire Transformer Block and Encoder-Decoder mechanism for a tiny sequence with real numbers.

**Setup:** Sequence Length ``L=2``, Model Dimension ``D=4``, Heads ``h=2``, ``d=2``.

### Phase 1: Embeddings (`embeddings.jl`)
Assume our embedded input sequence ``X \in \mathbb{R}^{4 \times 2}`` (Time 1 and Time 2) is:
```math
X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}
```

```@example toy
# Input Sequence X (Features: 4, Time: 2)
X = Float32[
    1 0;
    0 1;
    1 1;
    0 0
]

# Lux expects (features, sequence_length, batch)
X_lux = reshape(X, 4, 2, 1)
```

### Phase 2: RoPE & Self-Attention (`attention.jl`)
Assume projection matrices ``W_Q, W_K, W_V`` are Identity matrices, so initially ``Q = K = V = X``.
We split into ``h=2`` heads. 

**Head 1 (Top 2 rows):** 
```math
Q_1 = K_1 = V_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
```
**Head 2 (Bottom 2 rows):** 
```math
Q_2 = K_2 = V_2 = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}
```

**Applying RoPE to Head 1's Queries (``Q_1``):**
Let's rotate the vectors based on their time positions ``m=1`` and ``m=2``. Assume for simplicity the base angle ``\theta`` results in a ``90^\circ`` (``\pi/2``) rotation for ``m=1`` and ``180^\circ`` (``\pi``) for ``m=2``.
- Time 1 (``m=1``): Vector ``\begin{bmatrix} 1 \\ 0 \end{bmatrix}`` rotated ``90^\circ`` becomes ``\begin{bmatrix} 0 \\ 1 \end{bmatrix}``.
- Time 2 (``m=2``): Vector ``\begin{bmatrix} 0 \\ 1 \end{bmatrix}`` rotated ``180^\circ`` becomes ``\begin{bmatrix} 0 \\ -1 \end{bmatrix}``.

Rotated Queries:
```math
Q'_1 = \begin{bmatrix} 0 & 0 \\ 1 & -1 \end{bmatrix}
```
*(Keys ``K_1`` would also be rotated identically. For this example, let's assume we proceed with the standard unrotated ``Q_1, K_1`` for the dot product to keep the arithmetic obvious).*

**Dot Product & Softmax (Head 1):**
```math
S_1 = Q_1^T K_1 = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
```
Scale by ``\sqrt{2} \approx 1.414`` and apply Softmax column-wise:
```math
\text{softmax}(S'_1) = \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix}
```
Multiply by Values ``V_1``:
```math
A_1 = V_1 \times \text{softmax}^T = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix} = \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \end{bmatrix}
```

**Head 2 Output (``A_2``):**
```math
A_2 = \dots = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}
```

**Concatenate Heads (``A``):**
```math
A = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix} = \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \\ 1 & 1 \\ 0 & 0 \end{bmatrix}
```

```@example toy
using LinearAlgebra, NNlib
using EasyHybrid, EasyHybrid.Transformers, Lux, Random

# 1. Manual Math Execution
# Split into 2 heads (2 features per head)
Q1 = K1 = V1 = X[1:2, :]
Q2 = K2 = V2 = X[3:4, :]

d = 2

# (a) Standard Self-Attention (without RoPE)
S1_unrot = Q1' * K1
S1_unrot_probs = softmax(S1_unrot ./ sqrt(d); dims=1)
A1_unrot = V1 * S1_unrot_probs'

S2 = Q2' * K2
S2_probs = softmax(S2 ./ sqrt(d); dims=1)
A2 = V2 * S2_probs'

A_manual = vcat(A1_unrot, A2)
println("Manual Standard Attention Output (A): \n", round.(A_manual; digits=2))

# (b) Optional: Applying RoPE to Head 1 (90° for m=1, 180° for m=2)
R_90 = [0 -1; 1 0]
R_180 = [-1 0; 0 -1]
Q1_rotated = hcat(R_90 * Q1[:, 1], R_180 * Q1[:, 2])
K1_rotated = hcat(R_90 * K1[:, 1], R_180 * K1[:, 2])
S1_rope = Q1_rotated' * K1_rotated
S1_rope_probs = softmax(S1_rope ./ sqrt(d); dims=1)
A1_rope = V1 * S1_rope_probs'
A_manual_rope = vcat(A1_rope, A2)
println("\nManual Attention with RoPE on Head 1: \n", round.(A_manual_rope; digits=2))

# 2. Library Validation (EasyHybrid)
rng = Random.default_rng()
attention_layer = MultiHeadSelfAttention(4, 2)
ps, st = Lux.setup(rng, attention_layer)

# Force weights to Identity and zero biases to match theoretical equations
ps.query.weight .= Float32.(I(4))
ps.query.bias .= 0.0f0
ps.key.weight .= Float32.(I(4))
ps.value.weight .= Float32.(I(4))
ps.value.bias .= 0.0f0
ps.out.weight .= Float32.(I(4))
ps.out.bias .= 0.0f0

A_lux, _ = attention_layer(X_lux, ps, st)
println("\nLibrary MultiHeadSelfAttention Output: \n", round.(A_lux[:, :, 1]; digits=2))
```

### Phase 3: Residual & LayerNorm (`blocks.jl`)
We add the Attention output ``A`` back to the original input ``X`` (Residual Connection):
```math
H' = X + A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \\ 0 & 0 \end{bmatrix} + \begin{bmatrix} 0.67 & 0.33 \\ 0.33 & 0.67 \\ 1 & 1 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 1.67 & 0.33 \\ 0.33 & 1.67 \\ 2 & 2 \\ 0 & 0 \end{bmatrix}
```
RMSNorm then normalizes each column (feature vector per time step) using root-mean-square:
```math
\text{RMS}(H'_1) = \sqrt{\frac{1.67^2 + 0.33^2 + 2^2 + 0^2}{4} + \epsilon} \approx 1.31
```
```math
H'_{norm} \approx \begin{bmatrix} 1.34 & 0.14 \\ 0.14 & 1.34 \\ 1.48 & 1.48 \\ 0.0 & 0.0 \end{bmatrix}
```

```@example toy
using Statistics

# 1. Manual Math Execution
H_prime = X .+ A_manual

# Manual RMSNorm per column
rms(x) = sqrt(mean(x.^2) + 1e-5)
H_norm_manual = hcat([H_prime[:, i] ./ rms(H_prime[:, i]) for i in 1:2]...)
println("Manual RMSNorm: \n", round.(H_norm_manual; digits=2))

# 2. Library Validation
norm_layer = EasyHybrid.Transformers.RMSNorm(4; eps = 1.0f-5)
ps_n, st_n = Lux.setup(rng, norm_layer)
H_prime_lux = X_lux .+ A_lux
H_norm_lux, _ = norm_layer(H_prime_lux, ps_n, st_n)
println("\nLibrary RMSNorm Output: \n", round.(H_norm_lux[:, :, 1]; digits=2))
```

### Phase 4: FFN (`blocks.jl`)
Assume ``W_1`` simply multiplies everything by 2.
```math
\text{Swish}(W_1 H') = \text{Swish}\left(\begin{bmatrix} 3.34 & 0.66 \\ 0.66 & 3.34 \\ 4 & 4 \\ 0 & 0 \end{bmatrix}\right)
```
Swish (``x \cdot \sigma(x)``) activates these values, and ``W_2`` projects them back to form the final Encoder output ``M_{enc}``. Let's say the final Encoder memory is:
```math
M_{enc} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \\ -1 & -1 \end{bmatrix}
```

```@example toy
# FFN: W1 expands, Swish activates, W2 projects back
W1 = 2.0f0 * I(4)
W1_H = W1 * H_prime

swish(x) = x * sigmoid(x)
H_swish = swish.(W1_H)
println("H_swish: \n", round.(H_swish; digits=2))

# Let's explicitly define the mock M_enc from the text for Phase 5
M_enc = Float32[
    2  0;
    0  2;
    1  1;
   -1 -1
]
```

### Phase 5: Cross-Attention (`encoder_decoder.jl`)
Now, the Decoder wants to predict Time 3. It receives a concurrent forcing ``X_{dec} \in \mathbb{R}^{4 \times 1}`` (length 1):
```math
X_{dec} = \begin{bmatrix} 1 \\ 0 \\ 1 \\ 0 \end{bmatrix}
```
This is split into ``h=2`` heads (each ``d=2``):
- **Head 1**: ``Q_1 = [1, 0]^T``, ``K_1 = V_1 = M_{enc}[1:2, :] = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}``
  ```math
  S_1 = Q_1^T K_1 = [2, 0] \implies \text{softmax}([2, 0] / \sqrt{2}) = [0.804, 0.196]
  ```
  ```math
  A_1 = V_1 \times [0.804, 0.196]^T = \begin{bmatrix} 1.61 \\ 0.39 \end{bmatrix}
  ```
- **Head 2**: ``Q_2 = [1, 0]^T``, ``K_2 = V_2 = M_{enc}[3:4, :] = \begin{bmatrix} 1 & 1 \\ -1 & -1 \end{bmatrix}``
  ```math
  S_2 = Q_2^T K_2 = [1, 1] \implies \text{softmax}([1, 1] / \sqrt{2}) = [0.5, 0.5]
  ```
  ```math
  A_2 = V_2 \times [0.5, 0.5]^T = \begin{bmatrix} 1.0 \\ -1.0 \end{bmatrix}
  ```

Concatenating both heads yields:
```math
A_{cross} = \begin{bmatrix} A_1 \\ A_2 \end{bmatrix} = \begin{bmatrix} 1.61 \\ 0.39 \\ 1.0 \\ -1.0 \end{bmatrix}
```

```@example toy
# 1. Manual Math Execution
# Decoder concurrent forcing (length 1)
X_dec = Float32[1; 0; 1; 0]

Q_dec_1 = X_dec[1:2]
K_enc_1 = M_enc[1:2, :]
V_enc_1 = M_enc[1:2, :]

Q_dec_2 = X_dec[3:4]
K_enc_2 = M_enc[3:4, :]
V_enc_2 = M_enc[3:4, :]

S_cross_1 = Q_dec_1' * K_enc_1
S_cross_probs_1 = softmax(S_cross_1 ./ sqrt(d); dims=2)
A_cross_1 = V_enc_1 * S_cross_probs_1'

S_cross_2 = Q_dec_2' * K_enc_2
S_cross_probs_2 = softmax(S_cross_2 ./ sqrt(d); dims=2)
A_cross_2 = V_enc_2 * S_cross_probs_2'

A_cross_man = vcat(A_cross_1, A_cross_2)
println("Manual Final Cross-Attention Output (A_cross): \n", round.(A_cross_man; digits=2))

# 2. Library Validation
cross_attn = MultiHeadSelfAttention(4, 2)
ps_ca, st_ca = Lux.setup(rng, cross_attn)
ps_ca.query.weight .= Float32.(I(4))
ps_ca.query.bias .= 0.0f0
ps_ca.key.weight .= Float32.(I(4))
ps_ca.value.weight .= Float32.(I(4))
ps_ca.value.bias .= 0.0f0
ps_ca.out.weight .= Float32.(I(4))
ps_ca.out.bias .= 0.0f0

X_dec_lux = reshape(X_dec, 4, 1, 1)
M_enc_lux = reshape(M_enc, 4, 2, 1)

out_ca, _ = cross_attn(X_dec_lux, ps_ca, st_ca; context=M_enc_lux)
println("\nLibrary Cross-Attention Output: \n", round.(out_ca[:, :, 1]; digits=2))
```

---

## 7. Pedagogical Toy Example: Vision Models

Let's walk through the mathematical differences between global classification (`VisionTransformer`) and map-to-map forecasting (`VisionToVisionModel`).

**Setup:** We have a single-channel ``2 \times 2`` spatial map (``W=2, H=2, C=1``).
```math
X_{grid} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
```
We will extract patches of size ``P=1``. Total patches ``N=4``. Model dimension ``D=2``.

### Phase 1: PatchEmbedding (Conv2D)
A ``1 \times 1`` Conv2D acts as a linear map from ``C=1 \to D=2``.
Assume our filter matrix ``W_{emb} \in \mathbb{R}^{2 \times 1}`` is ``\begin{bmatrix} 1 \\ -1 \end{bmatrix}``.
In Julia's column-major ordering, the spatial grid ``X_{grid}`` is flattened column-by-column: ``(1,1) \to 1``, ``(2,1) \to 3``, ``(1,2) \to 2``, ``(2,2) \to 4``.
Multiplying each scalar pixel by ``W_{emb}`` produces our sequence of ``N=4`` tokens, each with dimension ``D=2``:
```math
H = \begin{bmatrix} 1 & 3 & 2 & 4 \\ -1 & -3 & -2 & -4 \end{bmatrix} \quad \in \mathbb{R}^{D \times N}
```

*(Assume this sequence ``H`` now passes through the `TransformerBlock` stack but remains unchanged for this example).*

### Scenario A: VisionTransformer (Scalar Classification)
For classification, we want to compress this sequence into a single global scalar representation. We use **Global Average Pooling** (GAP) to collapse the ``N=4`` patches:
```math
z = \frac{1}{4} \left( \begin{bmatrix} 1 \\ -1 \end{bmatrix} + \begin{bmatrix} 3 \\ -3 \end{bmatrix} + \begin{bmatrix} 2 \\ -2 \end{bmatrix} + \begin{bmatrix} 4 \\ -4 \end{bmatrix} \right) = \begin{bmatrix} 2.5 \\ -2.5 \end{bmatrix} \quad \in \mathbb{R}^{2 \times 1}
```
A final linear head maps this ``2 \times 1`` vector into class logits!

### Scenario B: VisionToVisionModel (Map-to-Map)
For spatial forecasting, we completely skip GAP. We must reconstruct the grid using `PatchUnEmbedding`.

**Step 1: Unflattening**
We reshape ``H \in \mathbb{R}^{2 \times 4}`` back into the ``2 \times 2`` spatial dimensions (``D \times W \times H``):
```math
H_{grid}[:, 1, 1] = \begin{bmatrix} 1 \\ -1 \end{bmatrix}, \quad H_{grid}[:, 2, 1] = \begin{bmatrix} 3 \\ -3 \end{bmatrix}, \quad H_{grid}[:, 1, 2] = \begin{bmatrix} 2 \\ -2 \end{bmatrix}, \quad H_{grid}[:, 2, 2] = \begin{bmatrix} 4 \\ -4 \end{bmatrix}
```

**Step 2: ConvTranspose**
We use a ``P=1`` transposed convolution to map the ``D=2`` features back to ``C_{out}=1`` target channels.
Assume ``W_{out} \in \mathbb{R}^{1 \times 2}`` is ``\begin{bmatrix} 1 & 1 \end{bmatrix}``.
For each pixel in the final grid, we compute the dot product:
```math
\hat{X}_{1,1} = 1(1) + 1(-1) = 0
```
```math
\hat{X}_{2,1} = 1(2) + 1(-2) = 0
```
```math
\hat{X}_{1,2} = 1(3) + 1(-3) = 0
```
```math
\hat{X}_{2,2} = 1(4) + 1(-4) = 0
```

The final map-to-map output grid perfectly retains its spatial integrity:
```math
\hat{X} = \begin{bmatrix} 0 & 0 \\ 0 & 0 \end{bmatrix}
```

```@example toy
# Vision Models Demonstration
X_grid = Float32[1 2; 3 4] # 2x2 grid

# Patch Embedding equivalent
W_emb = Float32[1; -1]
# Multiply each scalar pixel by W_emb to get D=2 sequence
H_seq = W_emb * reshape(X_grid, 1, 4) 
println("\nPatchEmbedding Sequence (H): \n", H_seq)

# Scenario A: GAP for VisionTransformer
z_gap = mean(H_seq; dims=2)
println("Global Average Pooling (z): \n", z_gap)

# Scenario B: ConvTranspose for VisionToVision
W_out = Float32[1 1]
# Multiply W_out by each sequence token and reshape to 2x2
X_hat_flat = W_out * H_seq
X_hat_grid = reshape(X_hat_flat, 2, 2)
println("PatchUnEmbedding Reconstructed Grid (X_hat): \n", X_hat_grid)
```
