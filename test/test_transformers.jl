using Test
using Random
using Lux
using NNlib
using EasyHybrid.Transformers

@testset "Transformers Forward Pass Tests" begin
    rng = Random.default_rng()
    Random.seed!(rng, 42)

    @testset "TransformerModel (1D Time Series)" begin
        # 1. Setup hyperparameters
        in_features = 5   # e.g., Covariates: Elevation, Land Cover, Time of Day, etc.
        out_features = 2  # e.g., Target Observations at the CURRENT step: Temp, Humidity
        d_model = 64
        seq_len = 24      # e.g., 24 hours of covariates
        batch_size = 32

        # 2. Initialize Model with a mock Hybrid stem (e.g. an initial feature transformation) and dropout
        model = TransformerModel(
            in_features = in_features,
            d_model = d_model,
            n_layers = 2,
            n_heads = 4,
            max_positions = 512,
            out_features = out_features,
            dropout_rate = 0.1f0,
            stem = Dense(in_features => in_features, relu)
        )

        ps, st = Lux.setup(rng, model)

        # 3. Create dummy data
        x = randn(Float32, in_features, seq_len, batch_size)

        # 4. Forward pass
        # Test training mode (dropout active)
        y, st_out = model(x, ps, st; causal = false)

        # Test inference mode (dropout inactive)
        st_test = Lux.testmode(st)
        y_test, _ = model(x, ps, st_test; causal = false)

        # 5. Verify shapes
        @test size(y) == (out_features, seq_len, batch_size)
        @test size(y_test) == (out_features, seq_len, batch_size)
        println("TransformerModel (Bidirectional + Dropout + Stem) Forward Pass OK. Output Shape: ", size(y))

        # 4b. Forward pass (causal=true, strict forecasting)
        y_causal, st_causal = model(x, ps, st; causal = true)

        # 5b. Verify shapes
        @test size(y_causal) == (out_features, seq_len, batch_size)
        println("TransformerModel Forward Pass (Causal Masked) OK. Output Shape: ", size(y_causal))
    end

    @testset "VisionTransformer (Spatial 2D -> Classification/Regression)" begin
        # 1. Setup hyperparameters
        patch_size = (16, 16)
        in_channels = 3   # e.g. RGB or multiple satellite bands
        d_model = 128
        image_size = (64, 64)
        batch_size = 16
        out_features = 10 # 10 classes or targets

        # 2. Initialize Model
        vit2d = VisionTransformer(
            patch_size = patch_size,
            in_channels = in_channels,
            d_model = d_model,
            n_layers = 2,
            n_heads = 4,
            max_positions = 256,
            num_classes = out_features,
            ndims = 2,
            use_rope = false
        )

        ps_vit, st_vit = Lux.setup(rng, vit2d)

        # 3. Create dummy image grid data: (W, H, C, B)
        x_2d = randn(Float32, image_size[1], image_size[2], in_channels, batch_size)

        # 4. Forward pass
        y_2d, st_out2d = vit2d(x_2d, ps_vit, st_vit)

        # 5. Verify shapes
        # Output should be (out_features, batch_size)
        @test size(y_2d) == (out_features, batch_size)
        println("VisionTransformer (2D) Forward Pass OK. Output Shape: ", size(y_2d))
    end

    @testset "VisionTransformer (Spatial-Temporal 3D)" begin
        # 1. Setup hyperparameters
        patch_size = (16, 16, 4) # Spatial-Temporal patches
        in_channels = 1          # Single variable, e.g., Temperature grid
        d_model = 64
        grid_size = (32, 32, 12) # (W, H, Time)
        batch_size = 8
        out_features = 1

        # 2. Initialize Model
        vit3d = VisionTransformer(
            patch_size = patch_size,
            in_channels = in_channels,
            d_model = d_model,
            n_layers = 2,
            n_heads = 4,
            max_positions = 512, # Need enough positions for the flattened grid
            num_classes = out_features,
            ndims = 3,
            use_rope = true     # Highly recommended for 3D
        )

        ps_vit3d, st_vit3d = Lux.setup(rng, vit3d)

        # 3. Create dummy 3D grid data: (W, H, T, C, B)
        x_3d = randn(Float32, grid_size[1], grid_size[2], grid_size[3], in_channels, batch_size)

        # 4. Forward pass
        y_3d, st_out3d = vit3d(x_3d, ps_vit3d, st_vit3d)

        # 5. Verify shapes
        @test size(y_3d) == (out_features, batch_size)
        println("VisionTransformer (3D Spatial-Temporal) Forward Pass OK. Output Shape: ", size(y_3d))
    end

    @testset "EncoderDecoderModel (Sequence-to-Sequence)" begin
        # 1. Setup hyperparameters
        in_features = 5   # e.g. continuous covariates sequence
        dec_features = 2  # e.g. target observations fed as shifted decoder inputs
        out_features = 2  # e.g. predicting target observations
        d_model = 64
        enc_seq_len = 24
        dec_seq_len = 6
        batch_size = 16

        # 2. Initialize Model
        model = EncoderDecoderModel(
            in_features = in_features,
            dec_features = dec_features,
            d_model = d_model,
            enc_layers = 2,
            dec_layers = 2,
            n_heads = 4,
            out_features = out_features
        )

        ps, st = Lux.setup(rng, model)

        # 3. Create dummy data
        enc_x = randn(Float32, in_features, enc_seq_len, batch_size)
        dec_x = randn(Float32, dec_features, dec_seq_len, batch_size)

        # 4. Forward pass
        # The encoder is typically unmasked (sees full sequence context),
        # but the decoder is causal (can't peek into the future of its own targets).
        y, st_out = model(enc_x, dec_x, ps, st; enc_causal = false, dec_causal = true)

        # 5. Verify shapes
        # Output should be (out_features, dec_seq_len, batch_size)
        @test size(y) == (out_features, dec_seq_len, batch_size)
        println("EncoderDecoderModel Forward Pass OK. Output Shape: ", size(y))
    end

    @testset "VisionEncoderDecoderModel (Spatio-Temporal to Sequence)" begin
        # 1. Setup hyperparameters
        patch_size = (16, 16, 4) # Spatial-Temporal patches for the encoder
        in_channels = 1          # Single variable, e.g., Temperature grid
        dec_features = 2         # Target observations fed to decoder
        out_features = 2         # Predicting target observations
        d_model = 64
        grid_size = (32, 32, 12) # (W, H, Time)
        dec_seq_len = 6
        batch_size = 4

        # 2. Initialize Model
        model = VisionEncoderDecoderModel(
            patch_size = patch_size,
            in_channels = in_channels,
            dec_features = dec_features,
            d_model = d_model,
            enc_layers = 2,
            dec_layers = 2,
            n_heads = 4,
            out_features = out_features,
            ndims = 3
        )

        ps, st = Lux.setup(rng, model)

        # 3. Create dummy data
        enc_x = randn(Float32, grid_size[1], grid_size[2], grid_size[3], in_channels, batch_size)
        dec_x = randn(Float32, dec_features, dec_seq_len, batch_size)

        # 4. Forward pass
        y, st_out = model(enc_x, dec_x, ps, st; enc_causal = false, dec_causal = true)

        # 5. Verify shapes
        @test size(y) == (out_features, dec_seq_len, batch_size)
        println("VisionEncoderDecoderModel Forward Pass OK. Output Shape: ", size(y))
    end
end
