using Test
using Random
using Lux
using NNlib
using Logging
using EasyHybrid.Transformers

@testset "Transformers Forward Pass Tests" begin
    rng = Random.Xoshiro(42)

    @testset "PatchEmbedding Configuration" begin
        pe = PatchEmbedding((16, 16), 3, 64; ndims = 2)
        # Lux Conv layers use Static.True() or standard boolean for cross_correlation
        has_cross_corr = pe.conv.cross_correlation == true || string(pe.conv.cross_correlation) == "True()"
        @test has_cross_corr
    end

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

        # 4. Test Training Mode (dropout active)
        # We use NullLogger to suppress Lux's warning about running training mode outside of autodiff
        with_logger(NullLogger()) do
            y_train, _ = model(x, ps, st; causal = false)
            @test size(y_train) == (out_features, seq_len, batch_size)
        end

        # 5. Switch to Inference Mode for the remaining tests
        st = Lux.testmode(st)

        # 6. Forward pass
        y, st_out = model(x, ps, st; causal = false)

        # 7. Verify shapes
        @test size(y) == (out_features, seq_len, batch_size)
        @info "TransformerModel (Bidirectional + Dropout + Stem) Forward Pass OK. Output Shape: $(size(y))"

        # 6b. Forward pass (causal=true, strict forecasting)
        y_causal, st_causal = model(x, ps, st; causal = true)

        # 7b. Verify shapes
        @test size(y_causal) == (out_features, seq_len, batch_size)
        @info "TransformerModel Forward Pass (Causal Masked) OK. Output Shape: $(size(y_causal))"
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
        st_vit = Lux.testmode(st_vit)

        # 3. Create dummy image grid data: (W, H, C, B)
        x_2d = randn(Float32, image_size[1], image_size[2], in_channels, batch_size)

        # 4. Forward pass
        y_2d, st_out2d = vit2d(x_2d, ps_vit, st_vit)

        # 5. Verify shapes
        # Output should be (out_features, batch_size)
        @test size(y_2d) == (out_features, batch_size)
        @info "VisionTransformer (2D) Forward Pass OK. Output Shape: $(size(y_2d))"
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
        st_vit3d = Lux.testmode(st_vit3d)

        # 3. Create dummy 3D grid data: (W, H, T, C, B)
        x_3d = randn(Float32, grid_size[1], grid_size[2], grid_size[3], in_channels, batch_size)

        # 4. Forward pass
        y_3d, st_out3d = vit3d(x_3d, ps_vit3d, st_vit3d)

        # 5. Verify shapes
        @test size(y_3d) == (out_features, batch_size)
        @info "VisionTransformer (3D Spatial-Temporal) Forward Pass OK. Output Shape: $(size(y_3d))"
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
        st = Lux.testmode(st)

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
        @info "EncoderDecoderModel Forward Pass OK. Output Shape: $(size(y))"
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
        st = Lux.testmode(st)

        # 3. Create dummy data
        enc_x = randn(Float32, grid_size[1], grid_size[2], grid_size[3], in_channels, batch_size)
        dec_x = randn(Float32, dec_features, dec_seq_len, batch_size)

        # 4. Forward pass
        y, st_out = model(enc_x, dec_x, ps, st; enc_causal = false, dec_causal = true)

        # 5. Verify shapes
        @test size(y) == (out_features, dec_seq_len, batch_size)
        @info "VisionEncoderDecoderModel Forward Pass OK. Output Shape: $(size(y))"
    end

    @testset "Direct Multi-Step Forecasting (Climate Scenario)" begin
        # Goal: Predict Net Ecosystem Exchange (NEE) for the next 7 days,
        # using the past 14 days of Temperature and Precipitation.
        # We also know the future Time of Day and Solar Radiation for the 7 forecast days.

        past_horizon = 14
        future_horizon = 7
        batch_size = 8

        # Encoder sees past: Temp, Precip (2 features)
        enc_features = 2

        # Decoder sees known future: TimeOfDay, Radiation (2 features)
        dec_features = 2

        # Output predicts: NEE (1 feature)
        out_features = 1
        d_model = 32

        model = EncoderDecoderModel(
            in_features = enc_features,
            dec_features = dec_features,
            d_model = d_model,
            enc_layers = 2,
            dec_layers = 2,
            n_heads = 2,
            out_features = out_features
        )

        ps, st = Lux.setup(rng, model)
        st = Lux.testmode(st)

        # Create dummy data
        past_covariates = randn(Float32, enc_features, past_horizon, batch_size)
        future_known_covariates = randn(Float32, dec_features, future_horizon, batch_size)

        # Forward pass:
        # Encoder processes the past covariates.
        # Decoder takes the known future covariates and the encoder's memory,
        # and directly predicts the target over the entire future horizon in one shot.
        # Since it's direct multi-step, the decoder does NOT need to be causal!
        # It can look at the entire sequence of future known covariates to predict the targets.
        predicted_nee, st_out = model(past_covariates, future_known_covariates, ps, st; enc_causal = false, dec_causal = false)
        @test size(predicted_nee) == (out_features, future_horizon, batch_size)
        @info "Direct Multi-Step Forecasting (Climate Scenario) OK. Predicted NEE Shape: $(size(predicted_nee))"
    end

    @testset "VisionToVisionModel (Image-to-Image Regression)" begin
        # This test demonstrates the exact macroscopic architecture used by models like Pangu-Weather.
        # It treats weather forecasting as a pure Image-to-Image (or Grid-to-Grid) translation task.
        # For full 3D Earth mapping (Longitude, Latitude, Pressure/Time), simply set ndims=3.

        patch_size = (16, 16)
        grid_size = (2, 2) # e.g. input is 32x32, 32/16 = 2 patches per dim

        # in_channels could represent (Temp, Humidity, etc.) for the current time step
        in_channels = 3

        # out_channels could represent (Temp, Humidity, etc.) for the next hour/day
        out_channels = 1
        d_model = 32
        batch_size = 4

        model = VisionToVisionModel(
            patch_size = patch_size,
            grid_size = grid_size,
            in_channels = in_channels,
            out_channels = out_channels,
            d_model = d_model,
            n_layers = 2,
            n_heads = 2,
            max_positions = 10,
            ndims = 2
        )

        ps, st = Lux.setup(rng, model)
        st = Lux.testmode(st)

        # 32x32 image with 3 channels
        x = randn(Float32, 32, 32, in_channels, batch_size)
        y, st_out = model(x, ps, st)

        # Output should be 32x32 image with 1 channel
        @test size(y) == (32, 32, out_channels, batch_size)
        @info "VisionToVisionModel Forward Pass OK. Output Shape: $(size(y))"
    end

    @testset "VisionToVisionEncoderDecoderModel (Grid-to-Grid Forecasting)" begin
        # This test demonstrates how to forecast future target maps based on a history of covariate maps.
        # Encoder input: Historical covariates (e.g. Past 14 days of grids).
        # Decoder input: Known future covariates (e.g. Next 7 days of solar radiation grids).
        # Output: Predicted future targets (e.g. Next 7 days of NEE grids).
        #
        # Note on 3D (Spatio-Temporal):
        # If you set ndims=3, both the input and output become (W, H, Time, Channels) volumes, seamlessly handling multi-step sequences.

        patch_size = (16, 16)
        grid_size = (2, 2)
        in_channels = 2  # e.g., Temp, Precip (Past grid)
        dec_channels = 1 # e.g., Radiation (Future known covariate grid)
        out_channels = 1 # e.g., predicted NEE map
        d_model = 32
        batch_size = 4

        model = VisionToVisionEncoderDecoderModel(
            patch_size = patch_size,
            grid_size = grid_size,
            in_channels = in_channels,
            dec_channels = dec_channels,
            out_channels = out_channels,
            d_model = d_model,
            enc_layers = 2,
            dec_layers = 2,
            n_heads = 2,
            ndims = 2
        )

        ps, st = Lux.setup(rng, model)
        st = Lux.testmode(st)

        # Past grid
        enc_x = randn(Float32, 32, 32, in_channels, batch_size)
        # Future covariate grid
        dec_x = randn(Float32, 32, 32, dec_channels, batch_size)

        # Forward pass predicting future target grid
        y, st_out = model(enc_x, dec_x, ps, st; enc_causal = false, dec_causal = false)

        @test size(y) == (32, 32, out_channels, batch_size)
        @info "VisionToVisionEncoderDecoderModel Forward Pass OK. Output Shape: $(size(y))"
    end

    @testset "LSTM-Style Regression (Encoder-Only)" begin
        # Goal: Predict a single target at the final time step using ONLY historical covariates.
        # This mirrors a classic LSTM where you feed a sequence and take the final hidden state output.

        in_features = 3
        out_features = 1
        d_model = 16
        batch_size = 4
        past_window = 10

        model = TransformerModel(
            in_features = in_features,
            out_features = out_features,
            d_model = d_model,
            n_layers = 2,
            n_heads = 2
        )

        ps, st = Lux.setup(rng, model)
        st = Lux.testmode(st)

        # x represents the past history (length 10)
        x = randn(Float32, in_features, past_window, batch_size)

        y, st_out = model(x, ps, st; causal = true)

        # The model outputs a prediction for EVERY step.
        # For LSTM-style regression, we simply slice out the final timestep prediction!
        y_final_step = y[:, end:end, :]

        @test size(y_final_step) == (out_features, 1, batch_size)
        @info "LSTM-Style Regression (Encoder-Only) OK. Output Shape: $(size(y_final_step))"
    end

    @testset "LSTM-Style Regression (Encoder-Decoder)" begin
        # Goal: Predict a single concurrent target at the final time step,
        # utilizing a history of past covariates AND known covariates EXACTLY at the target step.

        in_features = 3
        dec_features = 2 # Covariates known concurrently at the target step (e.g., Time of Day)
        out_features = 1
        d_model = 16
        batch_size = 4
        past_window = 10

        model = EncoderDecoderModel(
            in_features = in_features,
            dec_features = dec_features,
            out_features = out_features,
            d_model = d_model,
            enc_layers = 1,
            dec_layers = 1,
            n_heads = 2
        )

        ps, st = Lux.setup(rng, model)
        st = Lux.testmode(st)

        # enc_x represents the past history (length 10)
        enc_x = randn(Float32, in_features, past_window, batch_size)

        # dec_x represents the concurrent features exactly at the target step (length 1)
        dec_x = randn(Float32, dec_features, 1, batch_size)

        y, st_out = model(enc_x, dec_x, ps, st)

        # The decoder naturally outputs length 1 because dec_x has length 1!
        @test size(y) == (out_features, 1, batch_size)
        @info "LSTM-Style Regression (Encoder-Decoder) OK. Output Shape: $(size(y))"
    end
    @testset "Advanced VisionTransformer Features (LayerScale, CLS, Register, extract_features)" begin
        patch_size = (16, 16)
        in_channels = 3
        d_model = 64
        image_size = (32, 32)
        batch_size = 4
        out_features = 10

        vit = VisionTransformer(
            patch_size = patch_size,
            in_channels = in_channels,
            d_model = d_model,
            n_layers = 3,
            n_heads = 4,
            max_positions = 256,
            num_classes = out_features,
            use_rope = true,
            use_cls_token = true,
            n_register_tokens = 4,
            layer_scale_init = 1.0e-5
        )

        ps, st = Lux.setup(rng, vit)
        st = Lux.testmode(st)

        x = randn(Float32, image_size[1], image_size[2], in_channels, batch_size)

        # 1. Forward pass testing CLS token extraction and LayerScale
        y, st_out = vit(x, ps, st)
        @test size(y) == (out_features, batch_size)
        @info "Advanced ViT Features Forward Pass OK. Output Shape: $(size(y))"

        # 2. Test extract_features with n_blocks
        # Expected grid size: (32/16, 32/16) = (2, 2)
        # Expected output per block: (2, 2, d_model, batch_size) = (2, 2, 64, 4)
        features = extract_features(vit, x, ps, st; n_blocks = 2)

        @test length(features) == 2
        @test size(features[1]) == (2, 2, d_model, batch_size)
        @test size(features[2]) == (2, 2, d_model, batch_size)
        @info "extract_features (n_blocks) OK. Extracted $(length(features)) blocks of shape $(size(features[1]))."

        # 3. Test extract_features with specific blocks
        features_specific = extract_features(vit, x, ps, st; blocks = [1, 3])
        @test length(features_specific) == 2
        @test size(features_specific[1]) == (2, 2, d_model, batch_size)
        @test size(features_specific[2]) == (2, 2, d_model, batch_size)
        @info "extract_features (specific blocks) OK. Extracted blocks 1 and 3."
    end
end
