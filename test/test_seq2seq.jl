using Test
using EasyHybrid

@testset "Data Preprocessing: split_seq2seq" begin
    @testset "2D Time Series (Features, Time) - Forecasting" begin
        features = 2
        time = 20
        enc_window = 7
        dec_window = 3

        # Shape: (features, time)
        x = reshape(1:(features * time), features, time)
        y = reshape(1:(features * time), features, time)

        enc_x, dec_x, y_target = split_seq2seq(x, y; enc_window, dec_window, lead_time = 1)

        # No forcings provided, dec_x should be nothing
        @test isnothing(dec_x)

        num_samples = time - enc_window - dec_window + 1
        @test num_samples == 11
        @test size(enc_x) == (features, enc_window, num_samples)
        @test size(y_target) == (features, dec_window, num_samples)

        # Check first sample
        @test enc_x[:, :, 1] == x[:, 1:7]
        @test y_target[:, :, 1] == y[:, 8:10]

        # Check last sample
        @test enc_x[:, :, 11] == x[:, 11:17]
        @test y_target[:, :, 11] == y[:, 18:20]
    end

    @testset "3D Array (Spatial, Features, Time)" begin
        spatial = 5
        features = 2
        time = 15
        enc_window = 4
        dec_window = 2

        x = randn(Float32, spatial, features, time)
        forcings = randn(Float32, spatial, 1, time)
        y = randn(Float32, spatial, features, time)

        enc_x, dec_x, y_target = split_seq2seq(x, forcings, y; enc_window, dec_window)

        num_samples = time - enc_window - dec_window + 1
        @test num_samples == 10
        @test size(enc_x) == (spatial, features, enc_window, num_samples)
        @test size(dec_x) == (spatial, 1, dec_window, num_samples)
        @test size(y_target) == (spatial, features, dec_window, num_samples)

        # Check an arbitrary sample
        @test enc_x[:, :, :, 3] == x[:, :, 3:6]
        @test dec_x[:, :, :, 3] == forcings[:, :, 7:8]
        @test y_target[:, :, :, 3] == y[:, :, 7:8]
    end

    @testset "4D Spatio-Temporal Maps (W, H, Features, Time)" begin
        W, H = 8, 8
        features = 3
        dec_features = 1
        time = 24
        enc_window = 14
        dec_window = 7

        x = randn(Float32, W, H, features, time)
        forcings = randn(Float32, W, H, dec_features, time)
        y = randn(Float32, W, H, features, time)

        enc_x, dec_x, y_target = split_seq2seq(x, forcings, y; enc_window, dec_window)

        num_samples = time - enc_window - dec_window + 1
        @test num_samples == 4

        @test size(enc_x) == (W, H, features, enc_window, num_samples)
        @test size(dec_x) == (W, H, dec_features, dec_window, num_samples)
        @test size(y_target) == (W, H, features, dec_window, num_samples)

        # Check last sample
        @test enc_x[:, :, :, :, 4] == x[:, :, :, 4:17]
        @test dec_x[:, :, :, :, 4] == forcings[:, :, :, 18:24]
        @test y_target[:, :, :, :, 4] == y[:, :, :, 18:24]
    end

    @testset "LSTM Regression (lead_time = 0)" begin
        features = 2
        time = 20
        enc_window = 7
        dec_window = 1

        x = reshape(1:(features * time), features, time)
        y = reshape(1:(features * time), features, time)

        # lead_time = 0 means the target is concurrent with the last encoder step
        enc_x, dec_x, y_target = split_seq2seq(x, y; enc_window, dec_window, lead_time = 0)

        num_samples = time - enc_window - dec_window - 0 + 2
        @test num_samples == 14

        # Check first sample
        @test enc_x[:, :, 1] == x[:, 1:7]
        # Target should be exactly at time step 7
        @test y_target[:, :, 1] == y[:, 7:7]

        # Check last sample
        @test enc_x[:, :, end] == x[:, 14:20]
        # Target should be exactly at time step 20
        @test y_target[:, :, end] == y[:, 20:20]
    end
end
