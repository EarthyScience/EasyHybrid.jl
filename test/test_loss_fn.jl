using Test
using EasyHybrid
using Statistics

@testset "loss_fn methods" begin
    # Test data setup
    ŷ = [1.0, 2.0, 3.0, 4.0]
    y = [1.1, 1.9, 3.2, 3.8]
    y_nan = trues(4)  # all values are valid

    simple_loss(ŷ, y) = mean(abs2, ŷ .- y)
    weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
    scaled_loss(ŷ, y; scale = 1.0) = scale * mean(abs2, ŷ .- y)
    complex_loss(ŷ, y, w; scale = 1.0) = scale * w * mean(abs2, ŷ .- y)

    @testset "Predefined loss functions" begin
        # RMSE test
        @test loss_fn(ŷ, y, y_nan, Val(:rmse)) ≈ sqrt(mean(abs2, ŷ .- y))

        # MSE test
        @test loss_fn(ŷ, y, y_nan, Val(:mse)) ≈ mean(abs2, ŷ .- y)

        # MAE test
        @test loss_fn(ŷ, y, y_nan, Val(:mae)) ≈ mean(abs, ŷ .- y)

        # Pearson correlation test
        @test loss_fn(ŷ, y, y_nan, Val(:pearson)) ≈ cor(ŷ, y)

        # R² test
        r = cor(ŷ, y)
        @test loss_fn(ŷ, y, y_nan, Val(:r2)) ≈ r^2

        # NSE test
        nse = sum((ŷ .- y) .^ 2) / sum((y .- mean(y)) .^ 2)
        @test loss_fn(ŷ, y, y_nan, Val(:nse)) ≈ nse
    end

    @testset "Generic loss functions" begin
        # Simple function with no extra arguments
        @test loss_fn(ŷ, y, y_nan, simple_loss) ≈ mean(abs2, ŷ .- y)

        # Function with positional arguments
        @test loss_fn(ŷ, y, y_nan, (weighted_loss, (2.0,))) ≈ 2.0 * mean(abs2, ŷ .- y)

        # Function with keyword arguments
        @test loss_fn(ŷ, y, y_nan, (scaled_loss, (scale = 2.0,))) ≈ 2.0 * mean(abs2, ŷ .- y)

        # Function with both positional and keyword arguments
        @test loss_fn(ŷ, y, y_nan, (complex_loss, (2.0,), (scale = 3.0,))) ≈ 6.0 * mean(abs2, ŷ .- y)
    end

    @testset "NaN handling" begin
        y_nan = [true, true, false, true]
        valid_ŷ = ŷ[y_nan]
        valid_y = y[y_nan]

        # Test NaN handling for predefined functions
        @test loss_fn(ŷ, y, y_nan, Val(:mse)) ≈ mean(abs2, valid_ŷ .- valid_y)
        @test loss_fn(ŷ, y, y_nan, Val(:rmse)) ≈ sqrt(mean(abs2, valid_ŷ .- valid_y))

        # Test NaN handling for generic functions
        @test loss_fn(ŷ, y, y_nan, simple_loss) ≈ mean(abs2, valid_ŷ .- valid_y)
        @test loss_fn(ŷ, y, y_nan, (weighted_loss, (2.0,))) ≈ 2.0 * mean(abs2, valid_ŷ .- valid_y)
    end
end
