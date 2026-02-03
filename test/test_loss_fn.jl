using Statistics
using EasyHybrid: bestdirection, isbetter, check_training_loss, Minimize, Maximize

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
        nse = 1 - sum((ŷ .- y) .^ 2) / sum((y .- mean(y)) .^ 2)
        @test loss_fn(ŷ, y, y_nan, Val(:nse)) ≈ nse

        # PearsonLoss test (1 - Pearson correlation)
        r = cor(ŷ, y)
        @test loss_fn(ŷ, y, y_nan, Val(:pearsonLoss)) ≈ 1.0 - r

        # NSELoss test
        nse_loss = sum((ŷ .- y) .^ 2) / sum((y .- mean(y)) .^ 2)
        @test loss_fn(ŷ, y, y_nan, Val(:nseLoss)) ≈ nse_loss

        # KGE Loss test
        μ_s = mean(ŷ)
        μ_o = mean(y)
        σ_s = std(ŷ)
        σ_o = std(y)
        r = cor(ŷ, y)
        α = σ_s / σ_o
        β = μ_s / μ_o
        kge_loss = sqrt((r - 1.0)^2 + (α - 1.0)^2 + (β - 1.0)^2)
        @test loss_fn(ŷ, y, y_nan, Val(:kgeLoss)) ≈ kge_loss

        # KGE test (1 - KGE Loss)
        @test loss_fn(ŷ, y, y_nan, Val(:kge)) ≈ 1.0 - kge_loss

        # PBKGE Loss test (Partial Kling-Gupta Efficiency)
        μ_s = mean(ŷ)
        μ_o = mean(y)
        r = cor(ŷ, y)
        β = μ_s / μ_o
        pbkge_loss = sqrt((r - 1.0)^2 + (β - 1.0)^2)
        @test loss_fn(ŷ, y, y_nan, Val(:pbkgeLoss)) ≈ pbkge_loss

        # PBKGE test (1 - PBKGE Loss)
        @test loss_fn(ŷ, y, y_nan, Val(:pbkge)) ≈ 1.0 - pbkge_loss
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
        @test loss_fn(ŷ, y, y_nan, Val(:mae)) ≈ mean(abs, valid_ŷ .- valid_y)
        @test loss_fn(ŷ, y, y_nan, Val(:pearson)) ≈ cor(valid_ŷ, valid_y)

        r = cor(valid_ŷ, valid_y)
        @test loss_fn(ŷ, y, y_nan, Val(:r2)) ≈ r^2

        nse = 1 - sum((valid_ŷ .- valid_y) .^ 2) / sum((valid_y .- mean(valid_y)) .^ 2)
        @test loss_fn(ŷ, y, y_nan, Val(:nse)) ≈ nse
        @test loss_fn(ŷ, y, y_nan, Val(:pearsonLoss)) ≈ 1.0 - r

        nse_loss = sum((valid_ŷ .- valid_y) .^ 2) / sum((valid_y .- mean(valid_y)) .^ 2)
        @test loss_fn(ŷ, y, y_nan, Val(:nseLoss)) ≈ nse_loss

        # KGE Loss with NaN handling
        μ_s = mean(valid_ŷ)
        μ_o = mean(valid_y)
        σ_s = std(valid_ŷ)
        σ_o = std(valid_y)
        r = cor(valid_ŷ, valid_y)
        α = σ_s / σ_o
        β = μ_s / μ_o
        kge_loss = sqrt((r - 1.0)^2 + (α - 1.0)^2 + (β - 1.0)^2)
        @test loss_fn(ŷ, y, y_nan, Val(:kgeLoss)) ≈ kge_loss
        @test loss_fn(ŷ, y, y_nan, Val(:kge)) ≈ 1.0 - kge_loss

        # PBKGE Loss with NaN handling
        μ_s = mean(valid_ŷ)
        μ_o = mean(valid_y)
        r = cor(valid_ŷ, valid_y)
        β = μ_s / μ_o
        pbkge_loss = sqrt((r - 1.0)^2 + (β - 1.0)^2)
        @test loss_fn(ŷ, y, y_nan, Val(:pbkgeLoss)) ≈ pbkge_loss
        @test loss_fn(ŷ, y, y_nan, Val(:pbkge)) ≈ 1.0 - pbkge_loss

        # Test NaN handling for generic functions
        @test loss_fn(ŷ, y, y_nan, simple_loss) ≈ mean(abs2, valid_ŷ .- valid_y)
        @test loss_fn(ŷ, y, y_nan, (weighted_loss, (2.0,))) ≈ 2.0 * mean(abs2, valid_ŷ .- valid_y)
    end

    @testset "bestdirection" begin
        # Test that metrics to be maximized return Maximize
        @test bestdirection(Val(:pearson)) isa Maximize
        @test bestdirection(Val(:r2)) isa Maximize
        @test bestdirection(Val(:nse)) isa Maximize
        @test bestdirection(Val(:kge)) isa Maximize

        # Test that losses to be minimized return Minimize
        @test bestdirection(Val(:mse)) isa Minimize
        @test bestdirection(Val(:rmse)) isa Minimize
        @test bestdirection(Val(:mae)) isa Minimize
        @test bestdirection(Val(:pearsonLoss)) isa Minimize
        @test bestdirection(Val(:nseLoss)) isa Minimize
        @test bestdirection(Val(:kgeLoss)) isa Minimize
        @test bestdirection(Val(:pbkgeLoss)) isa Minimize
        @test bestdirection(Val(:pbkge)) isa Minimize

        # Test default case (anything else should be Minimize)
        @test bestdirection(Val(:unknown)) isa Minimize
    end

    @testset "isbetter" begin
        # Test isbetter for minimized metrics (smaller is better)
        @test isbetter(0.5, 1.0, :mse) == true
        @test isbetter(1.0, 0.5, :mse) == false
        @test isbetter(0.5, 0.5, :mse) == false  # equal is not better
        @test isbetter(0.3, 0.5, :rmse) == true
        @test isbetter(0.5, 0.3, :rmse) == false

        # Test isbetter for maximized metrics (larger is better)
        @test isbetter(0.8, 0.5, :pearson) == true
        @test isbetter(0.5, 0.8, :pearson) == false
        @test isbetter(0.5, 0.5, :pearson) == false  # equal is not better
        @test isbetter(0.9, 0.7, :r2) == true
        @test isbetter(0.7, 0.9, :r2) == false
        @test isbetter(0.85, 0.75, :nse) == true
        @test isbetter(0.75, 0.85, :nse) == false
        @test isbetter(0.9, 0.8, :kge) == true
        @test isbetter(0.8, 0.9, :kge) == false

        # Test with Minimize and Maximize types directly
        @test isbetter(0.5, 1.0, Minimize()) == true
        @test isbetter(1.0, 0.5, Minimize()) == false
        @test isbetter(0.8, 0.5, Maximize()) == true
        @test isbetter(0.5, 0.8, Maximize()) == false
    end

    @testset "check_training_loss" begin
        # Test that maximized metrics throw an error
        @test_throws ErrorException check_training_loss(:pearson)
        @test_throws ErrorException check_training_loss(:r2)
        @test_throws ErrorException check_training_loss(:nse)
        @test_throws ErrorException check_training_loss(:kge)

        # Test that minimized losses pass (return nothing)
        @test check_training_loss(:mse) === nothing
        @test check_training_loss(:rmse) === nothing
        @test check_training_loss(:mae) === nothing
        @test check_training_loss(:pearsonLoss) === nothing
        @test check_training_loss(:nseLoss) === nothing
        @test check_training_loss(:kgeLoss) === nothing
        @test check_training_loss(:pbkgeLoss) === nothing
        @test check_training_loss(:pbkge) === nothing
    end
end
