using Test
using EasyHybrid
using Statistics
using DimensionalData
import EasyHybrid: compute_loss

@testset "LoggingLoss" begin
    @testset "Constructor defaults" begin
        logging = LoggingLoss()
        @test logging.loss_types == [:mse]
        @test logging.training_loss == :mse
        @test logging.agg == sum
        @test logging.train_mode == true
    end

    @testset "Custom constructor" begin
        # Simple custom loss function
        custom_loss(ŷ, y) = mean(abs2, ŷ .- y)
        
        # Loss function with args
        weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
        
        # Loss function with kwargs
        scaled_loss(ŷ, y; scale=1.0) = scale * mean(abs2, ŷ .- y)

        @testset "Basic custom constructor" begin
            logging = LoggingLoss(
                loss_types=[:mse, :mae],
                training_loss=:mae,
                agg=mean,
                train_mode=false
            )
            @test logging.loss_types == [:mse, :mae]
            @test logging.training_loss == :mae
            @test logging.agg == mean
            @test logging.train_mode == false
        end

        @testset "Mixed loss_types" begin
            logging = LoggingLoss(
                loss_types=[:mse, custom_loss, (weighted_loss, (0.5,)), (scaled_loss, (scale=2.0,))],
                training_loss=:mse,
                agg=sum
            )
            @test length(logging.loss_types) == 4
            @test logging.loss_types[1] == :mse
            @test logging.loss_types[2] == custom_loss
            @test logging.loss_types[3] == (weighted_loss, (0.5,))
            @test logging.loss_types[4] == (scaled_loss, (scale=2.0,))
        end

        @testset "Custom training_loss variations" begin
            # Function as training_loss
            logging = LoggingLoss(
                loss_types=[:mse],
                training_loss=custom_loss
            )
            @test logging.training_loss == custom_loss

            # Tuple with args as training_loss
            logging = LoggingLoss(
                loss_types=[:mse],
                training_loss=(weighted_loss, (0.5,))
            )
            @test logging.training_loss == (weighted_loss, (0.5,))

            # Tuple with kwargs as training_loss
            logging = LoggingLoss(
                loss_types=[:mse],
                training_loss=(scaled_loss, (scale=2.0,))
            )
            @test logging.training_loss == (scaled_loss, (scale=2.0,))

            # Tuple with both args and kwargs
            complex_loss(x, y, w; scale=1.0) = scale * w * mean(abs2, x .- y)
            logging = LoggingLoss(
                loss_types=[:mse],
                training_loss=(complex_loss, (0.5,), (scale=2.0,))
            )
            @test logging.training_loss == (complex_loss, (0.5,), (scale=2.0,))
        end
    end
end

@testset "compute_loss" begin
    # Test data setup
    ŷ = Dict(:var1 => [1.0, 2.0, 3.0], :var2 => [2.0, 3.0, 4.0])
    y(target) = target == :var1 ? [1.1, 1.9, 3.2] : [1.8, 3.1, 3.9]
    y_sigma(target) = target == :var1 ? [0.1, 0.2, 0.1] : [0.2, 0.1, 0.2]
    y_nan(target) = trues(3)
    targets = [:var1, :var2]

    @testset "Predefined losses" begin
        # Test single predefined loss
        loss = compute_loss(ŷ, y, y_nan, targets, :mse, sum)
        @test loss isa Number
        
        # Test multiple predefined losses
        losses = compute_loss(ŷ, y, y_nan, targets, [:mse, :mae], sum)
        @test losses isa NamedTuple
        @test haskey(losses, :mse)
        @test haskey(losses, :mae)
    end

    @testset "Custom loss functions" begin
        # Simple custom loss
        custom_loss(ŷ, y) = mean(abs2, ŷ .- y)
        loss = compute_loss(ŷ, y, y_nan, targets, custom_loss, sum)
        @test loss isa Number

        # Custom loss with args
        weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
        loss = compute_loss(ŷ, y, y_nan, targets, (weighted_loss, (2.0,)), sum)
        @test loss isa Number

        # Custom loss with kwargs
        scaled_loss(ŷ, y; scale=1.0) = scale * mean(abs2, ŷ .- y)
        loss = compute_loss(ŷ, y, y_nan, targets, (scaled_loss, (scale=2.0,)), sum)
        @test loss isa Number

        # Custom loss with both
        complex_loss(ŷ, y, w; scale=1.0) = scale * w * mean(abs2, ŷ .- y)
        loss = compute_loss(ŷ, y, y_nan, targets, (complex_loss, (0.5,), (scale=2.0,)), sum)
        @test loss isa Number

        # custom loss with uncertainty
        function custom_loss_uncertainty(ŷ, y_and_sigma)
            y_vals, y_σ = y_and_sigma
            return mean(((ŷ .- y_vals).^2) ./ (y_σ .^2 .+ 1e-6))
        end
        loss = compute_loss(ŷ, (y, y_sigma), y_nan, targets, custom_loss_uncertainty, sum)
        @test loss isa Number
        # a single sigma number
        # loss = compute_loss(ŷ, (y, 0.01), y_nan, targets, custom_loss_uncertainty, sum) # TODO
        # @test loss isa Number
        losses = compute_loss(ŷ, (y, y_sigma), y_nan, targets, [custom_loss_uncertainty,], sum)
        @test losses isa NamedTuple
    end

    @testset "DimensionalData interface" begin
        # Create test DimensionalArrays
        ŷ_dim = Dict(
            :var1 => DimArray([1.0, 2.0, 3.0], (Ti(1:3),)),
            :var2 => DimArray([2.0, 3.0, 4.0], (Ti(1:3),))
        )
        y_dim = DimArray([1.1 1.8; 1.9 3.1; 3.2 3.9], (Ti(1:3), Dim{:col}([:var1, :var2])))
        y_nan_dim = DimArray(trues(3,2), (Ti(1:3), Dim{:col}([:var1, :var2])))

        # Test single predefined loss
        loss = compute_loss(ŷ_dim, y_dim, y_nan_dim, targets, :mse, sum)
        @test loss isa Number

        # Test multiple predefined losses
        losses = compute_loss(ŷ_dim, y_dim, y_nan_dim, targets, [:mse, :mae], sum)
        @test losses isa NamedTuple
        @test haskey(losses, :mse)
        @test haskey(losses, :mae)
    end

    @testset "Loss value correctness" begin
        # Test MSE calculation
        mse_loss = compute_loss(ŷ, y, y_nan, targets, :mse, sum)
        expected_mse = sum(mean(abs2, ŷ[k] .- y(k)) for k in targets)
        @test mse_loss ≈ expected_mse

        # Test MAE calculation
        mae_loss = compute_loss(ŷ, y, y_nan, targets, :mae, sum)
        expected_mae = sum(mean(abs, ŷ[k] .- y(k)) for k in targets)
        @test mae_loss ≈ expected_mae
    end

    @testset "Edge cases" begin
        # Empty targets
        @test_throws ArgumentError compute_loss(ŷ, y, y_nan, String[], :mse, sum)

        # Single target
        single_target = [:var1]
        loss = compute_loss(ŷ, y, y_nan, single_target, :mse, sum)
        @test loss isa Number

        # NaN handling
        y_nan_with_false(target) = [true, false, true]
        loss = compute_loss(ŷ, y, y_nan_with_false, targets, :mse, sum)
        @test !isnan(loss)
    end
end