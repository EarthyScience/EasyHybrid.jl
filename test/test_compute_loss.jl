using EasyHybrid: _compute_loss, PerTarget, _apply_loss, loss_fn
using Statistics
using DimensionalData

@testset "_compute_loss" begin
    # Test data setup
    ŷ = Dict(:var1 => [1.0, 2.0, 3.0], :var2 => [2.0, 3.0, 4.0])
    y(target) = target == :var1 ? [1.1, 1.9, 3.2] : [1.8, 3.1, 3.9]
    y_nan(target) = trues(3)
    targets = [:var1, :var2]

    @testset "Predefined losses" begin
        # Test single predefined loss
        loss = _compute_loss(ŷ, y, y_nan, targets, :mse, sum)
        @test loss isa Number

        # Test multiple predefined losses
        losses = _compute_loss(ŷ, y, y_nan, targets, [:mse, :mae], sum)
        @test losses isa NamedTuple
        @test haskey(losses, :mse)
        @test haskey(losses, :mae)
    end

    @testset "Custom loss functions" begin
        # Simple custom loss
        custom_loss(ŷ, y) = mean(abs2, ŷ .- y)
        loss = _compute_loss(ŷ, y, y_nan, targets, custom_loss, sum)
        @test loss isa Number

        # Custom loss with args
        weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
        loss = _compute_loss(ŷ, y, y_nan, targets, (weighted_loss, (2.0,)), sum)
        @test loss isa Number

        # Custom loss with kwargs
        scaled_loss(ŷ, y; scale = 1.0) = scale * mean(abs2, ŷ .- y)
        loss = _compute_loss(ŷ, y, y_nan, targets, (scaled_loss, (scale = 2.0,)), sum)
        @test loss isa Number

        # Custom loss with both
        complex_loss(ŷ, y, w; scale = 1.0) = scale * w * mean(abs2, ŷ .- y)
        loss = _compute_loss(ŷ, y, y_nan, targets, (complex_loss, (0.5,), (scale = 2.0,)), sum)
        @test loss isa Number

        @testset "Per-target losses" begin
            # Mix of predefined and custom
            loss_spec = PerTarget((:mse, custom_loss))
            loss_d = _compute_loss(ŷ, y, y_nan, targets, loss_spec, sum)
            l_mse = loss_fn(ŷ[:var1], y(:var1), y_nan(:var1), Val(:mse))
            l_custom = _apply_loss(ŷ[:var2], y(:var2), y_nan(:var2), custom_loss)
            @test loss_d ≈ l_mse + l_custom

            # Mix of custom losses with arguments
            loss_spec_args = PerTarget(((weighted_loss, (0.5,)), (scaled_loss, (scale = 2.0,))))
            loss_args = _compute_loss(ŷ, y, y_nan, targets, loss_spec_args, sum)
            l_weighted = _apply_loss(ŷ[:var2], y(:var2), y_nan(:var2), (weighted_loss, (0.5,)))
            l_scaled = _apply_loss(ŷ[:var2], y(:var2), y_nan(:var2), (scaled_loss, (scale = 2.0,)))
            @test loss_args ≈ l_weighted + l_scaled

            # Mismatched number of losses and targets
            @test_throws AssertionError _compute_loss(ŷ, y, y_nan, targets, PerTarget((:mse,)), sum)
        end
    end

    @testset "DimensionalData interface" begin
        # Create test DimensionalArrays
        ŷ_dim = Dict(
            :var1 => DimArray([1.0, 2.0, 3.0], (Ti(1:3),)),
            :var2 => DimArray([2.0, 3.0, 4.0], (Ti(1:3),))
        )
        y_dim = DimArray([1.1 1.8; 1.9 3.1; 3.2 3.9], (Ti(1:3), Dim{:col}([:var1, :var2])))
        y_nan_dim = DimArray(trues(3, 2), (Ti(1:3), Dim{:col}([:var1, :var2])))

        # Test single predefined loss
        loss = _compute_loss(ŷ_dim, y_dim, y_nan_dim, targets, :mse, sum)
        @test loss isa Number

        # Test multiple predefined losses
        losses = _compute_loss(ŷ_dim, y_dim, y_nan_dim, targets, [:mse, :mae], sum)
        @test losses isa NamedTuple
        @test haskey(losses, :mse)
        @test haskey(losses, :mae)
    end

    @testset "Loss value correctness" begin
        # Test MSE calculation
        mse_loss = _compute_loss(ŷ, y, y_nan, targets, :mse, sum)
        expected_mse = sum(mean(abs2, ŷ[k] .- y(k)) for k in targets)
        @test mse_loss ≈ expected_mse

        # Test MAE calculation
        mae_loss = _compute_loss(ŷ, y, y_nan, targets, :mae, sum)
        expected_mae = sum(mean(abs, ŷ[k] .- y(k)) for k in targets)
        @test mae_loss ≈ expected_mae
    end

    @testset "Edge cases" begin
        # Empty targets
        @test_throws ArgumentError _compute_loss(ŷ, y, y_nan, String[], :mse, sum)

        # Single target
        single_target = [:var1]
        loss = _compute_loss(ŷ, y, y_nan, single_target, :mse, sum)
        @test loss isa Number

        # NaN handling
        y_nan_with_false(target) = [true, false, true]
        loss = _compute_loss(ŷ, y, y_nan_with_false, targets, :mse, sum)
        @test !isnan(loss)
    end
end
