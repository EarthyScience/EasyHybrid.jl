using EasyHybrid: _compute_loss, PerTarget, _apply_loss, loss_fn
using EasyHybrid: _get_target_nan, _get_target_y, _loss_name, compute_loss, LoggingLoss
using EasyHybrid: constructHybridModel, to_keyedArray
using Statistics
using DimensionalData
using Random
using DataFrames

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

@testset "_get_target_nan" begin
    # Test with function
    y_nan_func(target) = target == :var1 ? [true, false, true] : [true, true, false]
    @test _get_target_nan(y_nan_func, :var1) == [true, false, true]
    @test _get_target_nan(y_nan_func, :var2) == [true, true, false]

    # Test with AbstractDimArray
    y_nan_dim = DimArray([true false; true true; false true], (Ti(1:3), Dim{:col}([:var1, :var2])))
    @test _get_target_nan(y_nan_dim, :var1) == [true, true, false]
    @test _get_target_nan(y_nan_dim, :var2) == [false, true, true]

    # Test with Vector of targets
    y_nan_dim_multi = DimArray([true false; true true; false true], (Ti(1:3), Dim{:col}([:var1, :var2])))
    result = _get_target_nan(y_nan_dim_multi, [:var1, :var2])
    @test size(result) == (3, 2)
    @test result[:, 1] == [true, true, false]
    @test result[:, 2] == [false, true, true]
end

@testset "_get_target_y" begin
    # Test with function
    y_func(target) = target == :var1 ? [1.0, 2.0, 3.0] : [2.0, 3.0, 4.0]
    @test _get_target_y(y_func, :var1) == [1.0, 2.0, 3.0]
    @test _get_target_y(y_func, :var2) == [2.0, 3.0, 4.0]

    # Test with AbstractDimArray
    y_dim = DimArray([1.0 2.0; 2.0 3.0; 3.0 4.0], (Ti(1:3), Dim{:col}([:var1, :var2])))
    @test _get_target_y(y_dim, :var1) == [1.0, 2.0, 3.0]
    @test _get_target_y(y_dim, :var2) == [2.0, 3.0, 4.0]

    # Test with Vector of targets
    y_dim_multi = DimArray([1.0 2.0; 2.0 3.0; 3.0 4.0], (Ti(1:3), Dim{:col}([:var1, :var2])))
    result = _get_target_y(y_dim_multi, [:var1, :var2])
    @test size(result) == (3, 2)
    @test result[:, 1] == [1.0, 2.0, 3.0]
    @test result[:, 2] == [2.0, 3.0, 4.0]

    # Test with Tuple (y_obs, y_sigma) where y_sigma is a Number
    y_obs_func(target) = target == :var1 ? [1.0, 2.0, 3.0] : [2.0, 3.0, 4.0]
    y_sigma_num = 0.5
    y_tuple_num = (y_obs_func, y_sigma_num)
    result = _get_target_y(y_tuple_num, :var1)
    @test result isa Tuple
    @test result[1] == [1.0, 2.0, 3.0]
    @test result[2] == 0.5

    # Test with Tuple (y_obs, y_sigma) where y_sigma is a function
    y_sigma_func(target) = target == :var1 ? 0.3 : 0.7
    y_tuple_func = (y_obs_func, y_sigma_func)
    result = _get_target_y(y_tuple_func, :var1)
    @test result isa Tuple
    @test result[1] == [1.0, 2.0, 3.0]
    @test result[2] == 0.3
    result2 = _get_target_y(y_tuple_func, :var2)
    @test result2[1] == [2.0, 3.0, 4.0]
    @test result2[2] == 0.7
end

@testset "_loss_name" begin
    # Test with Symbol
    @test _loss_name(:mse) == :mse
    @test _loss_name(:mae) == :mae
    @test _loss_name(:rmse) == :rmse

    # Test with Function
    custom_loss(ŷ, y) = mean(abs2, ŷ .- y)
    loss_name_func = _loss_name(custom_loss)
    @test loss_name_func isa Symbol
    # The name should be cleaned (remove # if present)
    @test !occursin("#", string(loss_name_func))

    # Test with Tuple (function with args)
    weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
    loss_name_tuple = _loss_name((weighted_loss, (0.5,)))
    @test loss_name_tuple isa Symbol
    @test loss_name_tuple == _loss_name(weighted_loss)

    # Test with Tuple (function with kwargs)
    scaled_loss(ŷ, y; scale = 1.0) = scale * mean(abs2, ŷ .- y)
    loss_name_tuple_kw = _loss_name((scaled_loss, (scale = 2.0,)))
    @test loss_name_tuple_kw isa Symbol
    @test loss_name_tuple_kw == _loss_name(scaled_loss)

    # Test with Tuple (function with both args and kwargs)
    complex_loss(ŷ, y, w; scale = 1.0) = scale * w * mean(abs2, ŷ .- y)
    loss_name_tuple_both = _loss_name((complex_loss, (0.5,), (scale = 2.0,)))
    @test loss_name_tuple_both isa Symbol
    @test loss_name_tuple_both == _loss_name(complex_loss)
end

@testset "compute_loss with extra_loss" begin
    # Simple mechanistic model for testing
    function test_mechanistic_model(; x1, a, b)
        return (; var1 = a .* x1 .+ b, var2 = 2.0f0 .* a .* x1 .+ b)
    end

    # Test parameters
    test_parameters = (
        a = (1.0f0, 0.0f0, 5.0f0),
        b = (2.0f0, 0.0f0, 10.0f0),
    )

    # Create hybrid model
    predictors = [:x2, :x3]
    forcing = [:x1]
    targets = [:var1, :var2]
    neural_param_names = [:a]
    global_param_names = [:b]

    HM = constructHybridModel(
        predictors,
        forcing,
        targets,
        test_mechanistic_model,
        test_parameters,
        neural_param_names,
        global_param_names;
        hidden_layers = [8, 8],
        activation = tanh
    )

    # Setup model parameters and state
    rng = Random.default_rng(314159)
    ps, st = LuxCore.setup(rng, HM)

    # Create test data as KeyedArray (all columns together)
    n_samples = 3
    df_test = DataFrame(
        x1 = Float32.([10.0, 11.0, 12.0]),
        x2 = Float32.([1.0, 2.0, 3.0]),
        x3 = Float32.([4.0, 5.0, 6.0]),
        var1 = Float32.([1.1, 1.9, 3.2]),
        var2 = Float32.([1.8, 3.1, 3.9])
    )
    x = to_keyedArray(df_test)

    # Create target data functions
    y_t(target) = target == :var1 ? df_test.var1 : df_test.var2
    y_nan(target) = trues(n_samples)

    @testset "Training mode with extra_loss" begin
        # Define extra loss function
        extra_loss_func(ŷ) = [sum(abs, ŷ.var1), sum(abs, ŷ.var2)]

        logging = LoggingLoss(
            loss_types = [:mse],
            training_loss = :mse,
            extra_loss = extra_loss_func,
            train_mode = true
        )

        loss_value, st_out, stats = compute_loss(HM, ps, st, (x, (y_t, y_nan)); logging = logging)

        # Should be a single number (aggregated main loss + extra loss)
        @test loss_value isa Number
        @test stats == NamedTuple()

        # Get actual predictions from the model
        ŷ_actual, _ = HM(x, ps, st)

        # Verify the loss includes extra loss
        main_loss = _compute_loss(
            ŷ_actual, y_t, y_nan, targets, :mse, sum
        )
        extra_loss_vals = extra_loss_func(ŷ_actual)
        expected_loss = sum([main_loss, extra_loss_vals...])
        @test loss_value ≈ expected_loss
    end

    @testset "Training mode without extra_loss" begin
        logging = LoggingLoss(
            loss_types = [:mse],
            training_loss = :mse,
            extra_loss = nothing,
            train_mode = true
        )

        loss_value, st_out, stats = compute_loss(HM, ps, st, (x, (y_t, y_nan)); logging = logging)

        @test loss_value isa Number
        @test stats == NamedTuple()

        # Get actual predictions from the model
        ŷ_actual, _ = HM(x, ps, st)

        # Should match the main loss only
        main_loss = _compute_loss(
            ŷ_actual, y_t, y_nan, targets, :mse, sum
        )
        @test loss_value ≈ main_loss
    end

    @testset "Evaluation mode with extra_loss" begin
        # Define extra loss function that returns a NamedTuple
        extra_loss_func(ŷ) = (var1_extra = sum(abs, ŷ.var1), var2_extra = sum(abs, ŷ.var2))

        logging = LoggingLoss(
            loss_types = [:mse, :mae],
            training_loss = :mse,
            extra_loss = extra_loss_func,
            train_mode = false
        )

        loss_value, st_out, stats = compute_loss(HM, ps, st, (x, (y_t, y_nan)); logging = logging)

        # Should be a NamedTuple with loss_types and extra_loss
        @test loss_value isa NamedTuple
        @test haskey(loss_value, :mse)
        @test haskey(loss_value, :mae)
        @test haskey(loss_value, :extra_loss)

        # Check extra_loss structure
        @test loss_value.extra_loss isa NamedTuple
        @test haskey(loss_value.extra_loss, :var1_extra)
        @test haskey(loss_value.extra_loss, :var2_extra)
        @test haskey(loss_value.extra_loss, :sum)  # aggregated extra loss

        # Check stats contains predictions
        @test stats isa NamedTuple
        @test haskey(stats, :var1)
        @test haskey(stats, :var2)
    end

    @testset "Evaluation mode without extra_loss" begin
        logging = LoggingLoss(
            loss_types = [:mse, :mae],
            training_loss = :mse,
            extra_loss = nothing,
            train_mode = false
        )

        loss_value, st_out, stats = compute_loss(HM, ps, st, (x, (y_t, y_nan)); logging = logging)

        # Should be a NamedTuple with only loss_types
        @test loss_value isa NamedTuple
        @test haskey(loss_value, :mse)
        @test haskey(loss_value, :mae)
        @test !haskey(loss_value, :extra_loss)

        # Check stats contains predictions
        @test stats isa NamedTuple
        @test haskey(stats, :var1)
        @test haskey(stats, :var2)
    end
end
