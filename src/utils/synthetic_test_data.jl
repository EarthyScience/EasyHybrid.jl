export gen_linear_data, gen_linear_data_2outputs
## Create some data y = a ⋅ x1 + b, where a = f(x2,x3), b=2
# init dummy data
function gen_linear_data(; seed = 123)
    Random.seed!(seed)
    df = DataFrame(rand(Float32, 1000, 3), :auto)
    # more variables
    df = @chain df begin
        @transform :a_syn = exp(-5.0f0((:x2 - 0.7f0))^2.0f0) + :x3 / 10.0f0
        @aside b = 2.0f0
        @transform :obs = :a_syn * :x1 + b + 0.1f0 * rand(Float32)
        @transform :pred_syn = :obs
        @transform :seqID = @bycol repeat(1:100, inner = 10)
    end
    return to_keyedArray(df)
end

function gen_linear_data_2outputs(; seed = 123)
    Random.seed!(seed)
    df = @chain DataFrame(rand(Float32, 1000, 3), :auto) begin
        @transform :seqID = @bycol repeat(1:100, inner = 10)
        groupby(:seqID)
        @transform :a_dyn_syn = @bycol cumsum(:x2 .- :x3)
        @transform :obs_dyn1 = :a_dyn_syn .* :x1 + 2.0f0
        @transform :obs_dyn2 = 0.5f0 .* :a_dyn_syn .* :x2
    end
    return to_keyedArray(df)
end
