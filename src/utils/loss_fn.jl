# add as many loss functions as needed
export loss_fn

"""
    loss_fn(ŷ, y, y_nan, loss_type)

Compute the loss for given predictions and targets using various loss specifications.

# Arguments
- `ŷ`: Predicted values
- `y`: Target values
- `y_nan`: Mask for NaN values
- `loss_type`: One of the following:
    - `Val(:rmse)`: Root Mean Square Error
    - `Val(:mse)`: Mean Square Error 
    - `Val(:mae)`: Mean Absolute Error
    - `Val(:pearson)`: Pearson correlation coefficient
    - `Val(:r2)`: R-squared
    - `Val(:pearsonLoss)`: 1 - Pearson correlation coefficient
    - `Val(:nseLoss)`: 1 - NSE
    - `::Function`: Custom loss function with signature `f(ŷ, y)`
    - `::Tuple{Function, Tuple}`: Custom loss with args `f(ŷ, y, args...)`
    - `::Tuple{Function, NamedTuple}`: Custom loss with kwargs `f(ŷ, y; kwargs...)`
    - `::Tuple{Function, Tuple, NamedTuple}`: Custom loss with both `f(ŷ, y, args...; kwargs...)`

# Examples
```julia
# Predefined loss
loss = loss_fn(ŷ, y, y_nan, Val(:mse))

# Custom loss function
custom_loss(ŷ, y) = mean(abs2, ŷ .- y)
loss = loss_fn(ŷ, y, y_nan, custom_loss)

# With positional arguments
weighted_loss(ŷ, y, w) = w * mean(abs2, ŷ .- y)
loss = loss_fn(ŷ, y, y_nan, (weighted_loss, (0.5,)))

# With keyword arguments
scaled_loss(ŷ, y; scale=1.0) = scale * mean(abs2, ŷ .- y)
loss = loss_fn(ŷ, y, y_nan, (scaled_loss, (scale=2.0,)))

# With both args and kwargs
complex_loss(ŷ, y, w; scale=1.0) = scale * w * mean(abs2, ŷ .- y)
loss = loss_fn(ŷ, y, y_nan, (complex_loss, (0.5,), (scale=2.0,)))
```

You can define additional predefined loss functions by adding more methods:
```julia
import EasyHybrid: loss_fn
function EasyHybrid.loss_fn(ŷ, y, y_nan, ::Val{:nse})
    return 1 - sum((ŷ[y_nan] .- y[y_nan]).^2) / sum((y[y_nan] .- mean(y[y_nan])).^2)
end
```
"""
function loss_fn end

function loss_fn(ŷ, y, y_nan, ::Val{:rmse})
    return sqrt(mean(abs2, (ŷ[y_nan] .- y[y_nan])))
end
function loss_fn(ŷ, y, y_nan, ::Val{:mse})
    return mean(abs2, (ŷ[y_nan] .- y[y_nan]))
end
function loss_fn(ŷ, y, y_nan, ::Val{:mae})
    return mean(abs, (ŷ[y_nan] .- y[y_nan]))
end
# pearson correlation coefficient
function loss_fn(ŷ, y, y_nan, ::Val{:pearson})
    return cor(ŷ[y_nan], y[y_nan])
end
function loss_fn(ŷ, y, y_nan, ::Val{:r2})
    r = cor(ŷ[y_nan], y[y_nan])
    return r*r
end

function loss_fn(ŷ, y, y_nan, ::Val{:pearsonLoss})
    return one(eltype(ŷ)) .- (cor(ŷ[y_nan], y[y_nan]))
end

function loss_fn(ŷ, y, y_nan, ::Val{:nseLoss})
    return sum((ŷ[y_nan] .- y[y_nan]).^2) / sum((y[y_nan] .- mean(y[y_nan])).^2)
end

# one minus nse
function loss_fn(ŷ, y, y_nan, ::Val{:nse})
    return one(eltype(ŷ)) - (sum((ŷ[y_nan] .- y[y_nan]).^2) / sum((y[y_nan] .- mean(y[y_nan])).^2))
end

function loss_fn(ŷ, y, y_nan, training_loss::Function)
    return training_loss(ŷ[y_nan], y[y_nan])
end
function loss_fn(ŷ, y, y_nan, training_loss::Tuple{Function, Tuple})
    f, args = training_loss
    return f(ŷ[y_nan], y[y_nan], args...)
end

function loss_fn(ŷ, y, y_nan, training_loss::Tuple{Function, NamedTuple})
    f, kwargs = training_loss
    return f(ŷ[y_nan], y[y_nan]; kwargs...)
end
function loss_fn(ŷ, y, y_nan, training_loss::Tuple{Function, Tuple, NamedTuple})
    f, args, kwargs = training_loss
    return f(ŷ[y_nan], y[y_nan], args...; kwargs...)
end