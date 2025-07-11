# add as many loss functions as needed
export loss_fn

"""
    loss_fn(ŷ, y, y_nan, ::Val{:rmse})
    loss_fn(ŷ, y, y_nan, ::Val{:mse})
    loss_fn(ŷ, y, y_nan, ::Val{:mae})
    loss_fn(ŷ, y, y_nan, ::Val{:pearson})
    loss_fn(ŷ, y, y_nan, ::Val{:r2})

Compute the loss for the given predictions and targets using the specified loss type.

# Arguments:
- `ŷ`: Predicted values.
- `y`: Target values.
- `y_nan`: Mask for NaN values.
- `::Val{:rmse}`: Root Mean Square Error or `::Val{:mse}`: Mean Square Error or `::Val{:mae}`: Mean Absolute Error or `::Val{:pearson}`: Pearson correlation coefficient or `::Val{:r2}`: R-squared.

You can define additional loss functions as needed by adding more methods to this function.
# Example:
In your working script just do the following:
```julia
import EasyHybrid: loss_fn
function EasyHybrid.loss_fn(ŷ, y, y_nan, ::Val{:nse})
    return 1 - sum((ŷ[y_nan] .- y[y_nan]).^2) / sum((y[y_nan] .- mean(y[y_nan])).^2)
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
# person correlation coefficient
function loss_fn(ŷ, y, y_nan, ::Val{:pearson})
    return cor(ŷ[y_nan], y[y_nan])
end
function loss_fn(ŷ, y, y_nan, ::Val{:r2})
    r = cor(ŷ[y_nan], y[y_nan])
    return r*r
end

# one minus nse
function loss_fn(ŷ, y, y_nan, ::Val{:nse})
    return sum((ŷ[y_nan] .- y[y_nan]).^2) / sum((y[y_nan] .- mean(y[y_nan])).^2)
end