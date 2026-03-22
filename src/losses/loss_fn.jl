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
    return r * r
end

function loss_fn(ŷ, y, y_nan, ::Val{:pearsonLoss})
    return one(eltype(ŷ)) .- (cor(ŷ[y_nan], y[y_nan]))
end

function loss_fn(ŷ, y, y_nan, ::Val{:nseLoss})
    return sum((ŷ[y_nan] .- y[y_nan]) .^ 2) / sum((y[y_nan] .- mean(y[y_nan])) .^ 2)
end

# one minus nse
function loss_fn(ŷ, y, y_nan, ::Val{:nse})
    return one(eltype(ŷ)) - (sum((ŷ[y_nan] .- y[y_nan]) .^ 2) / sum((y[y_nan] .- mean(y[y_nan])) .^ 2))
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

# Kling–Gupta Efficiency loss (to MINIMIZE)
function loss_fn(ŷ, y, y_nan, ::Val{:kgeLoss})
    ŷv = ŷ[y_nan]
    yv = y[y_nan]

    μ_s = mean(ŷv)
    μ_o = mean(yv)

    σ_s = std(ŷv)
    σ_o = std(yv)

    r = cor(ŷv, yv)

    α = σ_s / σ_o
    β = μ_s / μ_o

    # KGE_loss = sqrt((r - 1)^2 + (α - 1)^2 + (β - 1)^2)
    return sqrt(
        (r - one(eltype(ŷ)))^2 +
            (α - one(eltype(ŷ)))^2 +
            (β - one(eltype(ŷ)))^2
    )
end

function loss_fn(ŷ, y, y_nan, ::Val{:β})
    ŷv = ŷ[y_nan]
    yv = y[y_nan]

    μ_s = mean(ŷv)
    μ_o = mean(yv)

    β = μ_s / μ_o

    return β
end

function loss_fn(ŷ, y, y_nan, ::Val{:α})
    ŷv = ŷ[y_nan]
    yv = y[y_nan]

    σ_s = std(ŷv)
    σ_o = std(yv)
    α = σ_s / σ_o

    return α
end

# Kling–Gupta Efficiency metric (to MAXIMIZE, e.g. for reporting)
function loss_fn(ŷ, y, y_nan, ::Val{:kge})
    kge_loss = loss_fn(ŷ, y, y_nan, Val(:kgeLoss))
    return one(eltype(ŷ)) - kge_loss
end

# Kling–Gupta Efficiency loss (to MINIMIZE)
function loss_fn(ŷ, y, y_nan, ::Val{:pbkgeLoss})
    ŷv = ŷ[y_nan]
    yv = y[y_nan]

    μ_s = mean(ŷv)
    μ_o = mean(yv)

    r = cor(ŷv, yv)

    β = μ_s / μ_o

    return sqrt(
        (r - one(eltype(ŷ)))^2 +
            (β - one(eltype(ŷ)))^2
    )
end

function loss_fn(ŷ, y, y_nan, ::Val{:pbkge})
    pbkge_loss = loss_fn(ŷ, y, y_nan, Val(:pbkgeLoss))
    return one(eltype(ŷ)) - pbkge_loss
end

abstract type BestDirection end
struct Minimize <: BestDirection end
struct Maximize <: BestDirection end

# default: everything is minimized
bestdirection(::Any) = Minimize()

bestdirection(::Union{Val{:pearson}, Val{:r2}, Val{:nse}, Val{:kge}}) = Maximize()

isbetter(new, best, loss_type) = isbetter(new, best, bestdirection(Val(loss_type)))

# trait-dispatched implementations
isbetter(new, best, ::Minimize) = new < best
isbetter(new, best, ::Maximize) = new > best

function check_training_loss(loss_type)
    if bestdirection(Val(loss_type)) isa Maximize
        error(
            "Got a metric that is defined as `to be maximized` as a training loss: $(loss_type).\n" *
                "For training you must use a true loss (to be minimized), e.g. " *
                ":nseLoss (1-NSE), :kgeLoss (1-KGE), :pearsonLoss (1-Pearson), or :mse."
        )
    end
    return nothing
end
