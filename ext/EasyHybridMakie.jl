module EasyHybridMakie

using EasyHybrid
using Makie
using Makie.Colors
using DataFrames
import Makie
import EasyHybrid

include("HybridTheme.jl")

@debug "Extension loaded!"

Makie.convert_single_argument(wt::WrappedTuples) = Matrix(wt)

function Makie.series(wt::WrappedTuples; axislegend = (;) , attributes...)
    data_matrix, merged_attributes = _series(wt, attributes)
    p = Makie.series(data_matrix; merged_attributes...)
    Makie.axislegend(p.axis; merge=true, axislegend...)
    return p
end

function _series(wt::WrappedTuples, attributes)
    data_matrix = Matrix(wt)'
    plot_attributes = Makie.Attributes(;
        labels = string.(keys(wt))
        )
    user_attributes = Makie.Attributes(; attributes...)
    merged_attributes = merge(user_attributes, plot_attributes)
    return data_matrix, merged_attributes
end

# =============================================================================
# Prediction vs Observed Plotting Functions
# =============================================================================

"""
    plot_pred_vs_obs(ax, pred, obs, title_prefix)

Create a scatter plot comparing predicted vs observed values with performance metrics.

# Arguments
- `ax`: Makie axis to plot on
- `pred`: Vector of predicted values
- `obs`: Vector of observed values  
- `title_prefix`: Title prefix for the plot

# Returns
- Updates the axis with the plot and adds modeling efficiency to title
"""
function EasyHybrid.poplot(pred, obs, title_prefix)

    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])

    EasyHybrid.plot_pred_vs_obs!(ax, pred, obs, title_prefix)

    return fig

end

"""
    plot_pred_vs_obs!(fig, pred, obs, title_prefix, row::Int, col::Int)

Add a prediction vs observed plot to a figure at the specified position.

# Arguments
- `fig`: Makie figure to add plot to
- `pred`: Vector of predicted values
- `obs`: Vector of observed values
- `title_prefix`: Title prefix for the plot
- `row`: Row position in figure grid
- `col`: Column position in figure grid

# Returns
- Updated figure with the new plot
"""
function EasyHybrid.poplot!(fig, pred, obs, title_prefix, row::Int, col::Int)
    ax = Makie.Axis(fig[row, col])
    EasyHybrid.plot_pred_vs_obs!(ax, pred, obs, title_prefix)
end

function EasyHybrid.plot_pred_vs_obs!(ax, pred, obs, title_prefix)
    ss_res = sum((obs .- pred).^2)
    ss_tot = sum((obs .- mean(obs)).^2)
    modeling_efficiency = 1 - ss_res / ss_tot

    ax.title = "$title_prefix\nModeling Efficiency: $(round(modeling_efficiency, digits=3))"
    ax.xlabel = "Predicted"
    ax.ylabel = "Observed"
    ax.aspect = 1

    Makie.scatter!(ax, pred, obs, color=:purple, alpha=0.6, markersize=8)

    max_val = max(maximum(obs), maximum(pred))
    min_val = min(minimum(obs), minimum(pred))
    Makie.lines!(ax, [min_val, max_val], [min_val, max_val], color=:black, linestyle=:dash, linewidth=1, label="1:1 line")

    Makie.axislegend(ax; position=:lt)
end

# =============================================================================
# Generic Dispatch Methods for TrainResults
# =============================================================================

"""
    poplot!(results::TrainResults; target_cols=nothing, show_training=true, show_validation=true)

Create prediction vs observation plots from TrainResults object.

# Arguments
- `results`: TrainResults object from training
- `target_cols`: Specific target columns to plot (if nothing, plots all available targets)
- `show_training`: Whether to show training data plots (default: true)
- `show_validation`: Whether to show validation data plots (default: true)

# Returns
- Figure with prediction vs observation plots
"""
function EasyHybrid.poplot!(results::EasyHybrid.TrainResults; target_cols=nothing, show_training=true, show_validation=true)
    # Get available target columns from the data
    train_df = results.train_obs_pred
    val_df = results.val_obs_pred
    
    # Extract target columns (those without "_hat" suffix)
    all_cols = names(train_df)
    obs_cols = filter(col -> !endswith(col, "_pred"), all_cols)
    
    # Use specified target columns or all available
    targets_to_plot = isnothing(target_cols) ? obs_cols : target_cols
    
    # Count total plots needed
    n_plots = length(targets_to_plot) * (show_training + show_validation)

    # Create figure layout
    if (show_training && show_validation) && n_plots < 6
        n_cols = 2
    else
        n_cols = min(4, n_plots)  # Max 4 columns
    end
    n_rows = ceil(Int, n_plots / n_cols)
    
    fig = Makie.Figure(size=(300 * n_cols, 300 * n_rows))
    
    plot_idx = 1
    
    for target in targets_to_plot
        pred_col = target * "_pred"
        
        if show_training && target in names(train_df) && pred_col in names(train_df)
            row = ceil(Int, plot_idx / n_cols)
            col = ((plot_idx - 1) % n_cols) + 1
            
            # Filter out NaN values
            mask = .!isnan.(train_df[!, target]) .& .!isnan.(train_df[!, pred_col])
            obs_vals = train_df[mask, target]
            pred_vals = train_df[mask, pred_col]
            
            if length(obs_vals) > 0
                EasyHybrid.poplot!(fig, obs_vals, pred_vals, "Training: $target", row, col)
                plot_idx += 1
            end
        end
        
        if show_validation && target in names(val_df) && pred_col in names(val_df)
            row = ceil(Int, plot_idx / n_cols)
            col = ((plot_idx - 1) % n_cols) + 1
            
            # Filter out NaN values
            mask = .!isnan.(val_df[!, target]) .& .!isnan.(val_df[!, pred_col])
            obs_vals = val_df[mask, target]
            pred_vals = val_df[mask, pred_col]
            
            if length(obs_vals) > 0
                EasyHybrid.poplot!(fig, pred_vals, obs_vals, "Validation: $target", row, col)
                plot_idx += 1
            end
        end
    end
    
    return fig
end

"""
    poplot(results::TrainResults; kwargs...)

Convenience function that creates and returns a figure with prediction vs observation plots.
"""
function EasyHybrid.poplot(results::EasyHybrid.TrainResults; kwargs...)
    return EasyHybrid.poplot!(results; kwargs...)
end

# =============================================================================
# Additional Convenience Methods
# =============================================================================

"""
    poplot!(results::TrainResults, target::String; data_type=:both)

Create prediction vs observation plot for a specific target variable.

# Arguments
- `results`: TrainResults object from training
- `target`: Name of the target variable to plot
- `data_type`: Which data to plot (:training, :validation, or :both)

# Returns
- Figure with the specific target plot(s)
"""
function EasyHybrid.poplot!(results::EasyHybrid.TrainResults, target::String; data_type=:both)
    show_training = data_type in [:training, :both]
    show_validation = data_type in [:validation, :both]
    
    return EasyHybrid.poplot!(results; target_cols=[target], show_training=show_training, show_validation=show_validation)
end

"""
    poplot!(results::TrainResults, targets::Vector{String}; kwargs...)

Create prediction vs observation plots for specific target variables.

# Arguments
- `results`: TrainResults object from training
- `targets`: Vector of target variable names to plot
- Additional keyword arguments passed to main poplot! method

# Returns
- Figure with plots for the specified targets
"""
function EasyHybrid.poplot!(results::EasyHybrid.TrainResults, targets::Vector{String}; kwargs...)
    return EasyHybrid.poplot!(results; target_cols=targets, kwargs...)
end

"""
    poplot!(fig::Makie.Figure, results::TrainResults, row::Int, col::Int, target::String; data_type=:training)

Add a single prediction vs observation plot to an existing figure at specified position.

# Arguments
- `fig`: Existing Makie figure
- `results`: TrainResults object from training
- `row`: Row position in figure grid
- `col`: Column position in figure grid
- `target`: Name of the target variable to plot
- `data_type`: Which data to plot (:training or :validation)

# Returns
- Updated figure
"""
function EasyHybrid.poplot!(fig::Makie.Figure, results::EasyHybrid.TrainResults, row::Int, col::Int, target::String; data_type=:training)
    # Select appropriate dataframe
    df = data_type == :training ? results.train_obs_pred : results.val_obs_pred
    hat_col = target * "_hat"
    
    # Check if columns exist
    if !(target in names(df)) || !(hat_col in names(df))
        error("Target '$target' or predicted column '$hat_col' not found in $(data_type) data")
    end
    
    # Filter out NaN values
    mask = .!isnan.(df[!, target]) .& .!isnan.(df[!, hat_col])
    obs_vals = df[mask, target]
    pred_vals = df[mask, hat_col]
    
    if length(obs_vals) > 0
        title_prefix = data_type == :training ? "Training: $target" : "Validation: $target"
        EasyHybrid.poplot!(fig, pred_vals, obs_vals, title_prefix, row, col)
    else
        @warn "No valid data points found for target '$target' in $(data_type) data"
    end
    
    return fig
end

function EasyHybrid.plot_loss(loss, yscale)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; yscale=yscale, xlabel = "epoch", ylabel="loss")
    Makie.lines!(ax, loss; color = :grey25,label="Training Loss")
    on(loss) do _
        autolimits!(ax)
    end
    display(fig; title="EasyHybrid.jl", focus_on_show = true)
end

function EasyHybrid.plot_loss!(loss)
    if nameof(Makie.current_backend()) == :WGLMakie # TODO for our CPU cluster - alternatives?
        sleep(2.0) 
    end
    ax = Makie.current_axis()
    Makie.lines!(ax, loss; color = :tomato, label="Validation Loss")
    Makie.axislegend(ax; position=:rt)
end

function EasyHybrid.to_obs(o)
    Makie.Observable(o)
end

function EasyHybrid.to_point2f(i, p)
    Makie.Point2f(i, p)
end

function __init__()
    @debug "setting theme_easy_hybrid"
    # hybrid_latex = merge(theme_easy_hybrid(), theme_latexfonts())
    hybrid_latex = theme_easy_hybrid()
    set_theme!(hybrid_latex)
end

end