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

# =============================================================================
# Comprehensive Training Analysis
# =============================================================================

"""
    plot_training_overview(results::TrainResults; loss_type=:mse, param_names=nothing)

Create a comprehensive overview of training results including loss evolution, parameter evolution, and prediction vs observation plots.

# Arguments
- `results`: TrainResults object from training
- `loss_type`: Which loss type to plot for loss evolution (default: :mse)
- `param_names`: Specific parameter names to plot (if nothing, plots all available)

# Returns
- Figure with comprehensive training overview
"""
function EasyHybrid.plot_training_overview(results::EasyHybrid.TrainResults; loss_type=:mse, param_names=nothing)
    # Get available info
    available_params = keys(results.ps_history[1])
    params_to_plot = isnothing(param_names) ? collect(available_params) : param_names
    n_params = length(params_to_plot)

    # Layout: loss + params + poplots
    base_height = 200
    param_height = n_params > 0 ? 150 * ceil(Int, n_params / 2) : 0
    poplot_height = 350
    total_height = base_height + param_height + poplot_height

    fig = Makie.Figure(size=(900, total_height))

    current_row = 1

    # 1. Loss Evolution (top section)
    ax_loss = Makie.Axis(fig[current_row, 1:2]; yscale=log10, xlabel="Epoch", ylabel="Loss")
    EasyHybrid.plot_loss!(ax_loss, results; loss_type=loss_type)
    ax_loss.title = "Training Overview - Loss Evolution ($(uppercase(string(loss_type))))"
    current_row += 1

    # 2. Parameter Evolution
    if n_params > 0
        param_rows = ceil(Int, n_params / 2)
        for i in 1:param_rows
            for j in 1:2
                param_idx = (i-1) * 2 + j
                if param_idx <= n_params
                    param = params_to_plot[param_idx]
                    ax = Makie.Axis(fig[current_row + i - 1, j]; xlabel="Epoch", ylabel=string(param))
                    EasyHybrid.plot_parameters!(ax, results, param)
                    ax.title = "Parameter: $(param)"
                end
            end
        end
        current_row += param_rows
    end

    # 3. Prediction vs Observation Plots
    # Extract target columns from training data
    train_df = results.train_obs_pred
    all_cols = names(train_df)
    obs_cols = filter(col -> !endswith(col, "_pred"), all_cols)
    
    if !isempty(obs_cols)
        target_cols = min(2, length(obs_cols))  # Show max 2 targets
        
        for (i, target) in enumerate(obs_cols[1:target_cols])
            pred_col = target * "_pred"
            col_pos = i
            
            if target in names(train_df) && pred_col in names(train_df)
                ax = Makie.Axis(fig[current_row, col_pos]; xlabel="Predicted", ylabel="Observed", aspect=1)
                
                # Training data
                train_mask = .!isnan.(train_df[!, target]) .& .!isnan.(train_df[!, pred_col])
                if any(train_mask)
                    train_obs = train_df[train_mask, target]
                    train_pred = train_df[train_mask, pred_col]
                    Makie.scatter!(ax, train_pred, train_obs; color=:steelblue, alpha=0.6, markersize=6, label="Training")
                end
                
                # Validation data
                val_df = results.val_obs_pred
                if target in names(val_df) && pred_col in names(val_df)
                    val_mask = .!isnan.(val_df[!, target]) .& .!isnan.(val_df[!, pred_col])
                    if any(val_mask)
                        val_obs = val_df[val_mask, target]
                        val_pred = val_df[val_mask, pred_col]
                        Makie.scatter!(ax, val_pred, val_obs; color=:tomato, alpha=0.6, markersize=6, label="Validation")
                    end
                end
                
                # 1:1 line
                if any(train_mask) || (target in names(val_df) && pred_col in names(val_df) && any(.!isnan.(val_df[!, target]) .& .!isnan.(val_df[!, pred_col])))
                    all_obs = vcat(
                        any(train_mask) ? train_df[train_mask, target] : Float32[],
                        (target in names(val_df) && pred_col in names(val_df)) ? val_df[.!isnan.(val_df[!, target]) .& .!isnan.(val_df[!, pred_col]), target] : Float32[]
                    )
                    all_pred = vcat(
                        any(train_mask) ? train_df[train_mask, pred_col] : Float32[],
                        (target in names(val_df) && pred_col in names(val_df)) ? val_df[.!isnan.(val_df[!, target]) .& .!isnan.(val_df[!, pred_col]), pred_col] : Float32[]
                    )
                    
                    if !isempty(all_obs) && !isempty(all_pred)
                        min_val = min(minimum(all_obs), minimum(all_pred))
                        max_val = max(maximum(all_obs), maximum(all_pred))
                        Makie.lines!(ax, [min_val, max_val], [min_val, max_val]; color=:black, linestyle=:dash, linewidth=1, label="1:1 line")
                    end
                end
                
                ax.title = "Pred vs Obs: $(target)"
                if length(obs_cols) > 1
                    Makie.axislegend(ax; position=:lt)
                end
            end
        end
    end

    return fig
end
# =============================================================================
# Convenience Methods for Direct Plot Creation
# =============================================================================

"""
    poplot(results::TrainResults; kwargs...)

Convenience function that creates and returns a figure with prediction vs observation plots.
"""
function EasyHybrid.poplot(results::EasyHybrid.TrainResults; kwargs...)
    return EasyHybrid.poplot!(results; kwargs...)
end

# =============================================================================
# Original Observable-based Loss Plotting (for live training updates)
# =============================================================================

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

# =============================================================================
# Generic Dispatch Methods for Loss and Parameter Plotting
# =============================================================================

"""
    plot_loss(results::TrainResults; loss_type=:mse, yscale=log10, show_training=true, show_validation=true)

Plot training and validation loss history from TrainResults object.

# Arguments
- `results`: TrainResults object from training
- `loss_type`: Which loss type to plot (e.g., :mse, :nse, :mae)
- `yscale`: Y-axis scale function (default: log10)
- `show_training`: Whether to show training loss (default: true)
- `show_validation`: Whether to show validation loss (default: true)

# Returns
- Figure with loss plots
"""
function EasyHybrid.plot_loss(results::EasyHybrid.TrainResults; loss_type=:mse, yscale=log10, show_training=true, show_validation=true)
    fig = Makie.Figure(size=(600, 400))
    ax = Makie.Axis(fig[1, 1]; yscale=yscale, xlabel="Epoch", ylabel="Loss")
    
    epochs = 0:(length(results.train_history)-1)
    
    if show_training
        # Extract loss values for the specified loss type
        train_losses = Float64[]
        for loss_record in results.train_history
            # Extract loss value for the specified loss type
            loss_type_data = getproperty(loss_record, loss_type)
            if hasfield(typeof(loss_type_data), :sum)
                push!(train_losses, loss_type_data.sum)
            else
                # sum all values in the NamedTuple if no sum field
                push!(train_losses, sum(values(loss_type_data)))
            end
        end
        Makie.lines!(ax, epochs, train_losses; color=:grey25, label="Training Loss", linewidth=2)
    end
    
    if show_validation
        val_losses = Float64[]
        for loss_record in results.val_history
            # Extract loss value for the specified loss type
            loss_type_data = getproperty(loss_record, loss_type)
            if hasfield(typeof(loss_type_data), :sum)
                push!(val_losses, loss_type_data.sum)
            else
                # sum all values in the NamedTuple if no sum field
                push!(val_losses, sum(values(loss_type_data)))
            end
        end
        Makie.lines!(ax, epochs, val_losses; color=:tomato, label="Validation Loss", linewidth=2)
    end
    
    Makie.axislegend(ax; position=:rt)
    ax.title = "Loss Evolution - $(uppercase(string(loss_type)))"
    
    return fig
end

"""
    plot_loss!(ax::Axis, results::TrainResults; loss_type=:mse, show_training=true, show_validation=true)

Add loss plots to an existing axis.

# Arguments
- `ax`: Makie axis to plot on
- `results`: TrainResults object from training
- `loss_type`: Which loss type to plot
- `show_training`: Whether to show training loss
- `show_validation`: Whether to show validation loss

# Returns
- Updated axis
"""
function EasyHybrid.plot_loss!(ax::Makie.Axis, results::EasyHybrid.TrainResults; loss_type=:mse, show_training=true, show_validation=true)
    epochs = 0:(length(results.train_history)-1)
    
    if show_training
        train_losses = Float64[]
        for loss_record in results.train_history
            loss_type_data = getproperty(loss_record, loss_type)
            if hasfield(typeof(loss_type_data), :sum)
                push!(train_losses, loss_type_data.sum)
            else
                push!(train_losses, sum(values(loss_type_data)))
            end
        end
        Makie.lines!(ax, epochs, train_losses; color=:grey25, label="Training Loss", linewidth=2)
    end
    
    if show_validation
        val_losses = Float64[]
        for loss_record in results.val_history
            loss_type_data = getproperty(loss_record, loss_type)
            if hasfield(typeof(loss_type_data), :sum)
                push!(val_losses, loss_type_data.sum)
            else
                push!(val_losses, sum(values(loss_type_data)))
            end
        end
        Makie.lines!(ax, epochs, val_losses; color=:tomato, label="Validation Loss", linewidth=2)
    end
    
    return ax
end

"""
    plot_parameters(results::TrainResults; param_names=nothing, layout=:subplots)

Plot parameter evolution during training from TrainResults object.

# Arguments
- `results`: TrainResults object from training
- `param_names`: Specific parameter names to plot (if nothing, plots all available)
- `layout`: Layout style (:subplots for separate plots, :overlay for single plot)

# Returns
- Figure with parameter evolution plots
"""
function EasyHybrid.plot_parameters(results::EasyHybrid.TrainResults; param_names=nothing, layout=:subplots)
    # Get available parameter names
    available_params = keys(results.ps_history)
    params_to_plot = isnothing(param_names) ? collect(available_params) : param_names
    
    # Validate parameter names
    for param in params_to_plot
        if !(param in available_params)
            error("Parameter '$param' not found in parameter history. Available: $(available_params)")
        end
    end
    
    epochs = 0:(length(results.ps_history)-1)
    
    if layout == :subplots
        # Create subplot layout
        n_params = length(params_to_plot)
        n_cols = min(3, n_params)
        n_rows = ceil(Int, n_params / n_cols)
        
        fig = Makie.Figure(size=(300 * n_cols, 300 * n_rows))
        
        for (i, param) in enumerate(params_to_plot)
            row = ceil(Int, i / n_cols)
            col = ((i - 1) % n_cols) + 1
            
            ax = Makie.Axis(fig[row, col]; xlabel="Epoch", ylabel=string(param))
            
            # Extract parameter values over epochs
            param_values = Float64[]
            for ps_record in results.ps_history
                push!(param_values, getproperty(ps_record, param))
            end
            Makie.lines!(ax, epochs, param_values; color=:steelblue, linewidth=2)
            
            ax.title = "Parameter: $(param)"
        end
    else  # overlay
        fig = Makie.Figure(size=(600, 400))
        ax = Makie.Axis(fig[1, 1]; xlabel="Epoch", ylabel="Parameter Value")
        
        colors = Makie.Cycled(1:length(params_to_plot))
        
        for param in params_to_plot
            param_values = Float64[]
            for ps_record in results.ps_history
                push!(param_values, getproperty(ps_record, param))
            end
            Makie.lines!(ax, epochs, param_values; label=string(param), linewidth=2, color=colors)
        end
        
        Makie.axislegend(ax; position=:rt)
        ax.title = "Parameter Evolution"
    end
    
    return fig
end

"""
    plot_parameters!(ax::Axis, results::TrainResults, param_name::Symbol; color=:steelblue)

Add a single parameter evolution plot to an existing axis.

# Arguments
- `ax`: Makie axis to plot on
- `results`: TrainResults object from training
- `param_name`: Name of the parameter to plot
- `color`: Line color for the parameter plot

# Returns
- Updated axis
"""
function EasyHybrid.plot_parameters!(ax::Makie.Axis, results::EasyHybrid.TrainResults, param_name::Symbol; color=:steelblue)
    epochs = 0:(length(results.ps_history)-1)
    param_values = Float64[]
    for ps_record in results.ps_history
        push!(param_values, getproperty(ps_record, param_name))
    end
    
    Makie.lines!(ax, epochs, param_values; color=color, linewidth=2, label=string(param_name))
    
    return ax
end

"""
    plot_training_summary(results::TrainResults; loss_type=:mse, param_names=nothing)

Create a comprehensive summary plot showing loss evolution and parameter evolution.

# Arguments
- `results`: TrainResults object from training
- `loss_type`: Which loss type to plot for loss evolution
- `param_names`: Specific parameter names to plot (if nothing, plots all available)

# Returns
- Figure with both loss and parameter plots
"""
function EasyHybrid.plot_training_summary(results::EasyHybrid.TrainResults; loss_type=:mse, param_names=nothing)
    # Get parameter info
    available_params = keys(results.ps_history[1])
    params_to_plot = isnothing(param_names) ? collect(available_params) : param_names
    n_params = length(params_to_plot)
    
    # Create layout: loss plot on top, parameters below
    fig = Makie.Figure(size=(800, 200 + 150 * ceil(Int, n_params / 2)))
    
    # Loss plot
    ax_loss = Makie.Axis(fig[1, 1:2]; yscale=log10, xlabel="Epoch", ylabel="Loss")
    EasyHybrid.plot_loss!(ax_loss, results; loss_type=loss_type)
    ax_loss.title = "Training Summary - Loss Evolution"
    
    # Parameter plots
    epochs = 0:(length(results.ps_history)-1)
    
    for (i, param) in enumerate(params_to_plot)
        row = 2 + ceil(Int, (i-1) / 2)
        col = ((i - 1) % 2) + 1
        
        ax = Makie.Axis(fig[row, col]; xlabel="Epoch", ylabel=string(param))
        EasyHybrid.plot_parameters!(ax, results, param)
        ax.title = "Parameter: $(param)"
    end
    
    return fig
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