module EasyHybridMakie

using EasyHybrid
using Makie
using Makie.Colors
using DataFrames
import Makie
import EasyHybrid
import EasyHybrid: get_loss_value, get_monitor_values, collect_monitor_history
using Statistics
using DataStructures: CircularBuffer

include("HybridTheme.jl")

@debug "Extension loaded!"

Makie.convert_single_argument(wt::WrappedTuples) = Matrix(wt)

function Makie.series(wt::WrappedTuples; axislegend = (;), attributes...)
    data_matrix, merged_attributes = _series(wt, attributes)
    p = Makie.series(data_matrix; merged_attributes...)
    Makie.axislegend(p.axis; merge = true, axislegend...)
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

include("recipes/LossPlot.jl")
include("recipes/MonitorPlot.jl")
include("recipes/PredictionPlot.jl")
include("recipes/TimeSeriesPlot.jl")

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
function EasyHybrid.poplot(pred, obs, title_prefix; xlabel = "Predicted", ylabel = "Observed")

    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1])

    EasyHybrid.plot_pred_vs_obs!(ax, pred, obs, title_prefix; xlabel, ylabel)

    return fig

end

"""
    plot_pred_vs_obs!(fig, pred, obs, title_prefix, row::Int, col::Int; xlabel="Predicted", ylabel="Observed")

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
function EasyHybrid.poplot!(fig, pred, obs, title_prefix, row::Int, col::Int; xlabel = "Predicted", ylabel = "Observed")
    ax = Makie.Axis(fig[row, col])
    return EasyHybrid.plot_pred_vs_obs!(ax, pred, obs, title_prefix; xlabel, ylabel)
end

"""
    plot_pred_vs_obs!(ax, pred, obs, title_prefix; xlabel="Predicted", ylabel="Observed")

Add a scatter plot comparing predicted vs observed values with performance metrics on an existing axis.

# Arguments
- `ax`: Makie axis to plot on
- `pred`: Vector of predicted values
- `obs`: Vector of observed values
- `title_prefix`: Title prefix for the plot
- `xlabel`: Label for the x-axis (default: "Predicted")
- `ylabel`: Label for the y-axis (default: "Observed")

# Returns
- A `Legend` object containing the 1:1 line legend entry.
"""
function EasyHybrid.plot_pred_vs_obs!(ax, pred, obs, title_prefix; xlabel = "Predicted", ylabel = "Observed")
    ss_res = sum((obs .- pred) .^ 2)
    ss_tot = sum((obs .- mean(obs)) .^ 2)
    modeling_efficiency = 1 - ss_res / ss_tot

    ax.title = "$title_prefix\nModeling Efficiency: $(round(modeling_efficiency, digits = 3))"
    ax.xlabel = xlabel
    ax.ylabel = ylabel
    ax.aspect = 1

    Makie.scatter!(ax, pred, obs, color = :purple, alpha = 0.6, markersize = 8)

    max_val = max(maximum(obs), maximum(pred))
    min_val = min(minimum(obs), minimum(pred))
    Makie.lines!(ax, [min_val, max_val], [min_val, max_val], color = :black, linestyle = :dash, linewidth = 1, label = "1:1 line")

    return Makie.axislegend(ax; position = :lt)
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
function EasyHybrid.poplot!(results::EasyHybrid.TrainResults; target_cols = nothing, show_training = true, show_validation = true)
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

    fig = Makie.Figure(size = (300 * n_cols, 300 * n_rows))

    plot_idx = 1

    for target in targets_to_plot
        pred_col = target * "_pred"

        if show_training && target in names(train_df) && pred_col in names(train_df)
            row = ceil(Int, plot_idx / n_cols)
            col = ((plot_idx - 1) % n_cols) + 1

            # Filter out NaN values
            mask = .!isnan.(train_df[!, target]) .& .!isnan.(train_df[!, pred_col])
            obs = train_df[mask, target]
            pred = train_df[mask, pred_col]

            if length(obs) > 0
                EasyHybrid.poplot!(fig, pred, obs, "Training: $target", row, col)
                plot_idx += 1
            end
        end

        if show_validation && target in names(val_df) && pred_col in names(val_df)
            row = ceil(Int, plot_idx / n_cols)
            col = ((plot_idx - 1) % n_cols) + 1

            # Filter out NaN values
            mask = .!isnan.(val_df[!, target]) .& .!isnan.(val_df[!, pred_col])
            obs = val_df[mask, target]
            pred = val_df[mask, pred_col]

            if length(obs) > 0
                EasyHybrid.poplot!(fig, pred, obs, "Validation: $target", row, col)
                plot_idx += 1
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

"""
    plot_loss(loss, yscale)

Create an observable-based loss plot for live training updates.

# Arguments
- `loss`: Observable containing the training loss history
- `yscale`: Y-axis scale function (e.g. `log10`)

# Returns
- A Makie `Figure` object containing the loss plot
"""
function EasyHybrid.plot_loss(loss, yscale)
    fig = Makie.Figure()
    ax = Makie.Axis(fig[1, 1]; yscale = yscale, xlabel = "epoch", ylabel = "loss")
    Makie.lines!(ax, loss; color = :grey25, label = "Training Loss")
    on(loss) do _
        autolimits!(ax)
    end
    return display(fig; title = "EasyHybrid.jl", focus_on_show = true)
end

"""
    plot_loss!(loss)

Add a validation loss line to the current observable-based loss plot.

# Arguments
- `loss`: Observable containing the validation loss history

# Returns
- The axis legend added to the current plot
"""
function EasyHybrid.plot_loss!(loss)
    if nameof(Makie.current_backend()) == :WGLMakie # TODO for our CPU cluster - alternatives?
        sleep(2.0)
    end
    ax = Makie.current_axis()
    Makie.lines!(ax, loss; color = :tomato, label = "Validation Loss")
    return Makie.axislegend(ax; position = :rt)
end

"""
    log_tick_formatter(values)

Format logarithmic axis ticks as superscript powers of 10.

# Arguments
- `values`: Array of numeric values to format

# Returns
- Array of formatted string labels (e.g., "10²")
"""
function log_tick_formatter(values)
    return map(v -> "10" * Makie.UnicodeFun.to_superscript(round(Int64, v)), values)
end

"""
    _extract_monitor(monitor, name)

Extract monitor values for a specific parameter name.

# Arguments
- `monitor`: Dictionary or NamedTuple containing monitor values
- `name`: Symbol of the monitor parameter to extract

# Returns
- Tuple containing:
  - The extracted values (either a scalar or a quantile dictionary)
  - Boolean indicating if the values are quantiles
  - Array of quantile keys (empty if scalar)
"""
function _extract_monitor(monitor, name)
    entry = monitor[name]
    if haskey(entry, :quantile)
        q = entry[:quantile]
        return q, true, collect(keys(q))
    else
        return entry[:scalar], false, Symbol[]
    end
end

"""
    build_dashboards(history, cfg, y_train, y_val)

Initialize the training dashboards with static and live-updating components.

# Arguments
- `history`: `TrainingHistory` object containing metrics
- `cfg`: `TrainConfig` object containing plotting configuration
- `y_train`: Training targets
- `y_val`: Validation targets

# Returns
- Tuple containing:
  - `figures`: Dict mapping component name to Figure
  - `axes_dict`: Dict mapping component name to named tuple of axes
  - `plots_dict`: Dict mapping component name to named tuple of plots
"""
function EasyHybrid.build_dashboards(history, cfg, y_train, y_val)
    components = cfg.dashboard_components
    split = cfg.split_dashboard

    figures = Dict{Symbol, Any}()
    axes_dict = Dict{Symbol, Any}()
    plots_dict = Dict{Symbol, Any}()

    n_epochs = get_epochs(history)

    if split
        for comp in components
            fig = Makie.Figure(size = (800, 600))
            figures[comp] = fig
            _build_component!(comp, fig, fig[1, 1], history, cfg, y_train, y_val, n_epochs, axes_dict, plots_dict)
            display(fig)
        end
    else
        fig = Makie.Figure(size = (1200, 800))
        figures[:dashboard] = fig

        n_comp = length(components)
        rows = n_comp > 2 ? 2 : 1
        cols = ceil(Int, n_comp / rows)

        for (i, comp) in enumerate(components)
            r = ceil(Int, i / cols)
            c = ((i - 1) % cols) + 1
            _build_component!(comp, fig, fig[r, c], history, cfg, y_train, y_val, n_epochs, axes_dict, plots_dict)
        end
        display(fig)
    end

    return figures, axes_dict, plots_dict
end

function _build_component!(comp, fig, layout, history, cfg, y_train, y_val, n_epochs, axes_dict, plots_dict)
    return if comp == :loss
        vals_train = get_loss_value_t(history, cfg.training_loss, Symbol("$(cfg.agg)"))
        vals_val = get_loss_value_v(history, cfg.training_loss, Symbol("$(cfg.agg)"))

        ax, plt = lossplot(
            layout,
            n_epochs, vals_train, vals_val;
            axis = (;
                xlabel = "Epochs", ylabel = "Loss", yscale = log10,
                xtrimspine = (true, false), ytrimspine = true,
            )
        )
        Legend(layout[1, 1, Top()], ax, plt; orientation = :horizontal, halign = :left, framevisible = false)
        hidespines!(ax, :r, :t)
        z_rect = z_Rect2(n_epochs, vals_train, vals_val)
        plt_rect = lines!(ax, z_rect, color = :dodgerblue, linewidth = 1)

        ax_z = Axis(
            layout,
            width = Relative(0.35), height = Relative(0.35),
            halign = 0.95, valign = 1,
            xlabel = "", ylabel = "",
            rightspinecolor = :dodgerblue, leftspinecolor = :dodgerblue,
            topspinecolor = :dodgerblue, bottomspinecolor = :dodgerblue,
            title = "Zoomed View"
        )
        plt_z = lossplot!(ax_z, n_epochs, vals_train, vals_val)
        translate!(ax_z.blockscene, 0, 0, 150)

        axes_dict[:loss] = (; ax, ax_z)
        plots_dict[:loss] = (; plt, plt_rect, plt_z)

    elseif comp == :prediction
        y_pred_train = get_prediction_values(history, cfg.target_names[1], :train)
        y_pred_val = get_prediction_values(history, cfg.target_names[1], :validation)
        y_obs_train = getfield(y_train, cfg.target_names[1])
        y_obs_val = getfield(y_val, cfg.target_names[1])

        gd_pred = GridLayout(layout)
        ax_pred_train = Axis(
            gd_pred[1, 1]; xlabel = "", ylabel = "Observed", title = "Training",
            xtrimspine = true, ytrimspine = true, aspect = 1
        )
        hidespines!(ax_pred_train, :r, :t)
        plt_pred_train = predictionplot!(ax_pred_train, y_pred_train, y_obs_train)

        ax_pred_val = Axis(
            gd_pred[1, 2]; xlabel = "", ylabel = "", title = "Validation",
            xtrimspine = true, ytrimspine = true, aspect = 1
        )
        hidespines!(ax_pred_val, :l, :r, :t)
        plt_pred_val = predictionplot!(ax_pred_val, y_pred_val, y_obs_val; color = :tomato)
        hideydecorations!(ax_pred_val, grid = false, ticks = false)
        linkyaxes!(ax_pred_train, ax_pred_val)

        Box(gd_pred[1, 1:2, Top()]; color = (:grey25, 0.1), strokevisible = false)
        Label(gd_pred[1, 1:2, Top()], "$(cfg.target_names[1])")

        Box(gd_pred[1, 1:2, Bottom()]; color = (:grey45, 0.1), strokevisible = false)
        Label(gd_pred[1, 1:2, Bottom()], "Predicted")

        axes_dict[:prediction] = (; ax_pred_train, ax_pred_val)
        plots_dict[:prediction] = (; plt_pred_train, plt_pred_val)

    elseif comp == :timeseries
        y_pred_train = get_prediction_values(history, cfg.target_names[1], :train)
        y_pred_val = get_prediction_values(history, cfg.target_names[1], :validation)
        y_obs_train = getfield(y_train, cfg.target_names[1])
        y_obs_val = getfield(y_val, cfg.target_names[1])

        gd_ts = GridLayout(layout)
        ax_ts_train = Axis(
            gd_ts[1, 1]; xlabel = "Index", ylabel = "Value", title = "Training",
            xtrimspine = true, ytrimspine = true
        )
        hidespines!(ax_ts_train, :r, :t)
        plt_ts_train = timeseriesplot!(ax_ts_train, y_pred_train, y_obs_train)

        ax_ts_val = Axis(
            gd_ts[1, 2]; xlabel = "Index", ylabel = "", title = "Validation",
            xtrimspine = true, ytrimspine = true
        )
        hidespines!(ax_ts_val, :l, :r, :t)
        plt_ts_val = timeseriesplot!(ax_ts_val, y_pred_val, y_obs_val)
        hideydecorations!(ax_ts_val, grid = false, ticks = false)
        linkyaxes!(ax_ts_train, ax_ts_val)

        axes_dict[:timeseries] = (; ax_ts_train, ax_ts_val)
        plots_dict[:timeseries] = (; plt_ts_train, plt_ts_val)

    elseif comp == :monitor
        if !isempty(cfg.monitor_names)
            gl_m, axes_m, plts_m = setup_monitor_panel!(fig, layout, history, cfg)
            axes_dict[:monitor] = (; axes_m)
            plots_dict[:monitor] = (; plts_m)
        end
    end
end

"""
    z_Rect2(z_n_epochs, train_zoom, val_zoom)

Create a bounding rectangle for the zoomed-in view of the loss curve.

# Arguments
- `z_n_epochs`: Array of epoch indices for the zoomed window
- `train_zoom`: Array of training loss values in the zoomed window
- `val_zoom`: Array of validation loss values in the zoomed window

# Returns
- A `Rect2` representing the bounding box for the zoomed region
"""
function z_Rect2(z_n_epochs, train_zoom, val_zoom)
    mn_epoch = minimum(z_n_epochs)
    mx_epoch = maximum(z_n_epochs)
    xzoom_rect = mx_epoch - mn_epoch + 1
    mn_tv = minimum(map(minimum, [train_zoom, val_zoom]))
    mx_tv = maximum(map(maximum, [train_zoom, val_zoom]))
    z_rect = Rect2(mn_epoch - 0.5, 0.95 * mn_tv, xzoom_rect, 1.05 * (mx_tv - mn_tv))

    return z_rect
end

"""
    setup_monitor_panel!(fig, grid_position, history, cfg)

Initialize the monitor plot panel within a specific grid layout.

# Arguments
- `fig`: The parent Makie figure
- `grid_position`: Tuple representing the position in the GridLayout
- `history`: `TrainingHistory` object containing metrics
- `cfg`: `TrainConfig` object containing plotting configuration

# Returns
- Tuple containing:
  - `gl`: The created GridLayout for the panel
  - `axes`: Array of initialized axes
  - `plts`: Array of initialized plots
"""
function setup_monitor_panel!(fig, grid_position, history, cfg)
    monitor_names = cfg.monitor_names
    n = length(monitor_names)

    raw_train = get_monitor_values(history, monitor_names, :train)
    training_mon = collect_monitor_history(raw_train, monitor_names)
    raw_val = get_monitor_values(history, monitor_names, :validation)
    validation_mon = collect_monitor_history(raw_val, monitor_names)

    n_epochs = get_epochs(history)

    # Use a nested GridLayout at the given position
    gl = GridLayout(grid_position)

    axes = Vector{Axis}(undef, n)
    plts = Vector{MonitorPlot}(undef, n)

    for (i, name) in enumerate(monitor_names)
        y_train, is_q, qkeys = _extract_monitor(training_mon, name)
        y_val, _, _ = _extract_monitor(validation_mon, name)

        ax = Axis(
            gl[1, i];
            xlabel = "Epochs",
            ylabel = string(name),
            xtrimspine = (true, false),
            ytrimspine = true,
        )
        hidespines!(ax, :r, :t)

        plt = monitorplot!(
            ax, n_epochs, y_train, y_val;
            is_quantile = is_q,
            quantile_keys = qkeys,
        )
        Legend(
            gl[1, i, Top()], ax, plt;
            orientation = :horizontal,
            titleposition = :left,
            framevisible = false,
            nbanks = 2,
        )

        axes[i] = ax
        plts[i] = plt
    end

    return gl, axes, plts
end

"""
    update_monitor_panel!(axes, plts, history, cfg)

Update the monitor panel plots with new values from the training history.

# Arguments
- `axes`: Array of axes in the monitor panel
- `plts`: Array of plot objects in the monitor panel
- `history`: `TrainingHistory` object containing updated metrics
- `cfg`: `TrainConfig` object
"""
function update_monitor_panel!(axes, plts, history, cfg)
    monitor_names = cfg.monitor_names
    n_epochs = get_epochs(history)

    raw_train = get_monitor_values(history, monitor_names, :train)
    training_mon = collect_monitor_history(raw_train, monitor_names)
    raw_val = get_monitor_values(history, monitor_names, :validation)
    validation_mon = collect_monitor_history(raw_val, monitor_names)

    for (i, name) in enumerate(monitor_names)
        y_train, _, _ = _extract_monitor(training_mon, name)
        y_val, _, _ = _extract_monitor(validation_mon, name)
        update!(plts[i], n_epochs, y_train, y_val)
        autolimits!(axes[i])
    end
    return
end

"""
    update_step_dashboards!(dashboard, history, cfg)

Update all plots in the training dashboard with the latest epoch data.

# Arguments
- `dashboard`: The `TrainDashboard` object
- `history`: `TrainingHistory` object containing updated metrics
- `cfg`: `TrainConfig` object
"""
function EasyHybrid.update_step_dashboards!(dashboard, history, cfg)
    n_epochs = get_epochs(history)

    if haskey(dashboard.plots, :loss)
        zoom_epochs = 50
        vals_train = get_loss_value_t(history, cfg.training_loss, Symbol("$(cfg.agg)"))
        vals_val = get_loss_value_v(history, cfg.training_loss, Symbol("$(cfg.agg)"))

        update!(dashboard.plots[:loss].plt, n_epochs, vals_train, vals_val)
        autolimits!(dashboard.axes[:loss].ax)

        zoom_idx = max(1, length(vals_train) - zoom_epochs)
        train_zoom = vals_train[zoom_idx:end]
        val_zoom = vals_val[zoom_idx:end]
        z_n_epochs = n_epochs[zoom_idx:end]

        updatedRect2 = z_Rect2(z_n_epochs, train_zoom, val_zoom)
        update!(dashboard.plots[:loss].plt_rect, arg1 = updatedRect2)

        update!(dashboard.plots[:loss].plt_z, z_n_epochs, val_zoom, train_zoom)
        autolimits!(dashboard.axes[:loss].ax_z)
    end

    if haskey(dashboard.plots, :prediction)
        y_pred_train = get_prediction_values(history, cfg.target_names[1], :train)
        y_pred_val = get_prediction_values(history, cfg.target_names[1], :validation)
        update!(dashboard.plots[:prediction].plt_pred_train, y_pred_train)
        update!(dashboard.plots[:prediction].plt_pred_val, y_pred_val)
        autolimits!(dashboard.axes[:prediction].ax_pred_train)
        autolimits!(dashboard.axes[:prediction].ax_pred_val)
    end

    if haskey(dashboard.plots, :timeseries)
        y_pred_train = get_prediction_values(history, cfg.target_names[1], :train)
        y_pred_val = get_prediction_values(history, cfg.target_names[1], :validation)
        update!(dashboard.plots[:timeseries].plt_ts_train, y_pred_train)
        update!(dashboard.plots[:timeseries].plt_ts_val, y_pred_val)
        autolimits!(dashboard.axes[:timeseries].ax_ts_train)
        autolimits!(dashboard.axes[:timeseries].ax_ts_val)
    end

    if haskey(dashboard.plots, :monitor) && !isempty(cfg.monitor_names)
        update_monitor_panel!(dashboard.axes[:monitor].axes_m, dashboard.plots[:monitor].plts_m, history, cfg)
    end

    return nothing
end


"""
    dashboard_figure()

Get the current Makie figure for the dashboard.
"""
EasyHybrid.dashboard_figure() = Makie.current_figure()

"""
    record_history(args...; kargs...)

Record a video of the dashboard using `Makie.record`.
"""
EasyHybrid.record_history(args...; kargs...) = Makie.record(args...; backend = Makie.current_backend(), kargs...)

"""
    VideoStream(fig; kargs...)

Create a video stream using `Makie.VideoStream`.
"""
EasyHybrid.VideoStream(fig; kargs...) = Makie.VideoStream(fig; kargs...)

"""
    recordframe!(io)

Record a single frame to the video stream.
"""
EasyHybrid.recordframe!(io) = Makie.recordframe!(io)

"""
    save_fig(args...)

Save the current figure to a file.
"""
EasyHybrid.save_fig(args...) = Makie.save(args...)

"""
    save_video(path, io)

Save the recorded video stream to a file.
"""
EasyHybrid.save_video(path, io) = Makie.save(path, io)

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
function EasyHybrid.plot_loss(results::EasyHybrid.TrainResults; loss_type = :mse, yscale = log10, show_training = true, show_validation = true)
    fig = Makie.Figure(size = (600, 400))
    ax = Makie.Axis(fig[1, 1]; yscale = yscale, xlabel = "Epoch", ylabel = "Loss")

    epochs = 0:(length(results.train_history) - 1)

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
        Makie.lines!(ax, epochs, train_losses; color = :grey25, label = "Training Loss", linewidth = 2)
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
        Makie.lines!(ax, epochs, val_losses; color = :tomato, label = "Validation Loss", linewidth = 2)
    end

    Makie.axislegend(ax; position = :rt)
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
function EasyHybrid.plot_loss!(ax::Makie.Axis, results::EasyHybrid.TrainResults; loss_type = :mse, show_training = true, show_validation = true)
    epochs = 0:(length(results.train_history) - 1)

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
        Makie.lines!(ax, epochs, train_losses; color = :grey25, label = "Training Loss", linewidth = 2)
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
        Makie.lines!(ax, epochs, val_losses; color = :tomato, label = "Validation Loss", linewidth = 2)
    end

    Makie.axislegend(ax; position = :rt)

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
function EasyHybrid.plot_parameters(results::EasyHybrid.TrainResults; param_names = nothing, layout = :subplots)
    # Get available parameter names
    available_params = keys(results.epoch_history)
    params_to_plot = isnothing(param_names) ? collect(available_params) : param_names

    # Validate parameter names
    for param in params_to_plot
        if !(param in available_params)
            error("Parameter '$param' not found in parameter history. Available: $(available_params)")
        end
    end

    epochs = 0:(length(results.epoch_history) - 1)

    if layout == :subplots
        # Create subplot layout
        n_params = length(params_to_plot)
        n_cols = min(3, n_params)
        n_rows = ceil(Int, n_params / n_cols)

        fig = Makie.Figure(size = (300 * n_cols, 300 * n_rows))

        for (i, param) in enumerate(params_to_plot)
            row = ceil(Int, i / n_cols)
            col = ((i - 1) % n_cols) + 1

            ax = Makie.Axis(fig[row, col]; xlabel = "Epoch", ylabel = string(param))

            # Extract parameter values over epochs
            param_values = Float64[]
            for ps_record in results.epoch_history
                push!(param_values, getproperty(ps_record, param))
            end
            Makie.lines!(ax, epochs, param_values; color = :steelblue, linewidth = 2)

            ax.title = "Parameter: $(param)"
        end
    else  # overlay
        fig = Makie.Figure(size = (600, 400))
        ax = Makie.Axis(fig[1, 1]; xlabel = "Epoch", ylabel = "Parameter Value")

        colors = Makie.Cycled(1:length(params_to_plot))

        for param in params_to_plot
            param_values = Float64[]
            for ps_record in results.epoch_history
                push!(param_values, getproperty(ps_record, param))
            end
            Makie.lines!(ax, epochs, param_values; label = string(param), linewidth = 2, color = colors)
        end

        Makie.axislegend(ax; position = :rt)
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
function EasyHybrid.plot_parameters!(ax::Makie.Axis, results::EasyHybrid.TrainResults, param_name::Symbol; color = :steelblue)
    epochs = 0:(length(results.epoch_history) - 1)
    param_values = Float64[]
    for ps_record in results.epoch_history
        push!(param_values, getproperty(ps_record, param_name))
    end

    Makie.lines!(ax, epochs, param_values; color = color, linewidth = 2, label = string(param_name))

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
function EasyHybrid.plot_training_summary(results::EasyHybrid.TrainResults; loss_type = :mse, param_names = nothing, yscale = log10)
    # Get parameter info
    available_params = keys(results.epoch_history[1])
    params_to_plot = isnothing(param_names) ? collect(available_params) : param_names
    n_params = length(params_to_plot)

    fig = EasyHybrid.poplot(results)

    # Loss plot
    ax_loss = Makie.Axis(fig[2, 1:2]; yscale = yscale, xlabel = "Epoch", ylabel = "Loss")
    EasyHybrid.plot_loss!(ax_loss, results; loss_type = loss_type)
    ax_loss.title = "Training Summary - Loss Evolution"
    Makie.hidexdecorations!(ax_loss)

    # Parameter plots
    epochs = 0:(length(results.epoch_history) - 1)

    for (i, param) in enumerate(params_to_plot)
        row = i + 2
        col = 1:2

        ax = Makie.Axis(fig[row, col]; xlabel = "Epoch", ylabel = string(param))
        EasyHybrid.plot_parameters!(ax, results, param)
        ax.title = "Parameter: $(param)"

        Makie.linkxaxes!(ax_loss, ax)
    end

    return fig
end

"""
    to_obs(o)

Convert a value to a Makie Observable.
"""
function EasyHybrid.to_obs(o)
    return Makie.Observable(o)
end

"""
    to_point2f(i, p)

Create a Point2f from an index and a value.
"""
function EasyHybrid.to_point2f(i, p)
    return Makie.Point2f(i, p)
end

function __init__()
    @debug "setting theme_easy_hybrid"
    # hybrid_latex = merge(theme_easy_hybrid(), theme_latexfonts())
    hybrid_latex = theme_easy_hybrid()
    return set_theme!(hybrid_latex, GLMakie = (title = "EasyHybrid.jl", focus_on_show = true))
end

end
