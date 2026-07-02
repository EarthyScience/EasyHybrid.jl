# Modular Dashboard

`EasyHybrid.jl` features a modular, "Lego-like" training dashboard system. This allows for fine-grained control over which plots are rendered during training, how they are arranged, and which components are saved as animations.

## Configuration Options

You can customize the visualization of your training loop using `TrainConfig` (or by passing these directly as keyword arguments to `train()`). 

The three primary options are:
- `dashboard_components`: Controls which plots are rendered. You can pass a subset of `[:loss, :prediction, :timeseries, :monitor]`.
- `split_dashboard`: A boolean controlling whether the dashboard components are drawn on a single window grid layout (`false`) or spawned as independent windows (`true`).
- `save_animations`: Controls which dashboard windows to record as `.mp4` video streams when `save_training = true`. For example, `[:loss]` or `[:all]`.

## Usage Examples

### Example 1: Minimal Dashboard
If you are only interested in tracking the training loss and seeing the real-time predictions, you can isolate those components:

```julia
train(model, df, ...;
    dashboard_components = [:loss, :prediction],
    split_dashboard = false # (Default) Renders side-by-side in one window
)
```

Alternatively, you can construct a `TrainConfig` object explicitly:

```julia
train_cfg = TrainConfig(
    dashboard_components = [:loss, :prediction],
    split_dashboard = false
)

train(model, df, ...; train_cfg)
```

### Example 2: Split Dashboard with Specific Animations
If you have a multi-monitor setup and prefer each graph to be in its own large detached window, you can enable `split_dashboard`. You can also choose to only record animations for specific plots (like the loss curve) to save rendering time and storage:

```julia
train(model, df, ...;
    dashboard_components = [:loss, :prediction, :timeseries, :monitor],
    split_dashboard = true, # Each component spawns in its own standalone window
    save_training = true,
    save_animations = [:loss] # Only the loss figure is saved as an mp4
)
```

### Example 3: Save All Split Animations
To create separate videos for every single plotted metric:

```julia
train(model, df, ...;
    dashboard_components = [:loss, :prediction, :timeseries, :monitor],
    split_dashboard = true,
    save_training = true,
    save_animations = [:all] # Records multiple .mp4 files, one for each component
)
```

## Default Behavior
If no new configuration fields are provided, EasyHybrid behaves with its original defaults: rendering all requested plot types into a single interactive window and recording a single `.mp4` animation if `save_training` is enabled.
