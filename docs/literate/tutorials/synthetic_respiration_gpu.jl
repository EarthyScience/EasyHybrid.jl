# [![CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
#
# # EasyHybrid Example: Synthetic Data Analysis
#
# This example demonstrates how to use EasyHybrid to train a hybrid model
# on synthetic data for respiration modeling with Q10 temperature sensitivity.
#

# ## For development (local docs run)
# include("../../setup_local_docsrun.jl")

using EasyHybrid
using MLDataDevices

# Load whichever GPU backend is available on this machine (CUDA / AMDGPU /
# Metal / oneAPI), or fall back to CPU. Defines `GPU_BACKEND_PKG` and
# `GPU_DEVICE_TYPE` in the caller's scope.
include("../../setup_gpu_backend.jl")

cpu_device() isa CPUDevice
gpu_device() isa GPU_DEVICE_TYPE
nothing #hide

# ## Data Loading and Preprocessing
#
# Load synthetic dataset from GitHub into a `DataFrame`.

df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc")

# Select a subset of data for faster execution (especially when benchmarking).
df = df[1:20000, :]

# ## Define the Physical Model
#
# **RbQ10 model**: Respiration model with Q10 temperature sensitivity
#
# Parameters:
#   - ta: air temperature [°C]
#   - Q10: temperature sensitivity factor [-]
#   - rb: basal respiration rate [μmol/m²/s]
#   - tref: reference temperature [°C] (default: 15.0)

function RbQ10(; ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end

# ### Define Model Parameters
#
# Parameter specification: (default, lower_bound, upper_bound)

# Parameter name | Default | Lower | Upper
parameters = (
    rb = (3.0f0, 0.0f0, 13.0f0),
    Q10 = (2.0f0, 1.0f0, 4.0f0),
)

# ## Configure Hybrid Model Components
#
# Define input variables (forcing = temperature).
forcing = [:ta]

# Target variable (respiration).
target = [:reco]

# Parameter classification: global parameters are shared across all samples,
# neural-network-predicted parameters are produced by the NN per sample.
global_param_names = [:Q10]
neural_param_names = [:rb]

# ## Single NN Hybrid Model Training
#
# Predictor variables for the NN: solar radiation potential and its derivative.
predictors_single_nn = [:sw_pot, :dsw_pot]

# `constructHybridModel` arguments: predictors, forcing, targets, the
# process-based model function, parameter definitions, NN-predicted parameters
# and global parameters. Keyword arguments configure the neural network
# architecture, activation, output scaling and input batch normalization.
small_nn_hybrid_model = constructHybridModel(
    predictors_single_nn,
    forcing,
    target,
    RbQ10,
    parameters,
    neural_param_names,
    global_param_names,
    hidden_layers = [16, 16],
    activation = sigmoid,
    scale_nn_outputs = true,
    input_batchnorm = true,
)

large_nn_hybrid_model = constructHybridModel(
    predictors_single_nn,
    forcing,
    target,
    RbQ10,
    parameters,
    neural_param_names,
    global_param_names,
    hidden_layers = [512, 512, 512],
    activation = sigmoid,
    scale_nn_outputs = true,
    input_batchnorm = true,
)

# ### Train on DataFrame
#
# Configure training. Set `keep_history = true` to keep per-epoch history
# (losses, predictions, etc.), and `save_training = true` to persist training
# history and checkpoints to disk.

cfg = EasyHybrid.TrainConfig(
    nepochs = 10,
    batchsize = 512,
    opt = Adam(0.001),
    loss_types = [:mse, :nse],
    show_progress = false,
    keep_history = false,
    save_training = false,
    plotting = false,
)

using Suppressor
using Printf

function _pure(s)
    return s.time - s.gctime - s.compile_time - s.recompile_time
end

function _warn_overhead(tag, label, s, warn_frac)
    gc_frac = s.gctime / s.time
    compile_frac = s.compile_time / s.time
    recompile_frac = s.recompile_time / s.time
    overhead_frac = gc_frac + compile_frac + recompile_frac
    return if overhead_frac > warn_frac
        pct(x) = round(100 * x; digits = 1)
        @warn """[$label/$tag] $(pct(overhead_frac))% of elapsed time was overhead:
        gc=$(pct(gc_frac))% ($(round(s.gctime; digits = 3))s), \
        compile=$(pct(compile_frac))% ($(round(s.compile_time; digits = 3))s), \
        recompile=$(pct(recompile_frac))% ($(round(s.recompile_time; digits = 3))s).
        Elapsed ratio will be misleading; prefer the pure ratio."""
    end
end

# Benchmark `tune` on CPU vs GPU. Uses `@timed` so we can separate out
# compile/recompile/GC overhead from steady-state work, and reports BOTH:
#   - elapsed ratio (wall clock CPU / wall clock GPU)
#   - pure ratio    (elapsed minus GC/compile/recompile, CPU/GPU)
#
# `warn_frac` is the fraction of total time spent in GC / (re)compile above
# which a warning is emitted for that run; defaults to 5%.
# A short 1-epoch run on each device triggers Julia/Zygote specialization
# and GPU kernel compilation, so the timed runs below measure steady-state
# training rather than first-call compile latency.
#
# Return a slim summary. We deliberately drop `.value` (the full TrainResults)
# and `.gcstats` so that REPL/Literate printing of the return value stays compact.
function bench_cpu_vs_gpu(model, df, cfg; label::AbstractString, warmup::Bool = true, warn_frac::Real = 0.05, tune_kwargs...)

    if warmup
        warm_cfg = EasyHybrid.TrainConfig(;
            nepochs = 1,
            batchsize = cfg.batchsize,
            opt = cfg.opt,
            show_progress = false,
            save_training = false,
            keep_history = false,
            plotting = false,
        )
        @suppress tune(model, df, warm_cfg; gdev = cpu_device(), tune_kwargs...)
        @suppress tune(model, df, warm_cfg; gdev = gpu_device(), tune_kwargs...)
    end

    cpu_stats = @suppress @timed tune(model, df, cfg; gdev = cpu_device(), tune_kwargs...)
    gpu_stats = @suppress @timed tune(model, df, cfg; gdev = gpu_device(), tune_kwargs...)

    _warn_overhead("CPU", label, cpu_stats, warn_frac)
    _warn_overhead("GPU", label, gpu_stats, warn_frac)

    cpu_pure = _pure(cpu_stats)
    gpu_pure = _pure(gpu_stats)
    elapsed_ratio = cpu_stats.time / gpu_stats.time
    pure_ratio = cpu_pure / gpu_pure

    @printf(
        "[%s] CPU: total=%.3fs compile=%.3fs recompile=%.3fs gc=%.3fs pure=%.3fs\n",
        label, cpu_stats.time, cpu_stats.compile_time, cpu_stats.recompile_time, cpu_stats.gctime, cpu_pure
    )
    @printf(
        "[%s] GPU: total=%.3fs compile=%.3fs recompile=%.3fs gc=%.3fs pure=%.3fs\n",
        label, gpu_stats.time, gpu_stats.compile_time, gpu_stats.recompile_time, gpu_stats.gctime, gpu_pure
    )
    @printf(
        "[%s] with GPU we get: elapsed=%.2fx  pure=%.2fx\n",
        label, elapsed_ratio, pure_ratio
    )

    slim(s) = (; s.time, s.bytes, s.gctime, s.compile_time, s.recompile_time)
    return (; cpu = slim(cpu_stats), gpu = slim(gpu_stats), elapsed_ratio, pure_ratio)
end

# On our `gpu1-hpc22` machine these typically come out around 0.57x for the
# small NN and 2.73x for the large NN.

bench_cpu_vs_gpu(small_nn_hybrid_model, df, cfg; label = "small NN")
bench_cpu_vs_gpu(large_nn_hybrid_model, df, cfg; label = "large NN")

# ## Sweep: GPU speedup vs hidden-layer width
#
# Build a hybrid model for a range of hidden-layer widths (fixed depth of 3),
# benchmark each on CPU and GPU, and plot the elapsed and pure ratios.
# A ratio > 1 means GPU is faster than CPU; a ratio < 1 means GPU is slower.

widths = [16, 64, 256, 512, 1024]
depths = 2:5

results = [
    begin
            r = bench_cpu_vs_gpu(
                small_nn_hybrid_model, df, cfg;
                label = "d=$d w=$w", hidden_layers = fill(w, d),
            )
            (;
                depth = d,
                width = w,
                elapsed_ratio = r.elapsed_ratio,
                pure_ratio = r.pure_ratio,
                cpu_time = r.cpu.time,
                gpu_time = r.gpu.time,
            )
        end
        for d in depths for w in widths
]

using CairoMakie

fig = Figure(size = (820, 520))
ax = Axis(
    fig[1, 1];
    xlabel = "Hidden-layer width",
    ylabel = "GPU speedup (CPU / GPU)",
    title = "GPU speedup vs hidden-layer width, per depth",
    xscale = log2,
    xticks = (widths, string.(widths)),
)

# Break-even line (ratio = 1).
hlines!(ax, [1.0]; color = :gray, linestyle = :dash)

# One colour per depth; `pure` is solid, `elapsed` is dashed.
# Only the solid (pure) line gets a label → legend entries are per-depth.
palette = Makie.wong_colors()
for (i, d) in enumerate(depths)
    rs = filter(r -> r.depth == d, results)
    ws = [r.width for r in rs]
    el = [r.elapsed_ratio for r in rs]
    pr = [r.pure_ratio    for r in rs]
    c = palette[mod1(i, length(palette))]
    scatterlines!(ax, ws, pr; color = c, linestyle = :solid, marker = :circle, label = "depth=$d")
    scatterlines!(ax, ws, el; color = c, linestyle = :dash, marker = :xcross)
end

# Legend 1: depth → colour (auto-picked up from the labeled solid lines).
axislegend(ax, "depth"; position = :lt)

# Legend 2: line style → metric (built manually).
style_elements = [
    LineElement(color = :black, linestyle = :solid),
    LineElement(color = :black, linestyle = :dash),
]
Legend(fig[1, 2], style_elements, ["pure", "elapsed"], "metric")

fig


batchsizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
bs_hidden = [64, 64, 64]

bs_results = map(batchsizes) do b
    r = bench_cpu_vs_gpu(
        small_nn_hybrid_model, df, cfg;
        label = "b=$b", batchsize = b, hidden_layers = bs_hidden
    )
    (;
        batchsize = b, elapsed_ratio = r.elapsed_ratio, pure_ratio = r.pure_ratio,
        cpu_time = r.cpu.time, gpu_time = r.gpu.time,
    )
end

fig_bs = Figure(size = (720, 480))
ax_bs = Axis(
    fig_bs[1, 1];
    xlabel = "Batch size (hidden = $bs_hidden)",
    ylabel = "GPU speedup (CPU / GPU)",
    title = "GPU speedup vs batch size",
    xscale = log2,
    xticks = (batchsizes, string.(batchsizes)),
)

bs = [r.batchsize     for r in bs_results]
bs_elapsed = [r.elapsed_ratio for r in bs_results]
bs_pure = [r.pure_ratio    for r in bs_results]

# Break-even line (ratio = 1).
hlines!(ax_bs, [1.0]; color = :gray, linestyle = :dash)
scatterlines!(ax_bs, bs, bs_pure; color = :steelblue, linestyle = :solid, marker = :circle, label = "pure")
scatterlines!(ax_bs, bs, bs_elapsed; color = :steelblue, linestyle = :dash, marker = :xcross, label = "elapsed")

axislegend(ax_bs; position = :lt)
fig_bs
