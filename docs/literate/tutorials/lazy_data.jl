# # Training on Lazy (Out-of-Memory) Data
#
# Large datasets do not have to fit into memory to train a hybrid model. When `train`
# receives a disk-backed array, such as a [`YAXArray`](https://github.com/JuliaDataCubes/YAXArrays.jl)
# opened from a Zarr or NetCDF store, EasyHybrid reads the data in chunk-aligned blocks
# while training, instead of loading everything up front.
#
# This tutorial repacks the NetCDF file from [Getting Started](../get_started.md) into a
# Zarr store with the layout that suits training, opens it lazily and trains the same
# Q10 respiration model on it.
#
# The code for this tutorial can be found in [docs/literate/tutorials](https://github.com/EarthyScience/EasyHybrid.jl/tree/main/docs/literate/tutorials/) => lazy_data.jl.
#
#md # [![Jupyter notebook](https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter)](https://github.com/EarthyScience/EasyHybrid.jl/blob/gh-pages/dev/notebooks/tutorials/lazy_data.ipynb)
#md #
#md # ::: info Small data, real mechanics
#md #
#md # The example file has ~158 000 half-hourly time steps and would fit in memory easily.
#md # It is used here so the page builds quickly; the same code works unchanged for stores
#md # that are far larger than memory, on a local disk or in the cloud.
#md #
#md # :::
#
# ## 1. Load Packages

using EasyHybrid
using YAXArrays, Zarr
using DimensionalData: Dim
using DiskArrays: eachchunk
using CairoMakie

# ## 2. Repack the NetCDF File into Zarr
#
# Load the synthetic dataset used in Getting Started:

df = load_timeseries_netcdf("https://github.com/bask0/q10hybrid/raw/master/data/Synthetic4BookChap.nc")
first(df, 5)

# For training, store all variables in **one** 2-D array with
#
# - a `variable` dimension first, holding the variable names, and
# - the observations (here: time steps) last, named `batch_size`,
#
# which is the same `(variable × observation)` layout EasyHybrid uses internally. One read
# then returns all variables of a block of observations. Zarr cannot store `Symbol`
# labels, so the variable names are `String`s; EasyHybrid matches them to the model's
# `Symbol` names.

variables = ["ta", "sw_pot", "dsw_pot", "reco"]
values_matrix = permutedims(Float32.(Matrix(df[!, variables])))
cube = YAXArray(
    (Dim{:variable}(variables), Dim{:batch_size}(1:size(values_matrix, 2))),
    values_matrix,
)

# Chunk along the observations only. Each chunk holds every variable for 4096 time steps,
# so EasyHybrid reads exactly one chunk per block.

cube = setchunks(cube, (length(variables), 4096))
zarr_path = joinpath(mktempdir(), "q10_synthetic.zarr")
savecube(cube, zarr_path; driver = :zarr)

# ## 3. Open the Store Lazily
#
# `open_dataset` only reads the metadata; the values stay on disk until they are indexed.

lazy_cube = Cube(open_dataset(zarr_path))

# EasyHybrid detects that this array is disk-backed:

is_lazy_data(lazy_cube)

# The store is split into these chunks:

eachchunk(lazy_cube)

# Opening it costs a few kilobytes, compared with the full dataset in memory:

(lazy = Base.format_bytes(Base.summarysize(lazy_cube)), in_memory = Base.format_bytes(sizeof(values_matrix)))

# ## 4. Define the Hybrid Model
#
# The model is identical to the one in Getting Started.

function RbQ10(; ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end

parameters = (
    rb = (3.0f0, 0.0f0, 13.0f0),  # Basal respiration [μmol/m²/s]
    Q10 = (2.0f0, 1.0f0, 4.0f0),  # Temperature sensitivity [-]
)

hybrid_model = constructHybridModel(
    [:sw_pot, :dsw_pot],      # Predictors
    [:ta],                    # Forcing
    [:reco],                  # Target
    RbQ10,
    parameters,
    [:rb],                    # Neural network predicted parameters
    [:Q10],                   # Global parameters
    hidden_layers = [16, 16],
    activation = swish,
    scale_nn_outputs = true,
    input_batchnorm = true,
)

# ## 5. Train on the Lazy Cube
#
# Pass the lazy array to `train` exactly like a `DataFrame`. Behind the scenes:
#
# - one streaming pass, block by block, finds the usable observations (complete inputs,
#   an observed target), and the train/validation split selects among their indices;
# - every epoch, blocks are visited in random order, several of them are mixed in a
#   shuffle buffer (`lazy_shuffle_buffer` observations) and minibatches are drawn from it;
# - the next group of blocks is read on another thread while the current one is used
#   (start Julia with more than one thread, e.g. `julia -t 4`);
# - losses are evaluated block by block, so only the targets and predictions are kept.

out = train(
    hybrid_model,
    lazy_cube;
    nepochs = 20,
    batchsize = 512,
    opt = RMSProp(0.001),
    monitor_names = [:rb, :Q10],
    patience = 10,
    lazy_shuffle_buffer = 32768, # observations mixed together for minibatches
    show_progress = false,
    model_name = "RbQ10_lazy",
)

# ## 6. Check Results
#
# Training and validation losses:

EasyHybrid.plot_loss(out, yscale = identity)

# The learned global parameter:

out.train_diffs.Q10

# Observed versus predicted respiration on the validation period:

EasyHybrid.poplot(out)

# ## 7. Options and Practical Notes
#
# The lazy path is configured through `train` keywords (fields of `DataConfig`):
#
# | Keyword | Default | Meaning |
# |:--|:--|:--|
# | `lazy` | `nothing` | Auto-detect disk-backed arrays; `true`/`false` forces the lazy or in-memory path. |
# | `lazy_block_size` | one chunk | Observations read per block, rounded up to whole chunks. |
# | `lazy_shuffle_buffer` | `65536` | Observations from random blocks shuffled together for minibatches. |
# | `lazy_prefetch` | `true` | Read the next blocks on another thread (needs `julia -t N`, `N > 1`). |
#
# A few things to keep in mind:
#
# - **Chunking decides the read cost.** Chunk along the observations and keep all
#   variables in a chunk. A few thousand observations per chunk work well; a single
#   chunk holding the whole time series would force a full read.
# - **Shuffling.** Neighbouring time steps are strongly autocorrelated, so minibatches
#   drawn from a single chunk make training unstable. The shuffle buffer mixes several
#   random chunks; raise `lazy_shuffle_buffer` if training is noisier than with
#   in-memory data, lower it if memory is tight.
# - **Missing values.** Observations with a missing predictor or forcing, or with no
#   observed target, are dropped block by block, as for `DataFrame` inputs.
# - **Remote stores.** Any `YAXArray` works, for example a Zarr store opened over
#   HTTP or from S3 with `open_dataset`.
# - **Not supported yet:** sequence models (`sequence_length`) and the full-batch
#   `Optimization.jl` path (`full_batch = true`) need the data in memory; use
#   `lazy = false` for them.
#
# Lazy data can also be used directly, for example to inspect a minibatch:

lazy_data = LazyHybridData(lazy_cube, hybrid_model)
((x, forcings), (y, mask)) = EasyHybrid.MLUtils.getobs(lazy_data, 1:5)
x
