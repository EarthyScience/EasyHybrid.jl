# Data Pipeline Rules

## Split objects
- The training loops (`_train`, `_train_optimization`) never touch raw `x / forcings / y / mask` arrays directly. They receive two split objects from `prepare_training_splits` (`src/data/training_splits.jl`):
  - `InMemorySplit`: arrays already in memory (DataFrame, KeyedArray, in-memory DimArray, pre-split tuples).
  - `LazyHybridData`: disk-backed `(variable × observation)` arrays (e.g. `YAXArray` from Zarr), read block by block (`src/data/lazy_data.jl`).
- Anything the loops need from data goes through the split interface: `build_loader`, `evaluate_split`, `predict_split`, `observations`, `full_batch_data`. When adding a new training feature, implement it for **both** split types (or throw a clear `ArgumentError` for the lazy one).

## Lazy data
- Never `collect`/`Array` a whole lazy array. Read only chunk-aligned blocks (`_blocks`, `_read_filtered`) and keep inputs out of memory; only targets and predictions may be concatenated (`_concat_obs`).
- Training minibatches must mix several random blocks (`lazy_shuffle_buffer`): single time-contiguous chunks are autocorrelated and make training unstable.
- Variable names may be `String` (Zarr) or `Symbol`; match them via `string(name)`.
- Sequence models and `full_batch = true` are not supported for lazy data yet. In auto mode (`lazy = nothing`) unsupported setups (not 2-D with `:variable`, or `sequence_length`) fall back to in-memory; `full_batch` errors before any evaluation pass.
- Split proportions apply to *usable* observations (`lazy_splits` does one streaming `_keep_columns` pass first), matching the in-memory path.
- Never keep a full set of predictions per epoch for lazy splits: `compact_history!` reduces older snapshots to the `monitor_names` outputs.
- Don't call `first(::LazyBatchLoader)` or abandon its iteration: it leaves a prefetch task reading the store. Use `_placeholder_batch` for a stand-in batch.
- Tests use `DiskArrays.TestTypes.AccessCountDiskArray` to assert that no read spans more than a storage chunk (`test/test_lazy_data.jl`).
