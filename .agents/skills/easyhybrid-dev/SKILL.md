---
name: easyhybrid-dev
description: >-
  Comprehensive guide and workflow for developing, debugging, and extending EasyHybrid.jl.
  Use when implementing new features, fixing bugs, refactoring modules, or navigating the codebase.
---

# EasyHybrid.jl Development & Debugging Workflow

This skill guides you through developing and debugging components in `EasyHybrid.jl`.

---

## 1. Project Navigation Map

- **`src/models/`**:
  - `GenericHybridModel.jl`: Core `GenericHybridModel` type and dispatch.
  - `NNModels.jl`: Prebuilt neural network chains (MLP, RNN, LSTM).
  - `transformers/`: Sequence and transformer blocks.
  - `show_generic.jl`: Display methods for hybrid models.
- **`src/training/`**:
  - `train.jl` / `train_optimization.jl`: Training routines using `Optimization.jl`.
  - `dashboard.jl`, `history.jl`, `early_stopping.jl`: Training tracking and metrics.
  - `tune.jl`: Hyperparameter tuning integration.
- **`src/losses/`**:
  - `compute_loss.jl`, `compute_loss_types.jl`: Evaluation of standard and hybrid losses.
  - `loss_fn.jl`: Custom loss constructors and wrappers.
- **`src/data/`**:
  - `prepare_data.jl`, `sequences.jl`, `seq2seq.jl`: Sequence windowing, batch creation.
  - `split_data.jl`, `splits.jl`, `loaders.jl`: Train/val/test splitting and DataLoader helpers.
- **`src/config/`**:
  - `TrainingConfig.jl`, `DataConfig.jl`, `TrainingPaths.jl`, `config_yaml.jl`.
- **`src/io/`**:
  - `save.jl`, `paths.jl`, `checkpoints.jl`: Model serialization and checkpointing.
- **`src/utils/`**:
  - `macro_hybrid.jl`: `@hybrid` macro.
  - `extract_weights.jl`, `wrap_tuples.jl`, `helpers_cross_validation.jl`.
- **`ext/`**:
  - `EasyHybridMakie.jl`, `HybridTheme.jl`, `recipes/`: Makie plotting recipes.
- **`test/`**:
  - `runtests.jl` and individual test files for each component.

---

## 2. Standard Development Workflow

When implementing changes or fixing issues:

1. **Understand & Locate**: Identify the affected module in `src/` and relevant tests in `test/`.
2. **Implement**:
   - Keep functions type-stable and pure for AD compatibility.
   - Respect Lux parameter (`ps`) / state (`st`) functional idioms.
   - Use `ComponentArrays` when handling composite parameters.
3. **Format Code**:
   Run the Runic formatter:
   ```julia
   using Pkg; Pkg.activate("tools/formatter"); Pkg.instantiate()
   using Runic; Runic.main(["--verbose", "--inplace", "."])
   ```
4. **Check Explicit Imports**:
   ```julia
   using Pkg; Pkg.activate("tools/explicits"); Pkg.instantiate()
   using EasyHybrid, ExplicitImports
   print_explicit_imports(EasyHybrid)
   ```
5. **Run & Add Tests**:
   - Run relevant test suite in `test/` or run all tests via `Pkg.test()`.
   - Add unit tests for every bug fix or new feature.
