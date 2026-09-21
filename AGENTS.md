# EasyHybrid.jl — Agent Guidelines & Project Rules

Welcome to **EasyHybrid.jl**, a Julia package for hybrid modeling that seamlessly combines scientific/mechanistic models with neural networks (built with [Lux.jl](https://github.com/LuxDL/Lux.jl), [Optimization.jl](https://github.com/SciML/Optimization.jl), and [ComponentArrays.jl](https://github.com/jonniedie/ComponentArrays.jl)).

---

## 1. Project Overview & Architecture

### High-Level Design
A hybrid model in `EasyHybrid.jl` integrates:
- Neural network component: $h(x; \theta)$ parametrized by $\theta$
- Mechanistic / domain-specific component: $M(\cdot, z; \phi)$ driven by forcings $z$ and parameters $\phi$
- Training framework powered by `Optimization.jl`, automatic differentiation (`Zygote.jl` / `ForwardDiff.jl`), and flexible loss definitions.

### Source Code Map (`src/`)
- `src/EasyHybrid.jl`: Main package module, top-level re-exports, and component `include` order.
- `src/models/`:
  - `GenericHybridModel.jl`: Core struct and dispatch for hybrid models.
  - `NNModels.jl`: Pre-packaged neural network architectures (MLP, RNN, LSTM, etc.).
  - `transformers/`: Sequence and transformer-specific model layers.
  - `show_generic.jl`: Pretty-printing and terminal display for models.
- `src/training/`:
  - `train.jl` & `train_optimization.jl`: Training loops and `Optimization.jl` integration.
  - `dashboard.jl`, `history.jl`, `early_stopping.jl`: Training progress metrics and stopping criteria.
  - `initialization.jl`: Parameter initialization strategies.
  - `tune.jl`: Hyperparameter optimization with `Hyperopt.jl`.
- `src/losses/`:
  - `compute_loss_types.jl` & `compute_loss.jl`: Core loss dispatch and evaluation.
  - `loss_fn.jl`: Custom hybrid loss constructors.
- `src/data/`:
  - `prepare_data.jl`, `sequences.jl`, `seq2seq.jl`: Data formatting, time-series windowing, batching.
  - `splits.jl`, `split_data.jl`: Train/validation/test dataset splitting.
  - `loaders.jl`: Data loader utilities.
- `src/config/`: Configuration structs (`TrainingConfig`, `DataConfig`, `TrainingPaths`) and YAML serialization.
- `src/io/`: Checkpoint management, model serialization (`JLD2.jl`), path resolvers.
- `src/utils/`: Macro helpers (`@hybrid`), weight extraction, cross-validation helpers.
- `ext/`: Package extensions (e.g., `EasyHybridMakie.jl` for Makie.jl plotting recipes and themes).
- `test/`: Test suites executed by `runtests.jl`.
- `tools/`:
  - `tools/formatter/`: Code formatting via [Runic.jl](https://github.com/fredrikekre/Runic.jl).
  - `tools/explicits/`: Explicit import validation via [ExplicitImports.jl](https://github.com/ericphanson/ExplicitImports.jl).

---

## 2. Code Style & Formatting Guidelines

- **Formatting (Runic)**: The project strictly adheres to [Runic.jl](https://github.com/fredrikekre/Runic.jl) style.
  To format code, execute:
  ```julia
  using Pkg; Pkg.activate("tools/formatter"); Pkg.instantiate()
  using Runic; Runic.main(["--verbose", "--inplace", "."])
  ```
- **Explicit Imports**: Maintain explicit imports where possible (`using Package: symbol1, symbol2`). Avoid unconditional namespace pollution. Check imports with `tools/explicits/explicits.jl`.
- **ColPrac Guidelines**: Follow the [ColPrac](https://github.com/SciML/ColPrac) guidelines for collaborative practices in scientific Julia projects.
- **Documentation**: Provide clear Julia docstrings for all exported types and functions with signatures, description, arguments, and examples.
- **Preserve Existing Docs & Comments**: Never remove existing comments or docstrings unless explicitly requested or replacing obsolete docs.

---

## 3. Performance & Typing Rules

- **Type Stability**: Ensure functions are type-stable. Avoid non-constant global variables and untyped containers.
- **ComponentArrays & Lux.jl**: When working with neural network parameters, respect Lux functional idioms (explicit parameter `ps` and state `st` separation) and `ComponentArray` structure.
- **Dimensional Data**: Follow conventions when working with `DimArray`, `KeyedArray`, or `DataFrame` inputs and maintain dimension metadata where applicable.

---

## 4. Testing & Verification Protocols

- **Running Tests**: Run tests via the Julia package manager:
  ```julia
  using Pkg; Pkg.test()
  ```
  Or run specific test files in `test/`:
  ```julia
  julia --project -e 'include("test/test_generic_hybrid_model.jl")'
  ```
- **Regression Testing**: Every bug fix or new feature must include a corresponding unit test in `test/` (e.g. `test/test_<feature>.jl` included in `test/runtests.jl`).
- **Minimal Working Examples**: Verify changes locally with minimal synthetic datasets or existing test fixtures before finalizing.

---

## 5. Rule & Skill Maintenance

- **Adding Functionality**: Whenever new modules, files, exported types, or workflows are added, update `AGENTS.md`, `.agents/rules/`, and `.agents/skills/easyhybrid-dev/SKILL.md` navigation maps and documentation accordingly.
- **Deleting / Refactoring Functionality**: When removing or refactoring components, remove stale references, update tests, and synchronize rules and skills so the agent context remains accurate.

