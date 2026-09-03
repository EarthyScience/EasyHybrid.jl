---
name: run-tests
description: >-
  Run the test suite or targeted unit tests for EasyHybrid.jl.
  Use when validating bug fixes, testing new features, or running regression tests.
---

# Testing Workflow for EasyHybrid.jl

## 1. Running the Full Test Suite
To run all tests:
```bash
julia --project -e 'using Pkg; Pkg.test()'
```

## 2. Running Targeted Component Tests
To quickly run a specific test file during development:
```bash
# Example: Test GenericHybridModel
julia --project -e 'include("test/test_generic_hybrid_model.jl")'

# Example: Test loss computations
julia --project -e 'include("test/test_compute_loss.jl")'

# Example: Test data splitting
julia --project -e 'include("test/test_split_data_train.jl")'

# Example: Test transformers
julia --project -e 'include("test/test_transformers.jl")'
```

## 3. Test Best Practices
- Keep synthetic test datasets small so that tests run within seconds.
- Test forward pass outputs, parameter gradient calculations (loss backward pass), and edge cases.
