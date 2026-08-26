# Testing & Verification Rules

## Running Tests
- Test files reside in `test/` and are included in `test/runtests.jl`.
- Test execution should be verified using:
  ```julia
  using Pkg; Pkg.test()
  ```
  or by running targeted test files directly:
  ```bash
  julia --project -e 'include("test/test_<feature>.jl")'
  ```

## Regression & Feature Testing
- Every bug fix or new feature must be accompanied by unit tests.
- When creating tests for hybrid models, use minimal synthetic test datasets (`SyntheticTestData` or small matrices) to ensure tests execute quickly and reliably.
- Verify both CPU forward/backward passes and type correctness in test suites.

## Verification Before Finalizing
- Never conclude a bug fix or feature implementation without executing relevant tests and confirming zero errors.
- Ensure all tests in `test/runtests.jl` pass cleanly before finishing.
