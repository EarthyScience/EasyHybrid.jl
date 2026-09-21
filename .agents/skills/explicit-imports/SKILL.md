---
name: explicit-imports
description: >-
  Check, audit, and clean explicit imports across the EasyHybrid.jl codebase using ExplicitImports.jl.
  Use when adding new dependencies, modifying module imports in src/EasyHybrid.jl, or checking for unused/missing explicit imports.
---

# Explicit Imports Workflow

EasyHybrid.jl enforces clean and explicit imports to prevent namespace collisions and maintain clarity across the codebase.

## Execution Steps

### Run Explicit Imports Audit
```bash
julia --project=tools/explicits tools/explicits/explicits.jl
```

Or via Julia one-liner:
```julia
using Pkg
Pkg.activate("tools/explicits")
Pkg.instantiate()
using EasyHybrid, ExplicitImports
print_explicit_imports(EasyHybrid)
```

## Guidelines
- Avoid importing unused symbols.
- When new functions/types are introduced from external packages, add explicit `using Package: symbol1, symbol2` statements in `src/EasyHybrid.jl`.
- Re-run the audit after updating package dependencies or module inclusions.
