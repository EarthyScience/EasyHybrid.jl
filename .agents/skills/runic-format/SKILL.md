---
name: runic-format
description: >-
  Format the Julia codebase using Runic.jl according to project style guidelines.
  Use when asked to format code, or after making code modifications before submitting changes.
---

# Runic Formatting Workflow

EasyHybrid.jl uses [Runic.jl](https://github.com/fredrikekre/Runic.jl) for code formatting.

## Execution Steps

To format the entire project or specific files in place:

### Method 1: Using the formatter environment script
```bash
julia --project=tools/formatter tools/formatter/format.jl
```

### Method 2: In Julia REPL or one-liner
```julia
using Pkg
Pkg.activate("tools/formatter")
Pkg.instantiate()
using Runic
Runic.main(["--verbose", "--inplace", "."])
```

## Verification
- Check `git diff` / `git status` to verify that only expected formatting changes occurred.
- Ensure all comments and docstrings remain intact.
