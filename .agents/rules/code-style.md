# Code Style & Guidelines for EasyHybrid.jl

## Julia & ColPrac Standards
- Follow the [ColPrac](https://github.com/SciML/ColPrac) guide for collaborative practices in scientific Julia projects.
- Write clean, type-stable Julia code. Avoid untyped containers, type instability, and non-constant global variables.
- Avoid type piracy (do not define methods on types you don't own for functions you don't own, unless explicitly extending standard interfaces like `axiskeys` for `DimArray`).

## Formatting with Runic
- Code must always adhere to the [Runic.jl](https://github.com/fredrikekre/Runic.jl) formatting style.
- Use the project's formatting script at `tools/formatter/format.jl` to format changed files.

## Explicit Imports
- Keep imports explicit (`using Module: func1, Type2`) where possible to prevent namespace collisions and retain clarity.
- Check imported names using `tools/explicits/explicits.jl`.
- Do not remove existing module exports without explicit deprecation or user consent.

## Documentation & Comments
- Maintain docstrings for all exported functions, structs, and macros following standard Julia docstring conventions (signature, description, arguments, examples).
- Always preserve existing comments and docstrings when making changes.
