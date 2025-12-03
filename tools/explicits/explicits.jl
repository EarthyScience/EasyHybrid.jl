using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using EasyHybrid # the package you want to analyze
using ExplicitImports
print_explicit_imports(EasyHybrid)
