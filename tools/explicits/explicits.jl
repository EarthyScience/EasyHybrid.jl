using Pkg
Pkg.activate(@__DIR__)
dir = joinpath(@__DIR__, "..", "..")

Pkg.develop(path = dir)
Pkg.instantiate()

using EasyHybrid # the package you want to analyze
using ExplicitImports
print_explicit_imports(EasyHybrid)
