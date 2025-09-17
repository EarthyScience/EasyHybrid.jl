using EasyHybrid
using Documenter, DocumenterVitepress

makedocs(;
    modules=[EasyHybrid],
    authors="Lazaro Alonso, Bernhard Ahrens, Markus Reichstein",
    repo="https://github.com/EarthyScience/EasyHybrid.jl",
    sitename="EasyHybrid.jl",
    format=DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/EarthyScience/EasyHybrid.jl",
        devurl = "dev",
    ),
    pages=[
        "Home" => "index.md",
        "Get Started" => "get_started.md",
        "Tutorial" => [
            "Exponential Response" => "tutorials/exponential_res.md",
            "Hyperparameter Tuning" => "tutorials/hyperparameter_tuning.md",
            "Slurm" => "tutorials/slurm.md"
        ],
        "Research" =>[
            "Overview" => "research/overview.md"
            "RbQ10" => "research/RbQ10_results.md"
            "BulkDensitySOC" => "research/BulkDensitySOC_results.md"
        ],
        "API" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/EarthyScience/EasyHybrid.jl", # this must be the full URL!
    target=joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)