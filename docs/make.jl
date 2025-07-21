using EasyHybrid
using Documenter, DocumenterVitepress

makedocs(;
    modules=[EasyHybrid],
    authors="Lazaro Alonso, Markus Reichstein, Bernhard Ahrens",
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
            "Generic Hybrid" => "generic_hybrid.md",
        ],
        "Research" =>[
            "RbQ10" => "research/RbQ10_results.md"
            "BulkDensitySOC" => "research/BulkDensitySOC_results.md"
        ],
        "API" => "api.md",
    ],
    warnonly = true,
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/EarthyScience/EasyHybrid.jl", # this must be the full URL!
    devbranch = "main",
    push_preview = true,
)