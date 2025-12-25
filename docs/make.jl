# docs/make.jl
using EasyHybrid
using Documenter, DocumenterVitepress
literate_root = joinpath(@__DIR__, "literate")

# collect all .jl files recursively under docs/literate
jl_files = isdir(literate_root) ?
    [joinpath(root, f) for (root, _, files) in walkdir(literate_root) for f in files if endswith(f, ".jl")] :
    String[]

if !isempty(jl_files)
    @info "Running Literate.jl on $(length(jl_files)) files..."
    using Literate
    src_root = joinpath(@__DIR__, "src")

    function render_tree(indir::String, outdir::String)
        isdir(indir) || return
        for (root, _, files) in walkdir(indir)
            rel = relpath(root, indir)
            target = rel == "." ? outdir : joinpath(outdir, rel)
            mkpath(target)
            for f in files
                endswith(f, ".jl") || continue
                inpath = joinpath(root, f)
                @info "Literate -> " * relpath(inpath, literate_root)
                Literate.markdown(
                    inpath, target;
                    documenter = true,
                    execute = false,
                    credit = false,
                )
            end
        end
        return
    end

    # Typical folders you might want; add/remove as you wish
    render_tree(joinpath(literate_root, "tutorials"), joinpath(src_root, "tutorials"))
    render_tree(joinpath(literate_root, "research"), joinpath(src_root, "research"))
else
    @info "No Literate sources found — skipping Literate.jl step."
end

# -----------------------------------------------------------------------------

makedocs(;
    modules = [EasyHybrid],
    authors = "Lazaro Alonso, Bernhard Ahrens, Markus Reichstein",
    repo = "https://github.com/EarthyScience/EasyHybrid.jl",
    sitename = "EasyHybrid.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "https://github.com/EarthyScience/EasyHybrid.jl",
        devurl = "dev",
    ),
    pages = [
        "Home" => "index.md",
        "Get Started" => "get_started.md",
        "Tutorial" => [
            "Synthetic Respiration" => "tutorials/synthetic_respiration.md",
            "Exponential Response" => "tutorials/exponential_res.md",
            "Hyperparameter Tuning" => "tutorials/hyperparameter_tuning.md",
            "Slurm" => "tutorials/slurm.md",
            "Cross-validation" => "tutorials/folds.md",
            "Loss Functions" => "tutorials/losses.md",
        ],
        "Research" => [
            "Overview" => "research/overview.md",
        ],
        "API" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/EarthyScience/EasyHybrid.jl", # full URL!
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
