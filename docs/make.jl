# docs/make.jl
using EasyHybrid
using Documenter, DocumenterVitepress
literate_root = joinpath(@__DIR__, "literate")
notebook_root = joinpath(@__DIR__, "notebooks")

jl_files = isdir(literate_root) ?
    [joinpath(root, f) for (root, _, files) in walkdir(literate_root) for f in files if endswith(f, ".jl")] :
    String[]

if !isempty(jl_files)
    using Literate, Base64
    src_root = joinpath(@__DIR__, "src")
    notebook_preprocess(str) =
        "using Base64\ninclude_string(Main, String(base64decode($(repr(base64encode(read(joinpath(notebook_root, "setup.jl"), String)))))))\n\n" * str

    function render_tree(indir::String, md_outdir::String, nb_outdir::String)
        isdir(indir) || return
        for (root, _, files) in walkdir(indir)
            rel = relpath(root, indir)
            md_target = rel == "." ? md_outdir : joinpath(md_outdir, rel)
            nb_target = rel == "." ? nb_outdir : joinpath(nb_outdir, rel)
            mkpath(md_target)
            mkpath(nb_target)
            for f in files
                endswith(f, ".jl") || continue
                inpath = joinpath(root, f)
                Literate.markdown(inpath, md_target; documenter = true, execute = false, credit = false)
                Literate.notebook(inpath, nb_target; execute = false, documenter = false, credit = false, preprocess = notebook_preprocess)
            end
        end
    end
    render_tree(joinpath(literate_root, "tutorials"), joinpath(src_root, "tutorials"), joinpath(notebook_root, "tutorials"))
    render_tree(joinpath(literate_root, "research"), joinpath(src_root, "research"), joinpath(notebook_root, "research"))
end

# -----------------------------------------------------------------------------

makedocs(;
    modules = [EasyHybrid],
    authors = "Lazaro Alonso, Bernhard Ahrens, Markus Reichstein",
    sitename = "EasyHybrid.jl",
    format = DocumenterVitepress.MarkdownVitepress(
        repo = "github.com/EarthyScience/EasyHybrid.jl",
        devbranch = "main",
        devurl = "dev",
    ),
    source = "src",
    build = "build",
    pages = [
        "Home" => "index.md",
        "Get Started" => "get_started.md",
        "Tutorial" => [
            "Overview" => "tutorials/overview.md",
            "Building Models Examples" => "tutorials/building_models.md",
            "Dashboard" => "tutorials/dashboard.md",
            "Exponential Response" => "tutorials/exponential_res.md",
            "Hyperparameter Tuning" => "tutorials/hyperparameter_tuning.md",
            "GPU Acceleration" => "tutorials/gpu.md",
            "Synthetic Respiration on GPU" => "tutorials/synthetic_respiration_gpu.md",
            "Slurm" => "tutorials/slurm.md",
            "Cross-validation" => "tutorials/folds.md",
            "Sequence Hybrid Models (LSTM & Transformer)" => "tutorials/example_synthetic_sequence.md",
            "Loss Functions" => "tutorials/losses.md",
        ],
        "Research" => [
            "Overview" => "research/overview.md",
            "Synthetic Respiration" => "research/synthetic_respiration.md",
        ],
        "API" => "api.md",
    ],
)

DocumenterVitepress.deploydocs(;
    repo = "github.com/EarthyScience/EasyHybrid.jl.git",
    target = joinpath(@__DIR__, "build"),
    branch = "gh-pages",
    devbranch = "main",
    push_preview = true,
)
