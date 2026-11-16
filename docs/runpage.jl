# Run one documentation page in this process.
#
#   julia --project=docs docs/runpage.jl docs/src/get_started.md
#   julia --project=docs docs/runpage.jl docs/literate/tutorials/folds.jl
#
# Markdown pages run their `@example`, `@ansi`, and `@setup` blocks in order.
# Blocks that share a name share a module, as they do under Documenter.
# Literate `.jl` files are included as programs.

function example_blocks(path::AbstractString)
    lines = readlines(path)
    blocks = Tuple{String, String}[]
    i = 1
    n = length(lines)
    while i <= n
        m = match(r"^(`{3,})@(example|ansi|setup)[ \t]+(\S+)\s*$", lines[i])
        if m === nothing
            i += 1
            continue
        end
        fence = m.captures[1]
        name = m.captures[3]
        i += 1
        body = String[]
        while i <= n && strip(lines[i]) != fence
            push!(body, lines[i])
            i += 1
        end
        i <= n || error("Unclosed @$name block in $path")
        push!(blocks, (name, join(body, "\n")))
        i += 1
    end
    return blocks
end

function run_markdown_page(path::AbstractString)
    blocks = example_blocks(path)
    isempty(blocks) && error("No executable blocks in $path")
    mods = Dict{String, Module}()
    for (name, code) in blocks
        mod = get!(mods, name) do
            Module(Symbol("DocsPage_", name))
        end
        include_string(mod, code)
    end
    return nothing
end

function run_page(path::AbstractString)
    page = abspath(path)
    @info "Running documentation page" page
    if endswith(page, ".jl")
        include(page)
    elseif endswith(page, ".md")
        run_markdown_page(page)
    else
        error("Expected a .jl or .md page, got $page")
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("Usage: julia docs/runpage.jl <page.md|page.jl>")
    run_page(ARGS[1])
end
