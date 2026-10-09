import Pkg

let d = @__DIR__, p = abspath(joinpath(d, ".."))
    Pkg.activate(basename(p) == "notebooks" && isfile(joinpath(p, "Project.toml")) ? p : d)
end

isdefined(Base, :__EH_REQUIRE_WORLD) || let world = Base.get_world_counter()
    @eval Base begin
        const __EH_REQUIRE_WORLD = $world
        const __EH_AUTOADDING = Ref(false)
        function require(into::Module, mod::Symbol)
            __EH_AUTOADDING[] && return invoke_in_world(__EH_REQUIRE_WORLD, require, into, mod)
            try
                return invoke_in_world(__EH_REQUIRE_WORLD, require, into, mod)
            catch e
                e isa ArgumentError || rethrow()
                occursin("not found in current path", e.msg) || occursin("does not seem to be installed", e.msg) || rethrow()
                name = String(mod)
                __EH_AUTOADDING[] = true
                try
                    if name == "EasyHybrid"
                        start = dirname(something(active_project(), pwd()))
                        root = nothing
                        for n in 0:3
                            cand = normpath(joinpath(start, (".." for _ in 1:n)...))
                            f = joinpath(cand, "Project.toml")
                            isfile(f) && occursin("name = \"EasyHybrid\"", read(f, String)) && (root = cand; break)
                        end
                        root === nothing ? Main.Pkg.add("EasyHybrid") : Main.Pkg.develop(; path = root)
                    elseif haskey(Main.Pkg.project().dependencies, name)
                        Main.Pkg.instantiate()
                    else
                        Main.Pkg.add(name)
                    end
                finally
                    __EH_AUTOADDING[] = false
                end
                id = identify_package(into, name)
                id === nothing && (id = identify_package(name))
                id === nothing && rethrow()
                return invokelatest(require, id)
            end
        end
    end
end
