using Pkg
import TOML
import Pkg.Types: VersionSpec

"""
    _needs_instantiate_or_resolve(project_path; io=stdout)

Returns a pair of Bools `(needs_resolve, needs_instantiate)`.

- `needs_instantiate` is true if the environment hasn't been booted yet
  (no Manifest, no stamp) or the Project changed since the last boot.
- `needs_resolve` is true if the current Manifest's concrete versions
  fail to satisfy `[compat]` in Project.toml (or a direct dep is missing).
"""
function _needs_instantiate_or_resolve(project_path::AbstractString; io=stdout)
    project  = joinpath(project_path, "Project.toml")
    manifest = joinpath(project_path, "Manifest.toml")
    stamp    = joinpath(project_path, ".envboot.stamp")

    # Instantiate is needed if there's no Manifest or no stamp,
    # or the Project changed after the last successful boot.
    needs_instantiate =
        !isfile(manifest) ||
        !isfile(stamp) ||
        (stat(project).mtime > stat(stamp).mtime)

    # If there is no Manifest yet, we can't check compat; resolving would
    # happen implicitly after instantiate anyway.
    if !isfile(manifest)
        return (false, true)
    end

    # Check compat satisfaction of direct deps against what's in the Manifest
    proj = TOML.parsefile(project)
    compat_tbl = get(proj, "compat", Dict{String,Any}())
    deps_tbl   = get(proj, "deps",   Dict{String,Any}())

    # Use Pkg to inspect what the current Manifest pins
    # (requires the env to be active)
    Pkg.activate(project_path; io=devnull)
    deps_map = Pkg.dependencies()  # UUID => PackageEntry(name, version, ...)
    # Note: for direct deps we can map name -> UUID via Project.toml
    needs_resolve = false

    for (name, uuid_any) in deps_tbl
        # UUIDs in Project.toml are strings; normalize to Base.UUID
        uuid = Base.UUID(string(uuid_any))
        entry = get(deps_map, uuid, nothing)

        if entry === nothing || entry.version === nothing
            # direct dep not installed/pinned in this Manifest yet
            needs_resolve = true
            println(io, "[EasyHybrid] Resolve needed: $(name) is missing from Manifest")
            break
        end

        # If there's a compat spec, ensure the Manifest version satisfies it
        if haskey(compat_tbl, name)
            spec = VersionSpec(string(compat_tbl[name]))
            v    = entry.version
            if !(v in spec)
                needs_resolve = true
                println(io, "[EasyHybrid] Resolve needed: $(name) @ $(v) violates compat \"$(compat_tbl[name])\"")
                break
            end
        end
    end

    return (needs_resolve, needs_instantiate)
end

function _is_dev_pkg(pkg_name::AbstractString, dev_path::AbstractString, io=stdout)
    for (_, entry) in Pkg.dependencies()
        if entry.name == pkg_name
            if entry.is_direct_dep && entry.source == dev_path
                println(io, "[EasyHybrid] $(pkg_name) is already in development mode.")
                return true
            end
        end
    end
    println(io, "[EasyHybrid] $(pkg_name) is not yet in development mode.")
    return false
end

"""
    dev_environment!(project_path; pkg_name="EasyHybrid", dev_path=pwd(), io=stdout)

Smart dev boot:

1) Activates `project_path`.
2) If the Project changed (or Manifest/stamp missing), runs `Pkg.instantiate()`.
3) If Manifest versions violate `[compat]` or deps are missing, runs `Pkg.resolve()` first.
4) Touches `.envboot.stamp` on success.
5) Ensures `pkg_name` is developed from `dev_path` if not already present.
"""
function dev_environment!(project_path; pkg_name::AbstractString="EasyHybrid", dev_path=pwd(), io=stdout)
    project_path = joinpath(dev_path, project_path)
    isdir(project_path) || error("Project path $project_path does not exist; are you in the right directory?")

    # Activate first so Pkg APIs reflect the target env
    Pkg.activate(project_path)
    println(io, "[EasyHybrid] Activated: ", project_path)

    # Decide what to do
    needs_resolve, needs_instantiate = _needs_instantiate_or_resolve(project_path; io=io)

    # Resolve first if required (updates Manifest to satisfy compat)
    if needs_resolve
        println(io, "[EasyHybrid] Running Pkg.resolve() to satisfy compat…")
        Pkg.resolve(; io=devnull)
    else
        println(io, "[EasyHybrid] No Pkg.resolve() needed")
    end

    # Instantiate if the env is unbooted/stale (or after a resolve)
    if needs_instantiate || needs_resolve
        println(io, "[EasyHybrid] Running Pkg.instantiate()…")
        Pkg.instantiate(; io=devnull)
        touch(joinpath(project_path, ".envboot.stamp"))
        println(io, "[EasyHybrid] Environment ready.")
    else
        println(io, "[EasyHybrid] No Pkg.instantiate() needed; using existing Manifest.")
    end

    # Develop current working tree if the package isn't already a direct dep
    isdev = _is_dev_pkg(pkg_name, dev_path, io)
    if isdev
        println(io, "[EasyHybrid] $(pkg_name) is already in development mode.")
    else
        println(io, "[EasyHybrid] Developing $(pkg_name) from: ", dev_path)
        Pkg.develop(path=dev_path)
    end

    return nothing
end