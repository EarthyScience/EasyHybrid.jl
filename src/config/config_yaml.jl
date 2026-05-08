export load_hybrid_config, save_hybrid_config
export get_hybrid_config, get_train_config, get_full_config, get_parameters_config, get_mechanistic_model_config

function load_hybrid_config(path::String; dicttype = OrderedDict{String, Any})
    return load_file(path; dicttype)
end

function save_hybrid_config(config::OrderedDict, path::String)
    return write_file(path, config)
end

function get_hybrid_config(hm::LuxCore.AbstractLuxContainerLayer)
    hm_config = OrderedDict{String, Any}()
    for field in fieldnames(typeof(hm))
        hm_config[string(field)] = _yaml_field_value(getfield(hm, field))
    end
    return hm_config
end

_yaml_field_value(x) = x
_yaml_field_value(p::AbstractHybridModel) = get_parameters_config(p)
_yaml_field_value(f::Function) = get_mechanistic_model_config(f)

"""
    get_parameters_config(p::AbstractHybridModel)

Serialize the parameter table (`default`, `lower`, `upper` per parameter) of a
`HybridParams`/`ParameterContainer` into a nested `OrderedDict` suitable for
YAML output. Without this, only the compact `show` of the struct
(e.g. `HybridParams(ParameterContainer(RUE, Rb, Q10))`) would be written, which
drops all of the actual default and bound values.
"""
function get_parameters_config(p::AbstractHybridModel)
    pc = hasfield(typeof(p), :hybrid) ? p.hybrid : p
    out = OrderedDict{String, Any}()
    for name in keys(pc.values)
        d, l, u = pc.values[name]
        out[string(name)] = OrderedDict{String, Any}(
            "default" => d,
            "lower" => l,
            "upper" => u,
        )
    end
    return out
end

"""
    get_mechanistic_model_config(f::Function)

Build an `OrderedDict` describing a function for YAML output: its `name`, and
for each `Method`, its source `file`, starting `line`, and the full `source`
text extracted from disk by parsing one complete expression starting at that
line. The single-method case is flattened so the YAML stays compact.

Used to record `mechanistic_model` in the saved config so the exact function
definition (not just its name) is preserved alongside the run.
"""
function get_mechanistic_model_config(f::Function)
    out = OrderedDict{String, Any}()
    out["name"] = string(nameof(f))
    method_entries = OrderedDict{String, Any}[]
    for m in methods(f)
        entry = OrderedDict{String, Any}()
        try
            file, line = Base.functionloc(m)
            entry["file"] = string(file)
            entry["line"] = line
            src = _try_extract_function_source(string(file), line)
            if src !== nothing
                entry["source"] = src
            end
        catch err
            entry["error"] = sprint(showerror, err)
        end
        push!(method_entries, entry)
    end
    if length(method_entries) == 1
        merge!(out, method_entries[1])
    elseif !isempty(method_entries)
        out["methods"] = method_entries
    end
    return out
end

# Read `file` and return the source text of the first complete top-level
# expression starting at `line`. Returns `nothing` if the file can't be read or
# no expression can be parsed (e.g. function defined at the REPL or inside an
# eval'd string).
function _try_extract_function_source(file::AbstractString, line::Integer)
    isfile(file) || return nothing
    text = try
        read(file, String)
    catch
        return nothing
    end
    idx = firstindex(text)
    cur = 1
    while cur < line
        nl = findnext(==('\n'), text, idx)
        nl === nothing && return nothing
        idx = nextind(text, nl)
        cur += 1
    end
    expr_and_next = try
        Meta.parse(text, idx; greedy = true, raise = false)
    catch
        return nothing
    end
    expr_and_next === nothing && return nothing
    _, next_idx = expr_and_next
    last_idx = lastindex(text)
    endidx = next_idx > last_idx ? last_idx : prevind(text, next_idx)
    endidx < idx && return nothing
    return rstrip(text[idx:endidx])
end

function get_train_config(train_args::NamedTuple)
    train_config = OrderedDict{String, Any}()
    for field in fieldnames(typeof(train_args))
        train_config[string(field)] = getfield(train_args, field)
    end
    return train_config
end

function get_full_config(hm::LuxCore.AbstractLuxContainerLayer, train_args::NamedTuple)
    full_config = OrderedDict{String, Any}()
    full_config["hybrid_model"] = get_hybrid_config(hm)
    full_config["train_args"] = get_train_config(train_args)
    return full_config
end

to_namedtuple(cfg::TrainConfig) = NamedTuple{fieldnames(TrainConfig)}(getfield(cfg, f) for f in fieldnames(TrainConfig))
get_full_config(model, cfg::TrainConfig) = get_full_config(model, to_namedtuple(cfg))
