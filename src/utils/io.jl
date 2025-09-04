export get_all_groups
export load_group
function save_ps_st(file_name, hm, ps, st, save_ps)
    hm_name = string(nameof(typeof(hm)))
    # split physical parameters
    tmp_e = if !isempty(save_ps)
        ps_values = [copy(getproperty(ps, e)[1]) for e in save_ps]
        NamedTuple{save_ps}(ps_values)
    end

    jldopen(file_name, "w") do file
        file["HybridModel_$hm_name/epoch_0"] = (ps, st)
        if !isempty(save_ps)
            file["physical_params/epoch_0"] = tmp_e
        end
    end
end

function save_ps_st!(file_name, hm, ps, st, save_ps, epoch)
    hm_name = string(nameof(typeof(hm)))
    # split physical parameters
    tmp_e = if !isempty(save_ps)
        ps_values = [copy(getproperty(ps, e)[1]) for e in save_ps]
        NamedTuple{save_ps}(ps_values)
    end

    jldopen(file_name, "a+") do file
        file["HybridModel_$hm_name/epoch_$epoch"] = (ps, st)
        if !isempty(save_ps)
            file["physical_params/epoch_$epoch"] = tmp_e
        end
    end
end

function save_train_val_loss!(file_name, train_val, train_or_val_name, epoch)
    jldopen(file_name, "a+") do file
        file["$train_or_val_name/epoch_$epoch"] = train_val
    end
end

function save_predictions!(file_name, predictions, states, train_or_val_name)
    jldopen(file_name, "a+") do file
        file["predictions/$train_or_val_name"] = predictions
        file["predictions/$(train_or_val_name)_states"] = states
    end
end
function save_observations!(file_name, target_names, yobs, train_or_val_name)
    # keyed array to NamedTuple
    named_yobs = to_named_tuple(yobs, target_names)
    jldopen(file_name, "a+") do file
        file["observations/$train_or_val_name"] = named_yobs
    end
end

function to_named_tuple(ka, target_names)
    arrays = [Array(ka(k)) for k in target_names]
    return NamedTuple{Tuple(target_names)}(arrays)
end

function load_group(file_name, group)
    group = string(group)
    group_tmp = JLD2.load(file_name, group)
    group_keys = collect(keys(group_tmp))
    if occursin("epoch", first(group_keys))
        sorted_keys = sort(group_keys, by = k -> parse(Int, split(k, "_")[end]))
        collected_data = [group_tmp[k] for k in sorted_keys]
        return collected_data, sorted_keys
    else
        return group_tmp
    end
end

function get_all_groups(filename)
    groups = Symbol[]
    JLD2.jldopen(filename, "r") do file
        function recurse_groups(g, path="")
            for k in keys(g)
                obj = g[k]
                newpath = joinpath(path, k)
                if obj isa JLD2.Group
                    push!(groups, Symbol(newpath))
                    recurse_groups(obj, newpath)
                end
            end
        end
        recurse_groups(file)
    end
    return groups
end

function resolve_path(file_name; folder_to_save="")
    file_name = isnothing(file_name) ? "trained_model.jld2" : file_name
    if !endswith(file_name, ".jld2")
        error("This needs to be a jld2 file, please include the extension as in `file_name.jld2`")
    end
    file_name = if isabspath(file_name)
        return file_name
    else
        tmp_folder = get_output_path(; folder_to_save)
        return joinpath(tmp_folder, file_name)
    end
    return file_name
end
function get_output_path(; folder_to_save="")
    base_path = dirname(Base.active_project())
    
    # Check if we're in a docs environment (common indicators)
    is_docs = any([
        basename(pwd()) == "docs",
        isdir("src") && isfile("make.jl"),
        contains(base_path, "docs")
    ])
    
    if is_docs
        return mkpath(joinpath(base_path, "build"))
    else
        return mkpath(joinpath(base_path, "output_tmp"*folder_to_save))
    end
end 

function prog_path(file_name)
    file_name = isnothing(file_name) ? "prog.txt" : file_name
    if !endswith(file_name, ".txt")
        error("This needs to be a .txt file, please include the extension as in `file_name.txt`")
    end
    file_name = if isabspath(file_name)
        return file_name
    else
        tmp_folder = mkpath(joinpath(dirname(Base.active_project()), "output_tmp"))
        return joinpath(tmp_folder, file_name)
    end
    return file_name
end