export load_hybrid_config, save_hybrid_config
export get_hybrid_config, get_train_config, get_full_config

function load_hybrid_config(path::String; dicttype = OrderedDict{String, Any})
    return load_file(path; dicttype)
end

function save_hybrid_config(config::OrderedDict, path::String)
    return write_file(path, config)
end

function get_hybrid_config(hm::LuxCore.AbstractLuxContainerLayer)
    hm_config = OrderedDict{String, Any}()
    for field in fieldnames(typeof(hm))
        hm_config[string(field)] = getfield(hm, field)
    end
    return hm_config
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
