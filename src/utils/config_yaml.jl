export load_hybrid_config, save_hybrid_config
function load_hybrid_config(path::String; dicttype=OrderedDict{String,Any})
    return load_file(path; dicttype, )
end

function save_hybrid_config(config::Dict, path::String)
    return write_file(path, config)
end

function get_hybrid_config(hm::HybridModel)
    hm_config = Dict{String,Any}()
    for field in fieldnames(typeof(hm))
        hm_config[string(field)] = getfield(hm, field)
    end 
    return hm_config
end

function get_train_config(train_args::TrainResults)
    train_config = Dict{String,Any}()
    for field in fieldnames(typeof(train_args))
        train_config[string(field)] = getfield(train_args, field)
    end 
    return train_config
end