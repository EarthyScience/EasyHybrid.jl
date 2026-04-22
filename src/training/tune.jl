export ModelSpec, tune, best_hyperparams

struct ModelSpec
    hyper_model::NamedTuple
    hyper_train::NamedTuple
end

ModelSpec(; hyper_model = NamedTuple(), hyper_train = NamedTuple()) = ModelSpec(hyper_model, hyper_train)

@generated function to_namedtuple(x)
    T = x   # a type
    names = fieldnames(T)
    types = fieldtypes(T)
    vals = [:(getfield(x, $i)) for i in 1:length(names)]
    return :(NamedTuple{$names, Tuple{$(types...)}}(($(vals...),)))
end

"""
    tune(hybrid_model, data, mspec::ModelSpec; kwargs...)
    tune(hybrid_model, data; kwargs...)
    tune(hybrid_model, data, train_cfg::TrainConfig; data_cfg::DataConfig = DataConfig(), kwargs...)

Construct a new hybrid model from `hybrid_model` plus hyperparameters, then call [`train`](@ref).

Returns a [`TrainResults`](@ref) (or `nothing` if data preparation fails, as in `train`).
"""
function tune(hybrid_model, data, mspec::ModelSpec; kwargs...)
    kwargs_model = merge(to_namedtuple(hybrid_model), hybrid_model.config, (; kwargs...), mspec.hyper_model)
    kwargs_train = merge((; kwargs...), mspec.hyper_train)

    hm = constructHybridModel(; kwargs_model...)

    train_cfg, data_cfg = EasyHybrid.kwargs_to_configs((), kwargs_train)
    return train(hm, data; train_cfg, data_cfg)
end

function tune(hybrid_model, data; kwargs...)
    kwargs_model = merge(to_namedtuple(hybrid_model), hybrid_model.config, (; kwargs...))

    hm = constructHybridModel(; kwargs_model...)

    train_cfg, data_cfg = EasyHybrid.kwargs_to_configs((), (; kwargs...))
    return train(hm, data; train_cfg, data_cfg)
end

function tune(hybrid_model, data, train_cfg::TrainConfig; data_cfg::DataConfig = DataConfig(), kwargs...)
    kwargs_model = merge(to_namedtuple(hybrid_model), hybrid_model.config, to_namedtuple(train_cfg), to_namedtuple(data_cfg), (; kwargs...))
    hm = constructHybridModel(; kwargs_model...)

    train_cfg, data_cfg = EasyHybrid.kwargs_to_configs((), merge(to_namedtuple(train_cfg), to_namedtuple(data_cfg), (; kwargs...)))

    return train(hm, data; train_cfg, data_cfg)
end

function best_hyperparams(ho::Hyperoptimizer)
    return NamedTuple{Tuple(ho.params)}(ho.minimizer)
end
