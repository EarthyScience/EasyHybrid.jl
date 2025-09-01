export ModelSpec, tune, best_hyperparams

struct ModelSpec
    hyper_model::NamedTuple
    hyper_train::NamedTuple
end

ModelSpec(;hyper_model = NamedTuple(), hyper_train = NamedTuple()) = ModelSpec(hyper_model, hyper_train)

@generated function to_namedtuple(x)
    T = x   # a type
    names = fieldnames(T)
    types = fieldtypes(T)
    vals  = [:(getfield(x, $i)) for i in 1:length(names)]
    return :( NamedTuple{$names, Tuple{$(types...)}}(($(vals...),)) )
end

function tune(hybrid_model, data, mspec::ModelSpec; kwargs...)
    kwargs_model = merge(to_namedtuple(hybrid_model), (;kwargs...), mspec.hyper_model)
    kwargs_train = merge( (;kwargs...), mspec.hyper_train)

    hm = constructHybridModel(;kwargs_model...)

    a, b = EasyHybrid.split_data(data, hm)

    out = train(
        hm, 
        (a, b), 
        ();
        kwargs_train...
    )
end

function best_hyperparams(ho::Hyperoptimizer)
    NamedTuple{Tuple(ho.params)}(ho.minimizer)
end