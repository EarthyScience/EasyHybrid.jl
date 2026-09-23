using LuxZarr: LuxZarr, save_model, load_model, extract_model_info, reconstruct_model_from_info,
    LazyParameters, LazyState, LazyLuxModel, materialize

export save_model, load_model, LazyParameters, LazyState, LazyLuxModel, materialize

# --- Metadata Extraction ---

function LuxZarr.extract_model_info(m::HybridModel)
    nns_info = LuxZarr.extract_model_info(m.NNs)

    preds_info = if m.predictors isa NamedTuple
        d = Dict{String, Any}("__is_namedtuple__" => true)
        for (k, v) in pairs(m.predictors)
            d[string(k)] = string.(v)
        end
        d
    else
        string.(m.predictors)
    end

    param_dict = Dict{String, Any}()
    for name in keys(m.parameters.values)
        spec = m.parameters.values[name]
        d, l, u = spec[1], spec[2], spec[3]
        scale_sym = get(m.parameters.scales, name, :linear)
        param_dict[string(name)] = [Float64(d), Float64(l), Float64(u), string(scale_sym)]
    end

    mech_cfg = get_mechanistic_model_config(m.mechanistic_model)

    return Dict{String, Any}(
        "type" => "HybridModel",
        "summary" => string(m),
        "NNs" => nns_info,
        "predictors" => preds_info,
        "forcing" => string.(m.forcing),
        "targets" => string.(m.targets),
        "mechanistic_model_config" => mech_cfg,
        "parameters" => param_dict,
        "neural_param_names" => string.(m.neural_param_names),
        "global_param_names" => string.(m.global_param_names),
        "fixed_param_names" => string.(m.fixed_param_names),
        "scale_nn_outputs" => m.scale_nn_outputs,
        "start_from_default" => m.start_from_default,
        "config" => Dict{String, Any}(string(k) => _serialize_config_val(v) for (k, v) in pairs(m.config)),
    )
end

_serialize_config_val(v) = string(v)
_serialize_config_val(v::Union{Number, String, Bool}) = v
_serialize_config_val(v::Vector{Int}) = v
_serialize_config_val(v::Vector{<:AbstractString}) = v
_serialize_config_val(v::NamedTuple) = Dict{String, Any}(string(k) => _serialize_config_val(val) for (k, val) in pairs(v))

# --- Function Resolution & Model Reconstruction ---

function _resolve_mechanistic_model(mech_cfg::AbstractDict, user_override)
    user_override !== nothing && return user_override

    fn_name = get(mech_cfg, "name", "")
    if !isempty(fn_name)
        sym = Symbol(fn_name)
        if isdefined(Main, sym)
            return getfield(Main, sym)
        elseif isdefined(EasyHybrid, sym)
            return getfield(EasyHybrid, sym)
        end
    end

    src = get(mech_cfg, "source", nothing)
    if src !== nothing && !isempty(strip(src))
        try
            evaluated = Base.include_string(Main, src)
            if evaluated isa Function
                return evaluated
            end
            if !isempty(fn_name) && isdefined(Main, Symbol(fn_name))
                return getfield(Main, Symbol(fn_name))
            end
        catch err
            @debug "Failed to evaluate mechanistic model source: $err"
        end
    end

    throw(
        ArgumentError(
            "Could not resolve the mechanistic model function '$(fn_name)' automatically. " *
                "Please provide it explicitly via `load_model(path; mechanistic_model=your_function)`."
        )
    )
end

function _reconstruct_hybrid_model(info::AbstractDict; mechanistic_model = nothing, kwargs...)
    nns_info = info["NNs"]
    NNs = LuxZarr.reconstruct_model_from_info(nns_info; kwargs...)
    if NNs === nothing
        throw(ArgumentError("Failed to reconstruct neural network architecture for HybridModel."))
    end

    preds_raw = info["predictors"]
    predictors = if preds_raw isa AbstractDict && get(preds_raw, "__is_namedtuple__", false)
        keys_tuple = Tuple(Symbol(k) for k in keys(preds_raw) if k != "__is_namedtuple__")
        vals_tuple = Tuple(Symbol.(preds_raw[string(k)]) for k in keys_tuple)
        NamedTuple{keys_tuple}(vals_tuple)
    else
        Symbol.(preds_raw)
    end

    forcing = Symbol[Symbol(x) for x in info["forcing"]]
    targets = Symbol[Symbol(x) for x in info["targets"]]

    mech_cfg = get(info, "mechanistic_model_config", Dict{String, Any}())
    mech_fn = _resolve_mechanistic_model(mech_cfg, mechanistic_model)

    param_dict = info["parameters"]
    param_keys = Tuple(Symbol(k) for k in keys(param_dict))
    param_vals = Tuple(
        begin
                v = param_dict[string(k)]
                d, l, u = Float64(v[1]), Float64(v[2]), Float64(v[3])
                scale_sym = length(v) >= 4 ? Symbol(v[4]) : :linear
                (d, l, u, scale_sym)
            end
            for k in param_keys
    )
    parameters = ParameterContainer(NamedTuple{param_keys}(param_vals))

    neural_param_names = Symbol[Symbol(x) for x in info["neural_param_names"]]
    global_param_names = Symbol[Symbol(x) for x in info["global_param_names"]]
    fixed_param_names = Symbol[Symbol(x) for x in info["fixed_param_names"]]
    scale_nn_outputs = Bool(info["scale_nn_outputs"])
    start_from_default = Bool(info["start_from_default"])
    config = NamedTuple()

    return HybridModel(
        NNs,
        predictors,
        forcing,
        targets,
        mech_fn,
        parameters,
        neural_param_names,
        global_param_names,
        fixed_param_names,
        scale_nn_outputs,
        start_from_default,
        config,
    )
end

function LuxZarr.reconstruct_layer(::Val{:HybridModel}, info::AbstractDict; mechanistic_model = nothing, kwargs...)
    return _reconstruct_hybrid_model(info; mechanistic_model = mechanistic_model, kwargs...)
end

function LuxZarr.extract_model_info(m::InputBatchNorm)
    return Dict{String, Any}(
        "type" => "InputBatchNorm",
        "summary" => string(m),
        "layer" => extract_model_info(m.layer),
    )
end

function LuxZarr.reconstruct_layer(::Val{:InputBatchNorm}, info::AbstractDict; kwargs...)
    layer_info = get(info, "layer", nothing)
    layer = reconstruct_model_from_info(layer_info; kwargs...)
    layer === nothing && return nothing
    return InputBatchNorm(layer)
end

function LuxZarr.extract_model_info(m::RecurrenceOutputDense)
    return Dict{String, Any}(
        "type" => "RecurrenceOutputDense",
        "summary" => string(m),
        "layer" => extract_model_info(m.layer),
    )
end

function LuxZarr.reconstruct_layer(::Val{:RecurrenceOutputDense}, info::AbstractDict; kwargs...)
    layer_info = get(info, "layer", nothing)
    layer = reconstruct_model_from_info(layer_info; kwargs...)
    layer === nothing && return nothing
    return RecurrenceOutputDense(layer)
end
