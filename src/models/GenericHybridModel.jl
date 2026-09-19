export HybridModel, ParameterContainer, constructHybridModel

"""
    ParameterContainer{NT <: NamedTuple, T}

A container for holding the parameter definitions of a model, including their default values, lower bounds, and upper bounds.

$(TYPEDFIELDS)
"""
mutable struct ParameterContainer{NT <: NamedTuple, T}
    "The raw parameter definitions. A `NamedTuple` where each entry is a tuple of `(default, lower, upper)` bounds for a parameter."
    values::NT

    "A `ComponentArray` matrix representation of the parameter bounds, organized for efficient access by name and bound type."
    table::T

    function ParameterContainer(values::NT) where {NT <: NamedTuple}
        table = build_parameter_matrix(values)
        return new{NT, typeof(table)}(values, table)
    end
end

"""
    HybridModel{T, P} <: LuxCore.AbstractLuxContainerLayer{(:NNs,)}

A unified hybrid model struct that handles both single and multi neural network architectures.
It combines predictive neural networks (`NNs`) with a `mechanistic_model` to form a differentiable hybrid model.

$(TYPEDFIELDS)
"""
struct HybridModel{T, P, MM, NP, GP, FP, KW, SN, TG} <: LuxCore.AbstractLuxContainerLayer{(:NNs,)}
    "Neural network(s) used to predict parameters. Can be a single `Chain` or a `NamedTuple` of `Chain`s."
    NNs::T

    "Predictor variables for the neural networks. Can be a `Vector{Symbol}` or a `NamedTuple`."
    predictors::P

    "Forcing variables passed directly to the mechanistic model."
    forcing::Vector{Symbol}

    "Target variables the model will output/be trained against."
    targets::Vector{Symbol}

    "The core process-based or mechanistic model function."
    mechanistic_model::MM

    "Base parameters of the model (encapsulated in a `ParameterContainer`)."
    parameters::ParameterContainer

    "Names of the parameters predicted by the neural network(s)."
    neural_param_names::Vector{Symbol}

    "Names of the globally optimized (constant) parameters."
    global_param_names::Vector{Symbol}

    "Names of the fixed (non-optimized) parameters."
    fixed_param_names::Vector{Symbol}

    "Whether to scale neural network outputs to the parameter bounds."
    scale_nn_outputs::Bool

    "Whether to initialize global parameters from their default values."
    start_from_default::Bool

    "Configuration named tuple capturing the hyperparameters used for initialization."
    config::NamedTuple
end

function HybridModel(
        NNs, predictors, forcing, targets, mechanistic_model, parameters,
        neural_param_names, global_param_names, fixed_param_names,
        scale_nn_outputs, start_from_default, config,
    )
    NP, GP, FP = Tuple(neural_param_names), Tuple(global_param_names), Tuple(fixed_param_names)
    KW = _accepted_kwarg_names(
        mechanistic_model,
        Tuple(unique([forcing; neural_param_names; global_param_names; fixed_param_names])),
    )
    return HybridModel{
        typeof(NNs), typeof(predictors), typeof(mechanistic_model),
        NP, GP, FP, KW, scale_nn_outputs, Tuple(targets),
    }(
        NNs, predictors, forcing, targets, mechanistic_model, parameters,
        neural_param_names, global_param_names, fixed_param_names,
        scale_nn_outputs, start_from_default, config,
    )
end

"""
    constructHybridModel(predictors::Vector{Symbol}, forcing, targets, mechanistic_model, parameters, neural_param_names, global_param_names; kwargs...)

Construct a `HybridModel` with a single neural network architecture predicting all `neural_param_names` from the `predictors`.

# Arguments:
- `predictors::Vector{Symbol}`: Variables used as inputs to the neural network.
- `forcing`: Variables passed directly to the mechanistic model.
- `targets`: The target variables to predict.
- `mechanistic_model`: A function implementing the process-based model.
- `parameters`: A parameter container defining defaults, lowers, and uppers.
- `neural_param_names`: Names of the parameters to be predicted by the neural network.
- `global_param_names`: Names of the parameters to be globally optimized.
- `kwargs`: Additional configuration like `hidden_layers`, `activation`, `scale_nn_outputs`, etc.
"""
function constructHybridModel(
        predictors::Vector{Symbol},
        forcing,
        targets,
        mechanistic_model,
        parameters,
        neural_param_names,
        global_param_names;
        hidden_layers::Union{Vector{Int}, Chain} = [32, 32],
        activation = tanh,
        scale_nn_outputs = false,
        input_batchnorm = false,
        start_from_default = true,
        kwargs...
    )

    if !isa(parameters, ParameterContainer)
        parameters = ParameterContainer(parameters)
    end

    all_names = pnames(parameters)
    @assert all(n in all_names for n in neural_param_names) "neural_param_names ⊆ param_names"

    # if empty predictors do not construct NN
    if length(predictors) > 0 && length(neural_param_names) > 0

        in_dim = length(predictors)
        out_dim = length(neural_param_names)

        NN = prepare_hidden_chain(
            hidden_layers, in_dim, out_dim;
            activation = activation,
            input_batchnorm = input_batchnorm
        )
    else
        NN = Chain()
    end

    fixed_param_names = [ n for n in all_names if !(n in [neural_param_names..., global_param_names...]) ]

    # capture the configuration used for construction
    config = (;
        hidden_layers,
        activation,
        scale_nn_outputs,
        input_batchnorm,
        start_from_default,
        kwargs...,
    )

    return HybridModel(
        NN,
        predictors,
        forcing,
        targets,
        mechanistic_model,
        parameters,
        neural_param_names,
        global_param_names,
        fixed_param_names,
        scale_nn_outputs,
        start_from_default,
        config
    )
end

"""
    constructHybridModel(predictors::NamedTuple, forcing, targets, mechanistic_model, parameters, global_param_names; kwargs...)

Construct a `HybridModel` with multiple neural network architectures. A separate neural network is built for each key in the `predictors` NamedTuple.

# Arguments:
- `predictors::NamedTuple`: A NamedTuple where keys are network names, and values are vectors of predictor variables for that network.
- `forcing`: Variables passed directly to the mechanistic model.
- `targets`: The target variables to predict.
- `mechanistic_model`: A function implementing the process-based model.
- `parameters`: A parameter container defining defaults, lowers, and uppers.
- `global_param_names`: Names of the parameters to be globally optimized.
- `kwargs`: Additional configuration. `hidden_layers` and `activation` can also be NamedTuples to configure each network independently.
"""
function constructHybridModel(
        predictors::NamedTuple,
        forcing,
        targets,
        mechanistic_model,
        parameters,
        global_param_names;
        hidden_layers::Union{Vector{Int}, Chain, NamedTuple} = [32, 32],
        activation::Union{Function, NamedTuple} = tanh,
        scale_nn_outputs = false,
        input_batchnorm = false,
        start_from_default = true,
        kwargs...
    )

    if !isa(parameters, ParameterContainer)
        parameters = ParameterContainer(parameters)
    end

    all_names = pnames(parameters)
    neural_param_names = collect(keys(predictors))
    # Create neural networks based on predictors
    NNs = NamedTuple()
    for (nn_name, preds) in pairs(predictors)
        # Create a simple NN for each predictor set
        in_dim = length(preds)
        out_dim = 1
        if hidden_layers isa NamedTuple
            if activation isa NamedTuple
                nn = prepare_hidden_chain(
                    hidden_layers[nn_name], in_dim, out_dim;
                    activation = activation[nn_name],
                    input_batchnorm = input_batchnorm
                )
            else
                nn = prepare_hidden_chain(
                    hidden_layers[nn_name], in_dim, out_dim;
                    activation = activation,
                    input_batchnorm = input_batchnorm
                )
            end
        else
            nn = prepare_hidden_chain(
                hidden_layers, in_dim, out_dim;
                activation = activation,
                input_batchnorm = input_batchnorm
            )
        end
        NNs = merge(NNs, NamedTuple{(nn_name,), Tuple{typeof(nn)}}((nn,)))
    end

    fixed_param_names = [ n for n in all_names if !(n in [neural_param_names..., global_param_names...]) ]

    # capture the configuration used for construction
    config = (;
        hidden_layers,
        activation,
        scale_nn_outputs,
        input_batchnorm,
        start_from_default,
        kwargs...,
    )

    return HybridModel(
        NNs,
        predictors,
        forcing,
        targets,
        mechanistic_model,
        parameters,
        neural_param_names,
        global_param_names,
        fixed_param_names,
        scale_nn_outputs,
        start_from_default,
        config
    )
end

function constructHybridModel(
        ; predictors,
        forcing,
        targets,
        mechanistic_model,
        parameters,
        neural_param_names = nothing,
        global_param_names,
        kwargs...
    )
    if predictors isa Vector{Symbol}
        @assert neural_param_names !== nothing "Provide neural_param_names for Vector predictors"
        return constructHybridModel(
            predictors, forcing, targets, mechanistic_model, parameters,
            neural_param_names, global_param_names; kwargs...
        )
    elseif predictors isa NamedTuple
        return constructHybridModel(
            predictors, forcing, targets, mechanistic_model, parameters,
            global_param_names; kwargs...
        )
    else
        throw(ArgumentError("predictors must be Vector{Symbol} or NamedTuple, got $(typeof(predictors))"))
    end
end

"""
    _init_nn_params(rng, m::HybridModel{<:Any, <:NamedTuple})

Initialize parameters for a multi-neural network architecture.
Returns a `NamedTuple` containing the initialized parameters for each sub-network.
"""
function _init_nn_params(rng::AbstractRNG, m::HybridModel{<:Any, <:NamedTuple})
    return map(nn -> LuxCore.setup(rng, nn)[1], m.NNs)
end

"""
    _init_nn_params(rng, m::HybridModel{<:Any, <:Vector})

Initialize parameters for a single-neural network architecture.
Returns a `NamedTuple` containing a single `ps` field with the network's parameters.
"""
function _init_nn_params(rng::AbstractRNG, m::HybridModel{<:Any, <:Vector})
    ps_nn, _ = LuxCore.setup(rng, m.NNs)
    return (; ps = ps_nn)
end

# Initial parameters for HybridModel
function LuxCore.initialparameters(rng::AbstractRNG, m::HybridModel)
    nt = _init_nn_params(rng, m)

    # Then append each global parameter as a 1-vector of Float32
    if !isempty(m.global_param_names)
        if m.start_from_default
            for g in m.global_param_names
                default_val = scale_single_param_minmax(g, m.parameters)
                nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
            end
        else
            for g in m.global_param_names
                random_val = rand(rng, Float32)
                nt = merge(nt, NamedTuple{(g,), Tuple{Vector{Float32}}}(([random_val],)))
            end
        end
    end

    return nt
end

"""
    _init_nn_states(rng, m::HybridModel{<:Any, <:NamedTuple})

Initialize states for a multi-neural network architecture.
Returns a `NamedTuple` containing the initialized states for each sub-network.
"""
function _init_nn_states(rng::AbstractRNG, m::HybridModel{<:Any, <:NamedTuple})
    return map(nn -> LuxCore.setup(rng, nn)[2], m.NNs)
end

"""
    _init_nn_states(rng, m::HybridModel{<:Any, <:Vector})

Initialize states for a single-neural network architecture.
Returns a `NamedTuple` containing a single `st_nn` field with the network's states.
"""
function _init_nn_states(rng::AbstractRNG, m::HybridModel{<:Any, <:Vector})
    _, st_nn = LuxCore.setup(rng, m.NNs)
    return (; st_nn = st_nn)
end

# Initial states for HybridModel
function LuxCore.initialstates(rng::AbstractRNG, m::HybridModel)
    nn_states_nt = _init_nn_states(rng, m)
    nt = (;)

    # Then append each fixed parameter as a 1-vector of Float32
    if !isempty(m.fixed_param_names)
        for f in m.fixed_param_names
            default_val = default(m.parameters)[f]
            nt = merge(nt, NamedTuple{(f,), Tuple{Vector{Float32}}}(([Float32(default_val)],)))
        end
    end

    return merge(nn_states_nt, (; fixed = nt))
end

"""
    _run_nn(m::HybridModel{<:Any, <:NamedTuple}, ds_k::Tuple, ps, st)

Execute the forward pass for a multi-neural network architecture.
Applies each sub-network to its specific predictors, and applies scaling to the outputs if required.
Returns scaled parameter values, updated states, and raw network outputs.
"""
function _run_nn(m::HybridModel{<:Any, <:NamedTuple}, ds_k::Tuple, ps, st)
    nn_names = keys(m.NNs)
    applied = map(nn_names) do nn_name
        LuxCore.apply(m.NNs[nn_name], ds_k[1][nn_name], ps[nn_name], st[nn_name])
    end
    nn_outputs = NamedTuple{nn_names}(map(first, applied))
    nn_states = NamedTuple{nn_names}(map(last, applied))

    scaled_vals = Tuple(
        begin
            val = eachslice(nn_outputs[nn_name]; dims = 1)[1]
            m.scale_nn_outputs ? scale_single_param(param_name, val, m.parameters) : val
        end
            for (nn_name, param_name) in zip(keys(m.NNs), m.neural_param_names)
    )
    scaled_nn_params = NamedTuple{Tuple(m.neural_param_names)}(scaled_vals)

    return scaled_nn_params, nn_states, (; nn_outputs = nn_outputs)
end

"""
    _run_nn(m::HybridModel{<:Any, <:Vector}, ds_k::Tuple, ps, st)

Execute the forward pass for a single-neural network architecture.
Applies the neural network to the given predictors, slices the output for multiple predicted parameters, and scales them if required.
Returns scaled parameter values, updated states, and raw network outputs.
"""
function _run_nn(m::HybridModel{<:Any, <:Vector}, ds_k::Tuple, ps, st)
    if !isempty(m.neural_param_names)
        nn_out, st_nn = LuxCore.apply(m.NNs, ds_k[1], ps.ps, st.st_nn)
        nn_cols = eachslice(nn_out, dims = 1)
        nn_params = NamedTuple(zip(m.neural_param_names, nn_cols))

        if m.scale_nn_outputs
            scaled_nn_vals = Tuple(
                scale_single_param(name, nn_params[name], m.parameters)
                    for name in m.neural_param_names
            )
        else
            scaled_nn_vals = Tuple(nn_params[name] for name in m.neural_param_names)
        end
        scaled_nn_params = NamedTuple(zip(m.neural_param_names, scaled_nn_vals))
    else
        scaled_nn_params = NamedTuple()
        st_nn = st.st_nn
    end
    return scaled_nn_params, (; st_nn = st_nn), (;)
end

@generated function _hybrid_loss(
        m::HybridModel{<:Any, <:Vector, MM, NP, GP, FP, KW, SN, TG},
        ds_k::Tuple, ps, st, y, y_nan, ::Val{S}, agg
    ) where {MM, NP, GP, FP, KW, SN, TG, S}
    stmts = Expr[]
    names = (NP..., GP..., FP...)
    if NP === ()
        push!(stmts, :(st_nn = st.st_nn))
    else
        push!(stmts, :((nn_out, st_nn) = LuxCore.apply(m.NNs, ds_k[1], ps.ps, st.st_nn)))
        push!(stmts, :(slices = eachslice(nn_out, dims = 1)))
    end
    for (i, n) in enumerate(NP)
        rhs = SN ?
            :(scale_single_param(Val($(QuoteNode(n))), slices[$i], m.parameters)) :
            :(slices[$i])
        push!(stmts, Expr(:(=), n, rhs))
    end
    for n in GP
        push!(stmts, Expr(:(=), n, :(scale_single_param(Val($(QuoteNode(n))), getproperty(ps, $(QuoteNode(n))), m.parameters))))
    end
    for n in FP
        push!(stmts, Expr(:(=), n, :(getproperty(st.fixed, $(QuoteNode(n))))))
    end
    NP === () && GP === () && push!(stmts, :(ChainRulesCore.ignore_derivatives(ps)))
    NP === () && KW === () && push!(stmts, :(ChainRulesCore.ignore_derivatives(ds_k)))
    !SN && GP === () && push!(stmts, :(ChainRulesCore.ignore_derivatives(m.parameters)))
    if KW === nothing
        push!(stmts, :(all_params = (; $(names...))))
        push!(stmts, :(y_pred = m.mechanistic_model(; merge(ds_k[2], all_params)...)))
    else
        args = Expr[]
        for n in KW
            if n in names
                push!(args, Expr(:kw, n, n))
            else
                push!(args, Expr(:kw, n, :(getfield(ds_k[2], $(QuoteNode(n))))))
            end
        end
        push!(stmts, :(y_pred = m.mechanistic_model(; $(args...))))
    end
    if TG === ()
        push!(stmts, :(ChainRulesCore.ignore_derivatives((y, y_nan, agg))))
        push!(stmts, :(return agg(()), st_nn))
    else
        terms = Expr[]
        for t in TG
            qt = QuoteNode(t)
            yi = y <: NamedTuple ? :(getfield(y, $qt)) : :(_get_target_y(y, $qt))
            nani = y_nan <: NamedTuple ? :(getfield(y_nan, $qt)) : :(_get_target_y(y_nan, $qt))
            loss = S === :mse ?
                :(mean(abs2, ŷ_i[nan_i] .- y_i[nan_i])) :
                :(_apply_loss(ŷ_i, y_i, nan_i, $(QuoteNode(S))))
            push!(
                terms, :(
                    let
                        y_i = $yi
                        ŷ_i = _get_target_ŷ(y_pred, y_i, $qt)
                        nan_i = $nani
                        $loss
                    end
                )
            )
        end
        push!(stmts, :(return agg(($(terms...),)), st_nn))
    end
    return Expr(:block, stmts...)
end

"""
    (m::HybridModel)(ds_k::Tuple, ps, st)

Forward pass of the hybrid model.
Evaluates the neural networks to predict parameters, merges them with scaled global parameters and fixed parameters, and executes the mechanistic model.
Returns a tuple `(out, st_new)`.
"""
function (m::HybridModel)(ds_k::Tuple, ps, st)
    parameters = m.parameters

    # 1) Scale global parameters (handle empty case)
    if !isempty(m.global_param_names)
        global_vals = Tuple(
            scale_single_param(g, ps[g], parameters)
                for g in m.global_param_names
        )
        global_params = NamedTuple{Tuple(m.global_param_names), Tuple{typeof.(global_vals)...}}(global_vals)
    else
        global_params = NamedTuple()
    end

    # 2) Run neural network(s)
    scaled_nn_params, st_new_nns, out_extra = _run_nn(m, ds_k, ps, st)

    # 3) Pick fixed parameters (handle empty case)
    if !isempty(m.fixed_param_names)
        fixed_vals = Tuple(st.fixed[f] for f in m.fixed_param_names)
        fixed_params = NamedTuple{Tuple(m.fixed_param_names), Tuple{typeof.(fixed_vals)...}}(fixed_vals)
    else
        fixed_params = NamedTuple()
    end

    # 4) merge all parameters
    all_params = merge(scaled_nn_params, global_params, fixed_params)

    # 5) unpack forcing data
    forcing_data = ds_k[2]
    all_kwargs = merge(forcing_data, all_params)

    # 6) Apply mechanistic model. Only forward the kwargs it actually declares, so
    #    "loss-only" parameters (e.g. a learned noise scale used only in the loss)
    #    can be defined without the mechanistic model having to accept them. They
    #    still live in `all_params` and are exposed below under `parameters`.
    y_pred = m.mechanistic_model(; _mechanistic_kwargs(m.mechanistic_model, all_kwargs)...)

    # Parameters the mechanistic model does not consume (e.g. loss-only ones such as
    # a learned noise scale) are surfaced at the top level so they can be monitored
    # and plotted, in addition to always being available under `parameters`.
    extra_params = _extra_params(m.mechanistic_model, all_params)
    out = (; y_pred..., extra_params..., parameters = all_params, out_extra...)
    st_new = (; st_new_nns..., fixed = st.fixed)

    return out, ChainRulesCore.ignore_derivatives(st_new)
end

function (m::HybridModel)(ds_k, ps, st)
    # Forward pass fallback when ds_k is not explicitly typed as Tuple
    return m(Tuple(ds_k), ps, st)
end

"""
    _mechanistic_kwargs(f, all_kwargs::NamedTuple)

Select from `all_kwargs` only the keyword arguments the mechanistic model `f`
declares, so parameters used solely by the loss (e.g. a learned noise scale) do
not need to be accepted by `f`. Falls back to passing everything when `f` slurps
`kwargs...` or its keyword signature cannot be introspected.
"""
function _mechanistic_kwargs(f, all_kwargs::NamedTuple)
    keep = ChainRulesCore.ignore_derivatives() do
        _accepted_kwarg_names(f, keys(all_kwargs))
    end
    keep === nothing && return all_kwargs
    return NamedTuple{keep}(map(k -> all_kwargs[k], keep))
end

"""
    _extra_params(f, all_params::NamedTuple)

The parameters not consumed by the mechanistic model `f` (e.g. loss-only ones such
as a learned noise scale). They are surfaced at the top level of the model output
so they can be monitored/plotted, in addition to always being available under
`parameters`. Returns an empty `NamedTuple` when `f` consumes everything.
"""
function _extra_params(f, all_params::NamedTuple)
    keep = ChainRulesCore.ignore_derivatives() do
        acc = _accepted_kwarg_names(f, keys(all_params))
        acc === nothing ? () : Tuple(k for k in keys(all_params) if !(k in acc))
    end
    return NamedTuple{keep}(map(k -> all_params[k], keep))
end

# Returns the tuple of `all_kwargs` names accepted by `f`, or `nothing` to signal
# "pass everything" (the model slurps `kwargs...`, or has no introspectable kwargs).
function _accepted_kwarg_names(f, available::Tuple)
    names = Symbol[]
    for mth in methods(f)
        for d in Base.kwarg_decl(mth)
            endswith(string(d), "...") && return nothing  # slurps kwargs → keep all
            push!(names, d)
        end
    end
    isempty(names) && return nothing
    return Tuple(k for k in available if k in names)
end

function (m::HybridModel)(df::DataFrame, ps, st)
    @warn "Only makes sense in test mode, not training!"

    # Process numeric or missing-containing columns
    for col in names(df)
        what_type = eltype(df[!, col])
        if what_type <: Union{Missing, Real} || what_type <: Real
            df[!, col] = Float32.(coalesce.(df[!, col], NaN))
        end
    end

    all_data = to_keyedArray(df)
    x, _ = prepare_data(m, all_data)
    out, _ = m(x, ps, LuxCore.testmode(st))
    dfnew = copy(df)
    n_samples = x[1] isa NamedTuple ? size(first(values(x[1])), 2) : size(x[1], 2)
    for k in keys(out)
        if length(out[k]) == n_samples
            dfnew[!, String(k) * "_pred"] = out[k]
        end
    end
    return dfnew
end
