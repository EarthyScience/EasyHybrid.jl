export @hybrid

"""
    HybridModelSpec

Internal intermediate representation (IR) capturing parsed AST specifications for `@hybrid`.
"""
struct HybridModelSpec
    mechanistic::Any
    targets::Any
    forcing::Any
    neural::Any
    parameters::Any
    global_params::Any
    fixed_params::Any
    kwargs::Vector{Pair{Symbol, Any}}
    source::LineNumberNode
end

"""
    _parse_hybrid_block(block::Expr, source::LineNumberNode)

Parse a `begin ... end` AST block into a `HybridModelSpec`.
"""
function _parse_hybrid_block(block::Expr, source::LineNumberNode)
    mechanistic = nothing
    targets = nothing
    forcing = nothing
    neural = nothing
    parameters = nothing
    global_params = nothing
    fixed_params = nothing
    kwargs = Pair{Symbol, Any}[]
    current_source = source

    for arg in block.args
        if arg isa LineNumberNode
            current_source = arg
            continue
        end

        if arg isa Expr && arg.head in (:(=), :kw)
            key = arg.args[1]
            val = arg.args[2]

            if key in (:mechanistic, :model, :mechanistic_model, :process)
                mechanistic = val
            elseif key in (:target, :targets)
                targets = val
            elseif key in (:forcing, :forcings)
                forcing = val
            elseif key in (:neural, :predictors, :nn)
                neural = val
            elseif key in (:parameters, :params)
                parameters = val
            elseif key in (:global_params, :global_param_names, :globals)
                global_params = val
            elseif key in (:fixed_params, :fixed_param_names, :fixed)
                fixed_params = val
            else
                push!(kwargs, Symbol(key) => val)
            end
        elseif arg isa Expr && arg.head == :call && arg.args[1] == :(=>)
            # Allow top-level pair for neural specification: [:in] => [:out]
            neural = arg
        end
    end

    spec = HybridModelSpec(
        mechanistic,
        targets,
        forcing,
        neural,
        parameters,
        global_params,
        fixed_params,
        kwargs,
        current_source
    )
    _validate_hybrid_spec(spec)
    return spec
end

"""
    _parse_hybrid_inline(args, source::LineNumberNode)

Parse inline macro arguments (e.g. `@hybrid model targets=...`) into a `HybridModelSpec`.
"""
function _parse_hybrid_inline(args, source::LineNumberNode)
    mechanistic = nothing
    targets = nothing
    forcing = nothing
    neural = nothing
    parameters = nothing
    global_params = nothing
    fixed_params = nothing
    kwargs = Pair{Symbol, Any}[]

    positional_args = []

    for arg in args
        if arg isa Expr && arg.head in (:(=), :kw)
            key = arg.args[1]
            val = arg.args[2]

            if key in (:mechanistic, :model, :mechanistic_model, :process)
                mechanistic = val
            elseif key in (:target, :targets)
                targets = val
            elseif key in (:forcing, :forcings)
                forcing = val
            elseif key in (:neural, :predictors, :nn)
                neural = val
            elseif key in (:parameters, :params)
                parameters = val
            elseif key in (:global_params, :global_param_names, :globals)
                global_params = val
            elseif key in (:fixed_params, :fixed_param_names, :fixed)
                fixed_params = val
            else
                push!(kwargs, Symbol(key) => val)
            end
        else
            push!(positional_args, arg)
        end
    end

    # Assign positional arguments following the physics-first mental model:
    # 1: mechanistic, 2: targets, 3: forcing, 4: neural, 5: parameters, 6: global_params
    if length(positional_args) >= 1 && mechanistic === nothing
        mechanistic = positional_args[1]
    end
    if length(positional_args) >= 2 && targets === nothing
        targets = positional_args[2]
    end
    if length(positional_args) >= 3 && forcing === nothing
        forcing = positional_args[3]
    end
    if length(positional_args) >= 4 && neural === nothing
        neural = positional_args[4]
    end
    if length(positional_args) >= 5 && parameters === nothing
        parameters = positional_args[5]
    end
    if length(positional_args) >= 6 && global_params === nothing
        global_params = positional_args[6]
    end

    spec = HybridModelSpec(
        mechanistic,
        targets,
        forcing,
        neural,
        parameters,
        global_params,
        fixed_params,
        kwargs,
        source
    )
    _validate_hybrid_spec(spec)
    return spec
end

"""
    _validate_hybrid_spec(spec::HybridModelSpec)

Validate that essential fields are defined with clear, actionable error messages.
"""
function _validate_hybrid_spec(spec::HybridModelSpec)
    if spec.mechanistic === nothing
        error("$(spec.source): `@hybrid` requires a mechanistic model function (e.g. `mechanistic = my_model`).")
    end
    if spec.targets === nothing
        error("$(spec.source): `@hybrid` requires target variable(s) (e.g. `targets = :obs` or `targets = [:obs]`).")
    end
    if spec.parameters === nothing
        error("$(spec.source): `@hybrid` requires parameter definitions (e.g. `parameters = params_namedtuple`).")
    end
    return nothing
end

"""
    _emit_hybrid_model(spec::HybridModelSpec)

Generate hygiene-safe AST dispatching to `EasyHybrid.constructHybridModel`.
"""
function _emit_hybrid_model(spec::HybridModelSpec)
    kw_exprs = [Expr(:kw, k, esc(v)) for (k, v) in spec.kwargs]
    if spec.fixed_params !== nothing
        push!(kw_exprs, Expr(:kw, :fixed_param_names, esc(spec.fixed_params)))
    end

    return quote
        $EasyHybrid.constructHybridModel(
            $(esc(spec.mechanistic)),
            $(esc(spec.targets)),
            $(esc(spec.forcing)),
            $(esc(spec.neural)),
            $(esc(spec.parameters)),
            $(esc(spec.global_params));
            $(kw_exprs...)
        )
    end
end

"""
    @hybrid begin ... end
    @hybrid mechanistic_model targets=... forcing=... neural=... parameters=... [kwargs...]

Construct a `HybridModel` using a declarative domain-specific language (DSL).

### Supported Fields in Block Syntax:
- `mechanistic` / `model`: Process function `f(; forcing..., params...)`.
- `targets` / `target`: Target variable(s) (`Symbol` or `Vector{Symbol}`).
- `forcing` / `forcings`: Observational driver variables passed to mechanistic model.
- `neural` / `predictors`:
  - `Pair` (e.g. `[:in1, :in2] => [:p1, :p2]`): Single-NN architecture.
  - `NamedTuple` (e.g. `(p1 = [:in1], p2 = [:in2])`): Multi-NN architecture.
  - `nothing` / `Symbol[]`: Pure mechanistic model (zero NNs).
- `parameters` / `params`: `ParameterContainer` or `NamedTuple` of bounds.
- `global_params` / `globals` (optional): Variables globally optimized (auto-inferred if omitted).
- `fixed_params` / `fixed` (optional): Explicit fixed parameters.
- Additional configuration keywords: `hidden_layers`, `activation`, `scale_nn_outputs`, `input_batchnorm`, etc.

# Examples

### Block DSL:
```julia
model = @hybrid begin
    mechanistic = rs_comp
    targets = :R_soil
    forcing = [:Temp]
    neural = [:SWC, :TA] => [:Rb_het, :Rb_root, :Rb_myc]
    parameters = params_rs_comp
    hidden_layers = [16, 16]
    activation = tanh
    scale_nn_outputs = true
end
```

### Inline Syntax:
```julia
model = @hybrid rs_comp targets=:R_soil forcing=[:Temp] neural=([:SWC, :TA] => :Rb) parameters=params
```
"""
macro hybrid(args...)
    source = __source__
    if length(args) == 1 && args[1] isa Expr && args[1].head == :block
        spec = _parse_hybrid_block(args[1], source)
    else
        spec = _parse_hybrid_inline(args, source)
    end
    return _emit_hybrid_model(spec)
end
