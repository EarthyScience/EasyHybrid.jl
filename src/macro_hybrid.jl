export @hybrid

"""
    @hybrid ModelName α β γ

Macro to define hybrid model structs with arbitrary numbers of physical parameters.

This defines a struct with:
- Default fields: `NN` (neural network), `predictors`, `forcing`, `targets`.
- Additional physical parameters, i.e., `α β γ`.

# Examples
```julia
@hybrid MyModel α β γ
@hybrid FluidModel (:viscosity, :density)
@hybrid SimpleModel :a :b
```
"""
macro hybrid(name, params...)
    # Handle both `:a :b :c` and `(:a, :b, :c)` forms
    param_exprs = length(params) == 1 && params[1] isa Expr && params[1].head == :tuple ?
        params[1].args : collect(params)
    
    # Normalize/Validate to symbols
    param_syms = [p isa Symbol ? p :
        p isa QuoteNode && p.value isa Symbol ? p.value :
        error("Parameter names must be Symbols, got: $p (type: $(typeof(p)))") for p in param_exprs]
    
    # Type parameters: D for NN, T1–T3 for standard fields, T4+ for custom params
    standard_types = [:D, :T1, :T2, :T3]
    param_types = [Symbol("T$(i + 3)") for i in 1:length(param_syms)]
    all_types = [standard_types; param_types]
    
    # Field names
    standard_fields = [:NN, :predictors, :forcing, :targets]
    all_fields = [standard_fields; param_syms]
    
    # Tuple of quoted field names for supertype
    field_tuple = Expr(:tuple, [QuoteNode(field) for field in all_fields]...)
    
    # Struct field declarations (untyped)
    struct_fields = [field for field in all_fields]
    
    # Constructor parameter list with types
    standard_constructor_params = [
        Expr(:(::), :NN, :D),
        Expr(:(::), :predictors, :T1),
        Expr(:(::), :forcing, :T2),
        Expr(:(::), :targets, :T3)
    ]
    param_constructor_params = [Expr(:(::), param, param_types[i]) for (i, param) in enumerate(param_syms)]
    all_constructor_params = [standard_constructor_params; param_constructor_params]
    # Create named tuple for trainable parameters (physical parameters)
    trainable_params = if isempty(param_syms)
        # empty named tuple - need to splice nothing
        []
    else
        # Create individual parameter assignments for splatting
        [Expr(:(=), param, :(layer.$param)) for param in param_syms]
    end

    docstring = """
            $(name)(NN, predictors, forcing, targets$(isempty(param_syms) ? "" : ", " * join(string.(param_syms), ", ")))
        
        A hybrid model with:
        - `NN`: a neural network
        - `predictors`: a tuple of names, i.e, (:clay, :moist)
        - `forcing`: data names, i.e. (:temp, )
        - `targets`: data names, i.e. (:ndvi, )
        $(isempty(param_syms) ? "" : "- Physical parameters: " * join(string.(param_syms), ", "))
        
        All physical parameters are trainable by default. Use `?Lux.Experimental.freeze` to make specific parameters non-trainable during training.
        """
    # Build the complete struct definition
    complete_def = quote
        @doc $docstring
        struct $(name){$(all_types...)} <: LuxCore.AbstractLuxContainerLayer{$(field_tuple)}
            $(struct_fields...)
            
            function $(name)($(all_constructor_params...)) where {$(all_types...)}
                new{$(all_types...)}(
                    NN,
                    collect(predictors),
                    collect(forcing),
                    collect(targets),
                    $(param_syms...)
                )
            end
        end
        # Define initialparameters method
        function LuxCore.initialparameters(rng::AbstractRNG, layer::$(name))
            ps, _ = LuxCore.setup(rng, layer.NN)
            return (; ps, $(trainable_params...))
        end
        
        # Define initialstates method  
        function LuxCore.initialstates(rng::AbstractRNG, layer::$(name))
            _, st = LuxCore.setup(rng, layer.NN)
            return (; st)
        end
    end
    
    return esc(complete_def)
end