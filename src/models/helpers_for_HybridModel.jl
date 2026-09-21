export display_parameter_bounds, construct_dispatch_functions, build_parameter_matrix, build_parameter_scales

function construct_dispatch_functions(f)
    function new_f end  # Create a new generic function

    println("constructing on KeyedArray function for $f")
    function new_f(forcing_data::KeyedArray, parameters::NamedTuple, forcing_names::Vector{Symbol})
        forcing = toNamedTuple(forcing_data, forcing_names)
        parameter_container = ParameterContainer(parameters)
        return f(; forcing..., values(default(parameter_container))...)
    end

    function new_f(forcing_data::DataFrame, parameters::NamedTuple, forcing_names::Vector{Symbol})
        forcing = (; (name => forcing_data[!, name] for name in forcing_names)...)
        parameter_container = ParameterContainer(parameters)
        return f(; forcing..., values(default(parameter_container))...)
    end

    println("repeating kwargs style functions for $f (orignal function: $f)")
    function new_f(; kwargs...)
        return f(; kwargs...)
    end

    return new_f
end


"""
    build_parameter_matrix(parameter_defaults_and_bounds::NamedTuple)

Build a ComponentArray matrix from a NamedTuple containing parameter defaults and bounds.

This function converts a NamedTuple where each value is a tuple of (default, lower, upper) bounds
into a ComponentArray with named axes for easy parameter management in hybrid models.

# Arguments
- `parameter_defaults_and_bounds::NamedTuple`: A NamedTuple where each key is a parameter name and each value is a 
  tuple of (default, lower, upper) for that parameter.

# Returns
- `ComponentArray`: A 2D ComponentArray with:
  - Row axis: Parameter names (from the NamedTuple keys)
  - Column axis: Bound types (:default, :lower, :upper)
  - Data: The parameter values organized in a matrix format

# Example
```julia
# Define parameter defaults and bounds
parameter_defaults_and_bounds = (
    θ_s = (0.464f0, 0.302f0, 0.700f0),     # Saturated water content [cm³/cm³]
    h_r = (1500.0f0, 1500.0f0, 1500.0f0),  # Pressure head at residual water content [cm]
    α   = (log(0.103f0), log(0.01f0), log(7.874f0)),  # Shape parameter [cm⁻¹]
    n   = (log(3.163f0 - 1), log(1.100f0 - 1), log(20.000f0 - 1)),  # Shape parameter [-]
)

# Build the ComponentArray
parameter_matrix = build_parameter_matrix(parameter_defaults_and_bounds)

# Access specific parameter bounds
parameter_matrix.θ_s.default  # Get default value for θ_s
parameter_matrix[:, :lower]   # Get all lower bounds
parameter_matrix[:, :upper]   # Get all upper bounds
```

# Notes
- The function expects each value in the NamedTuple to be a tuple with exactly 3 elements
- The order of bounds is always (default, lower, upper)
- The resulting ComponentArray can be used for parameter optimization and constraint handling
"""
function build_parameter_matrix(parameter_defaults_and_bounds::NamedTuple)
    param_names = collect(keys(parameter_defaults_and_bounds))
    bound_names = (:default, :lower, :upper)
    data = [ parameter_defaults_and_bounds[p][i] for p in param_names, i in 1:length(bound_names) ]
    row_ax = ComponentArrays.Axis(param_names)
    col_ax = ComponentArrays.Axis(bound_names)
    return ComponentArray(data, row_ax, col_ax)
end

"The scaling warps supported by [`scale_single_param`](@ref)."
const SUPPORTED_PARAMETER_SCALES = (:linear, :log, :logit)

"""
    build_parameter_scales(parameter_defaults_and_bounds::NamedTuple)

Extract the per-parameter scaling warp from a parameter definition `NamedTuple`.

Each entry may carry an optional 4th element (`:linear`, `:log`, or `:logit`)
after `(default, lower, upper)`. Entries with only three elements default to
`:linear`. Returns a `NamedTuple` mapping each parameter name to its warp.
"""
function build_parameter_scales(parameter_defaults_and_bounds::NamedTuple)
    param_names = collect(keys(parameter_defaults_and_bounds))
    scales = map(param_names) do p
        spec = parameter_defaults_and_bounds[p]
        length(spec) >= 4 ? Symbol(spec[4]) : :linear
    end
    return NamedTuple{Tuple(param_names)}(Tuple(scales))
end

"""
    _validate_parameter_scales(values::NamedTuple, scales::NamedTuple)

Check that every parameter uses a supported warp with a domain-compatible
`(lower, upper)`: `:log` needs `lower > 0`, `:logit` needs `0 < lower < upper < 1`.
Throws an informative error otherwise.
"""
function _validate_parameter_scales(values::NamedTuple, scales::NamedTuple)
    for name in keys(values)
        spec = values[name]
        d, l, u = spec[1], spec[2], spec[3]
        s = scales[name]
        s in SUPPORTED_PARAMETER_SCALES ||
            error("Unknown scale `$s` for parameter `$name`; supported scales are $(SUPPORTED_PARAMETER_SCALES).")
        l <= u || error("Parameter `$name` has lower bound $l greater than upper bound $u.")
        (l <= d <= u) || error("Parameter `$name` default $d is outside its bounds [$l, $u].")
        if s === :log
            l > 0 || error("`:log`-scaled parameter `$name` requires a lower bound > 0 (got $l).")
        elseif s === :logit
            (0 < l && u < 1) ||
                error("`:logit`-scaled parameter `$name` requires 0 < lower < upper < 1 (got ($l, $u)).")
        end
    end
    return nothing
end
