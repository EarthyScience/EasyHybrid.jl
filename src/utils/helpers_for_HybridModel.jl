export display_parameter_bounds
export build_parameters

"""
    build_parameters(parameters::NamedTuple, f::DataType) -> AbstractHybridModel

Constructs a parameter container from a named tuple of parameter bounds and wraps it in a user-defined subtype of `AbstractHybridModel`.

# Arguments
- `parameters::NamedTuple`: Named tuple where each entry is a tuple of (default, lower, upper) bounds for a parameter.
- `f::DataType`: A constructor for a subtype of `AbstractHybridModel` that takes a `ParameterContainer` as its argument.

# Returns
- An instance of the user-defined `AbstractHybridModel` subtype containing the parameter container.
"""
function build_parameters(parameters::NamedTuple, f::DataType)
    ca = EasyHybrid.ParameterContainer(parameters)
    return f(ca)
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
    param_names     = collect(keys(parameter_defaults_and_bounds))
    bound_names = (:default, :lower, :upper)
    data = [ parameter_defaults_and_bounds[p][i] for p in param_names, i in 1:length(bound_names) ]
    row_ax = ComponentArrays.Axis(param_names)
    col_ax = ComponentArrays.Axis(bound_names)
    return ComponentArray(data, row_ax, col_ax)
end

"""
    Base.display(io::IO, parameter_container::ParameterContainer)

Display a ParameterContainer containing parameter bounds in a formatted table.

This function creates a nicely formatted table showing parameter names as row labels
and bound types (default, lower, upper) as column headers.

# Arguments
- `io::IO`: Output stream
- `parameter_container::ParameterContainer`: A ParameterContainer with parameter bounds (typically created by `build_parameter_matrix`)

# Returns
- Displays a formatted table using PrettyTables.jl

# Example
```julia
# Create parameter defaults and bounds
parameter_defaults_and_bounds = (
    θ_s = (0.464f0, 0.302f0, 0.700f0),
    α   = (log(0.103f0), log(0.01f0), log(7.874f0)),
    n   = (log(3.163f0 - 1), log(1.100f0 - 1), log(20.000f0 - 1)),
)

# Build ParameterContainer and display
parameter_container = ParameterContainer(parameter_defaults_and_bounds)
display(parameter_container)  # or just parameter_container
```

# Notes
- Requires PrettyTables.jl to be loaded
- The table shows parameter names as row labels and bound types as column headers
- Default alignment is right-aligned for all columns
"""
function Base.display(parameter_container::T) where T <: AbstractHybridModel
    display_parameter_bounds(parameter_container)
end

"""
    display_parameter_bounds(parameter_container::ParameterContainer; alignment=:r)

Legacy function for displaying parameter bounds with custom alignment.

# Arguments
- `parameter_container::ParameterContainer`: A ParameterContainer with parameter bounds
- `alignment`: Alignment for table columns (default: right-aligned for all columns)

# Returns
- Displays a formatted table using PrettyTables.jl
"""
function display_parameter_bounds(parameter_container::T; alignment=:r) where {T <: AbstractHybridModel}
    table = parameter_container.hybrid.table
    PrettyTables.pretty_table(
        table;  
        header     = collect(keys(table.axes[2])),
        row_labels = collect(keys(table.axes[1])),
        alignment  = alignment
    )
end



function Base.display(hm::HybridModel)
    display(hm.NN)
    
    println("global parameters: ", hm.global_param_names)
    println("neural parameters: ", hm.neural_param_names)
    println("fixed parameters: ", hm.fixed_param_names)

    println("parameter defaults and bounds:")
    display(hm.parameters)

end