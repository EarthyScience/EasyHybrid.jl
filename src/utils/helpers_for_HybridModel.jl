export build_cm
export display_parameter_bounds

"""
    build_cm(values::NamedTuple)

Build a ComponentArray matrix from a NamedTuple containing parameter bounds.

This function converts a NamedTuple where each value is a tuple of (default, lower, upper) bounds
into a ComponentArray with named axes for easy parameter management in hybrid models.

# Arguments
- `values::NamedTuple`: A NamedTuple where each key is a parameter name and each value is a 
  tuple of (default, lower, upper) bounds for that parameter.

# Returns
- `ComponentArray`: A 2D ComponentArray with:
  - Row axis: Parameter names (from the NamedTuple keys)
  - Column axis: Bound types (:default, :lower, :upper)
  - Data: The parameter values organized in a matrix format

# Example
```julia
# Define parameter bounds
values = (
    θ_s = (0.464f0, 0.302f0, 0.700f0),     # Saturated water content [cm³/cm³]
    h_r = (1500.0f0, 1500.0f0, 1500.0f0),  # Pressure head at residual water content [cm]
    α   = (log(0.103f0), log(0.01f0), log(7.874f0)),  # Shape parameter [cm⁻¹]
    n   = (log(3.163f0 - 1), log(1.100f0 - 1), log(20.000f0 - 1)),  # Shape parameter [-]
)

# Build the ComponentArray
cm = build_cm(values)

# Access specific parameter bounds
cm.θ_s.default  # Get default value for θ_s
cm[:, :lower]   # Get all lower bounds
cm[:, :upper]   # Get all upper bounds
```

# Notes
- The function expects each value in the NamedTuple to be a tuple with exactly 3 elements
- The order of bounds is always (default, lower, upper)
- The resulting ComponentArray can be used for parameter optimization and constraint handling
"""
function build_cm(values::NamedTuple)
    param_names     = collect(keys(values))
    bound_names = (:default, :lower, :upper)
    data = [ values[p][i] for p in param_names, i in 1:length(bound_names) ]
    row_ax = ComponentArrays.Axis(param_names)
    col_ax = ComponentArrays.Axis(bound_names)
    return ComponentArray(data, row_ax, col_ax)
end

"""
    display_parameter_bounds(ca::ComponentArray; alignment=:r)

Display a ComponentArray containing parameter bounds in a formatted table.

This function creates a nicely formatted table showing parameter names as row labels
and bound types (default, lower, upper) as column headers.

# Arguments
- `ca::ComponentArray`: A ComponentArray with parameter bounds (typically created by `build_cm`)
- `alignment`: Alignment for table columns (default: right-aligned for all columns)

# Returns
- Displays a formatted table using PrettyTables.jl

# Example
```julia
# Create parameter bounds
values = (
    θ_s = (0.464f0, 0.302f0, 0.700f0),
    α   = (log(0.103f0), log(0.01f0), log(7.874f0)),
    n   = (log(3.163f0 - 1), log(1.100f0 - 1), log(20.000f0 - 1)),
)

# Build ComponentArray and display
cm = build_cm(values)
display_parameter_bounds(cm)
```

# Notes
- Requires PrettyTables.jl to be loaded
- The table shows parameter names as row labels and bound types as column headers
- Default alignment is right-aligned for all columns, but can be customized
"""
function display_parameter_bounds(ca::ComponentArray, alignment=:r)
    PrettyTables.pretty_table(
        ca;
        header     = collect(keys(ca.axes[2])),
        row_labels = collect(keys(ca.axes[1])),
        alignment  = alignment
    )
end