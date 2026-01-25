#### Data handling
export select_predictors, to_keyedArray, to_dimArray, split_data
export toDataFrame, toNamedTuple, toArray

# Make vec each entry of NamedTuple (since broadcast ist reserved)
"""
evec(nt::NamedTuple)
"""
function evec(nt::NamedTuple)
    return map(vec, nt)
end

# Start from Dataframe, select variables and make a Flux-compatible tensor
"""
select_predictors(df, predictors)
"""
function select_predictors(df, predictors)
    return select(df, predictors) |> Matrix |> transpose
end
# Start from KeyedArray, selct variables and make a Flux-compatible tensor
"""
select_predictors(dk::KeyedArray, predictors)
"""
function select_predictors(dk::KeyedArray, predictors)
    return dk(predictors) |> Array
end

"""
select_cols(df::KeyedArray, predictors, x)
"""
function select_cols(dk::KeyedArray, predictors, x)
    return dk([predictors..., x])
end

"""
select_variable(df::KeyedArray, x)
"""
function select_variable(dk::KeyedArray, x)
    return dk(x) |> Vector
end

"""
select_cols(df, predictors, x)
"""
function select_cols(df, predictors, x)
    return select(df, [predictors..., x])
end

# Convert a DataFrame to a Keyedarray where variables are in 1st dim (rows)
"""
tokeyedArray(df::DataFrame)
"""
function to_keyedArray(df::DataFrame)
    d = Matrix(df) |> transpose
    return KeyedArray(d, variable = Symbol.(names(df)), batch_size = 1:size(d, 2))
end

# Convert a DataFrame to a DimArray where variables are in 1st dim (rows)
"""
to_dimArray(df::DataFrame)
"""
function to_dimArray(df::DataFrame)
    d = Matrix(df) |> transpose |> Array
    return DimArray(d, (Dim{:variable}(Symbol.(names(df))), Dim{:batch_size}(1:size(d, 2))))
end

# Cast a grouped dataframe into a KeyedArray, where the group is the third dimension
# Only one group dimension is currently considered
"""
tokeyedArray(dfg::Union{Vector,GroupedDataFrame{DataFrame}}, vars=All())
"""
function to_keyedArray(dfg::Union{Vector, GroupedDataFrame{DataFrame}}, vars = All())
    dkg = [select(df, vars) |> tokeyedArray for df in dfg]
    dkg = reduce((x, y) -> cat(x, y, dims = 3), dkg)
    newKeyNames = (AxisKeys.dimnames(dkg)[1:2]..., dfg.cols[1])
    newKeys = (axiskeys(dkg)[1:2]..., unique(dfg.groups))
    return (wrapdims(dkg |> Array; (; zip(newKeyNames, newKeys)...)...))
end

# Create dataloaders for training and validation
# Splits a normal dataframe into train/val and creates minibatches of x and y,
# where x is a KeyedArray and y a normal one (need to recheck why KeyedArray did not work with Zygote)
"""
split_data(df::DataFrame, target, xvars; f=0.8, batchsize=32, shuffle=true, partial=true)
"""
function split_data(df::DataFrame, target, xvars; f = 0.8, batchsize = 32, shuffle = true, partial = true)
    d_train, d_vali = partition(df, f; shuffle)
    # wrap training data into Flux.DataLoader
    # println(xvars)
    x = select(d_train, xvars) |> tokeyedArray
    y = select(d_train, target) |> Matrix |> transpose |> collect # tokeyedArray does not work bc of Zygote
    data_t = (; x, y)
    #println(size(y), size(data_t.x))
    trainloader = Flux.DataLoader(data_t; batchsize, shuffle, partial) # batches for training
    trainall = Flux.DataLoader(data_t; batchsize = size(y, 2), shuffle, partial) # whole training set for plotting
    # wrap validation data into Flux.DataLoader
    x = select(d_vali, xvars) |> tokeyedArray
    y = select(d_vali, target) |> Matrix |> transpose |> collect
    data_v = (; x, y)
    valloader = Flux.DataLoader(data_v; batchsize = size(y, 2), shuffle = false, partial = false) # whole validation for early stopping and plotting
    return trainloader, valloader, trainall
end

# As above but uses a seqID to keep same seqIDs in the same batch
# For instance needed for recurrent modelling
# Creates tensors with a third dimension, i.e. size is (nvar, seqLen, batchsize)
# Which is unfortunate since Recur in Flux wants sequence as last/3rd dimension
"""
split_data(df::DataFrame, target, xvars, seqID; f=0.8, batchsize=32, shuffle=true, partial=true)
"""
function split_data(df::DataFrame, target, xvars, seqID; f = 0.8, batchsize = 32, shuffle = true, partial = true)
    dfg = groupby(df, seqID)
    dkg = to_keyedArray(dfg)
    # Do the partitioning via indices of the 3rd dimension (e.g. seqID) because
    # partition does not allow partitioning along that dimension (or even not arrays at all)
    idx_tr, idx_vali = partition(axiskeys(dkg)[3], f; shuffle)
    # wrap training data into Flux.DataLoader
    x = dkg(variable = xvars, seqID = idx_tr)
    y = dkg(variable = target, seqID = idx_tr) |> Array
    data_t = (; x, y)
    trainloader = Flux.DataLoader(data_t; batchsize, shuffle, partial)
    trainall = Flux.DataLoader(data_t; batchsize = size(x, 3), shuffle = false, partial = false)
    # wrap validation data into Flux.DataLoader
    x = dkg(variable = xvars, seqID = idx_vali)
    y = dkg(variable = target, seqID = idx_vali) |> Array
    data_v = (; x, y)
    valloader = Flux.DataLoader(data_v; batchsize = size(x, 3), shuffle = false, partial = false)
    return trainloader, valloader, trainall
end

using AxisKeys
using NamedDims: NamedDims  # Required for NamedDims.dim with KeyedArrays
using DataFrames
using DimensionalData: DimensionalData, AbstractDimArray, Dim, DimArray, dims, lookup, At

_key_to_colname(k) = k isa Symbol ? k : Symbol(string(k))

# Helper to get dimension index from dimension name (works for both KeyedArray and DimArray)
_dim_index(ka::KeyedArray, name::Symbol) = NamedDims.dim(ka, name)
function _dim_index(da::AbstractDimArray, name::Symbol)
    dim_names = DimensionalData.name.(dims(da))
    idx = findfirst(==(name), dim_names)
    isnothing(idx) && throw(ArgumentError("Dimension :$name not found in array with dimensions $dim_names"))
    return idx
end

# Helper to extract raw array data (works for both KeyedArray and DimArray)
_raw_array(ka::KeyedArray) = Array(AxisKeys.keyless(ka))
_raw_array(da::AbstractDimArray) = Array(parent(da))

# Helper to select a single value along a named dimension
_select_at(ka::KeyedArray, dim_name::Symbol, key) = ka(; NamedTuple{(dim_name,)}((key,))...)
_select_at(da::AbstractDimArray, dim_name::Symbol, key) = view(da, Dim{dim_name}(At(key)))

# 2D Labeled Array -> DataFrame (works for both KeyedArray and DimArray)
"""
    toDataFrame(arr::Union{KeyedArray{T, 2}, AbstractDimArray{T, 2}}, cols_dim=:variable, index_dim=:batch_size; index_col=:index)

Convert a 2D labeled array (KeyedArray or DimArray) to a DataFrame.

# Arguments
- `arr`: The 2D labeled array to convert
- `cols_dim`: Dimension name to use as DataFrame columns (default: `:variable`)
- `index_dim`: Dimension name to use as DataFrame row index (default: `:batch_size`)
- `index_col`: Name for the index column in the result (default: `:index`)

# Returns
- `DataFrame` with columns from `cols_dim` keys and an index column from `index_dim` keys
"""
function toDataFrame(
        arr::Union{KeyedArray{T, 2}, AbstractDimArray{T, 2}},
        cols_dim::Symbol = :variable,
        index_dim::Symbol = :batch_size;
        index_col::Symbol = :index,
    ) where {T}

    dcols = _dim_index(arr, cols_dim)
    didx = _dim_index(arr, index_dim)

    # Reorder so rows=index_dim, cols=cols_dim (i.e., didx=1, dcols=2)
    arr2 = (didx == 1 && dcols == 2) ? arr : permutedims(arr, (didx, dcols))

    data = _raw_array(arr2)
    col_names = _key_to_colname.(collect(axiskeys(arr2, 2)))

    df = DataFrame(data, col_names; makeunique = true)
    df[!, index_col] = collect(axiskeys(arr2, 1))
    return df
end

# 3D Labeled Array -> Dict(slice_key => DataFrame)
"""
    toDataFrame(arr::AbstractLabeledArray{T, 3}, cols_dim=:variable, index_dim=:batch_size; slice_dim=:time, index_col=:index)

Convert a 3D labeled array (KeyedArray or DimArray) to a Dict of DataFrames, one per slice.

# Arguments
- `arr`: The 3D labeled array to convert
- `cols_dim`: Dimension name to use as DataFrame columns (default: `:variable`)
- `index_dim`: Dimension name to use as DataFrame row index (default: `:batch_size`)
- `slice_dim`: Dimension name to slice along (default: `:time`)
- `index_col`: Name for the index column in each result DataFrame (default: `:index`)

# Returns
- `Dict{Any, DataFrame}` mapping slice keys to DataFrames
"""
function toDataFrame(
        arr::Union{KeyedArray{T, 3}, AbstractDimArray{T, 3}},
        cols_dim::Symbol = :variable,
        index_dim::Symbol = :batch_size;
        slice_dim::Symbol = :time,
        index_col::Symbol = :index,
    ) where {T}

    out = Dict{Any, DataFrame}()
    for k in axiskeys(arr, slice_dim)
        slice = _select_at(arr, slice_dim, k)
        out[k] = toDataFrame(slice, cols_dim, index_dim; index_col = index_col)
    end
    return out
end

# Convenience: extract specific targets from a labeled array into a DataFrame
"""
    toDataFrame(arr, target_names)

Extract specific target variables from a labeled array into a DataFrame with `_pred` suffix.

# Arguments
- `arr`: A labeled array or NamedTuple-like object with property access
- `target_names`: Vector of target variable names to extract

# Returns
- `DataFrame` with columns named `<target>_pred` for each target
"""
function toDataFrame(ka, target_names)
    data = [getproperty(ka, t_name) for t_name in target_names]

    if length(target_names) == 1
        # For single target, convert to vector and create DataFrame with one column
        data_vector = vec(vec(data...))
        return DataFrame(string(target_names[1]) * "_pred" => data_vector)
    else
        # For multiple targets, create DataFrame with multiple columns
        return DataFrame(data, string.(target_names) .* "_pred")
    end
end

# =============================================================================
# Array unpacking functions (works for both KeyedArray and DimArray)
# =============================================================================

"""
    toNamedTuple(ka::Union{KeyedArray, AbstractDimArray}, variables::Vector{Symbol})

Extract specified variables from a KeyedArray or DimArray and return them as a NamedTuple of vectors.

# Arguments:
- `ka`: The KeyedArray or DimArray to unpack
- `variables`: Vector of symbols representing the variables to extract

# Returns:
- NamedTuple with variable names as keys and vectors as values

# Example:
```julia
# Extract SW_IN and TA from an array
data = toNamedTuple(ds, [:SW_IN, :TA])
sw_in = data.SW_IN
ta = data.TA
```
"""
function toNamedTuple(ka::KeyedArray, variables::Vector{Symbol})
    vals = [dropdims(ka(variable = [var]), dims = 1) for var in variables]
    return (; zip(variables, vals)...)
end

function toNamedTuple(ka::AbstractDimArray, variables::Vector{Symbol})
    vals = [dropdims(ka[variable = At([var])], dims = 1) for var in variables]
    return (; zip(variables, vals)...)
end

function toNamedTuple(ka::KeyedArray, variables::NTuple{N, Symbol}) where {N}
    vals = ntuple(i -> ka(variable = [variables[i]]), N)
    return NamedTuple{variables}(vals)
end

function toNamedTuple(ka::AbstractDimArray, variables::NTuple{N, Symbol}) where {N}
    ntuple(i -> ka[variable = At([variables[i]])], N)
    return NamedTuple{variables}(vals)
end

"""
toNamedTuple(ka::KeyedArray)
Extract all variables from a KeyedArray and return them as a NamedTuple of vectors.

# Arguments:
- `ka`: The KeyedArray to unpack

# Returns:
- NamedTuple with all variable names as keys and vectors as values

# Example:
```julia
# Extract all variables from an array
data = toNamedTuple(ds)
# Access individual variables
sw_in = data.SW_IN
ta = data.TA
nee = data.NEE
```
"""
function toNamedTuple(ka::KeyedArray)
    variables = Symbol.(axiskeys(ka, :variable))  # Get all variable names from :variable dimension
    return toNamedTuple(ka, variables)
end

function toNamedTuple(ka::AbstractDimArray)
    variables = Symbol.(lookup(ka, :variable))  # Get all variable names from :variable dimension
    return toNamedTuple(ka, variables)
end

"""
toNamedTuple(ka::KeyedArray, variable::Symbol)
Extract a single variable from a KeyedArray and return it as a vector.

# Arguments:
- `ka`: The KeyedArray or DimArray to unpack
- `variable`: Symbol representing the variable to extract

# Returns:
- Vector containing the variable data

# Example:
```julia
# Extract just SW_IN from an array
sw_in = toNamedTuple(ds, :SW_IN)
```
"""
function toNamedTuple(ka::KeyedArray, variable::Symbol)
    return ka(variable = variable)
end

function toNamedTuple(ka::AbstractDimArray, variable::Symbol)
    return ka[variable = At(variable)]
end

function toArray(ka::KeyedArray, variable)
    return ka(variable = variable)
end

function toArray(ka::AbstractDimArray, variable)
    return ka[variable = At(variable)]
end
