#### Data handling
export select_predictors, to_keyedArray, split_data
export toDataFrame

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
    return KeyedArray(d, row=Symbol.(names(df)), col=1:size(d, 2))
end

# Cast a grouped dataframe into a KeyedArray, where the group is the third dimension
# Only one group dimension is currently considered 
"""
tokeyedArray(dfg::Union{Vector,GroupedDataFrame{DataFrame}}, vars=All())
"""
function to_keyedArray(dfg::Union{Vector,GroupedDataFrame{DataFrame}}, vars=All())
    dkg = [select(df, vars) |> tokeyedArray for df in dfg]
    dkg = reduce((x, y) -> cat(x, y, dims=3), dkg)
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
function split_data(df::DataFrame, target, xvars; f=0.8, batchsize=32, shuffle=true, partial=true)
    d_train, d_vali = partition(df, f; shuffle)
    # wrap training data into Flux.DataLoader
    # println(xvars)
    x = select(d_train, xvars) |> tokeyedArray
    y = select(d_train, target) |> Matrix |> transpose |> collect # tokeyedArray does not work bc of Zygote
    data_t = (; x, y)
    #println(size(y), size(data_t.x))
    trainloader = Flux.DataLoader(data_t; batchsize, shuffle, partial) # batches for training
    trainall = Flux.DataLoader(data_t; batchsize=size(y, 2), shuffle, partial) # whole training set for plotting
    # wrap validation data into Flux.DataLoader
    x = select(d_vali, xvars) |> tokeyedArray
    y = select(d_vali, target) |> Matrix |> transpose |> collect
    data_v = (; x, y)
    valloader = Flux.DataLoader(data_v; batchsize=size(y, 2), shuffle=false, partial=false) # whole validation for early stopping and plotting
    return trainloader, valloader, trainall
end

# As above but uses a seqID to keep same seqIDs in the same batch
# For instance needed for recurrent modelling
# Creates tensors with a third dimension, i.e. size is (nvar, seqLen, batchsize)
# Which is unfortunate since Recur in Flux wants sequence as last/3rd dimension
"""
split_data(df::DataFrame, target, xvars, seqID; f=0.8, batchsize=32, shuffle=true, partial=true)
"""
function split_data(df::DataFrame, target, xvars, seqID; f=0.8, batchsize=32, shuffle=true, partial=true)
    dfg = groupby(df, seqID)
    dkg = to_keyedArray(dfg)
    #@show axiskeys(dkg)[1]
    # Do the partitioning via indices of the 3rd dimension (e.g. seqID) because
    # partition does not allow partitioning along that dimension (or even not arrays at all)
    idx_tr, idx_vali = partition(axiskeys(dkg)[3], f; shuffle)
    # wrap training data into Flux.DataLoader
    x = dkg(row=xvars, seqID=idx_tr)
    y = dkg(row=target, seqID=idx_tr) |> Array
    data_t = (; x, y)
    trainloader = Flux.DataLoader(data_t; batchsize, shuffle, partial)
    trainall = Flux.DataLoader(data_t; batchsize=size(x, 3), shuffle=false, partial=false)
    # wrap validation data into Flux.DataLoader
    x = dkg(row=xvars, seqID=idx_vali)
    y = dkg(row=target, seqID=idx_vali) |> Array
    data_v = (; x, y)
    valloader = Flux.DataLoader(data_v; batchsize=size(x, 3), shuffle=false, partial=false)
    return trainloader, valloader, trainall
end

function toDataFrame(ka)
    data_array = Array(ka')
    df = DataFrame(data_array, ka.row)
    df.index = ka.col
    return df
end

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