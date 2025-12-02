export make_folds

"""
    make_folds(df::DataFrame; k::Int=5, shuffle=true) -> Vector{Int}

Assigns each observation in the DataFrame `df` to one of `k` folds for cross-validation.

# Arguments
- `df::DataFrame`: The input DataFrame whose rows are to be split into folds.
- `k::Int=5`: Number of folds to create.
- `shuffle=true`: Whether to shuffle the data before assigning folds.

# Returns
- `folds::Vector{Int}`: A vector of length `nrow(df)` where each entry is an integer in `1:k` indicating the fold assignment for that observation.
"""
function make_folds(df::DataFrame; k::Int = 5, shuffle = true)
    n = numobs(df)
    _, val_idx = kfolds(n, k)
    folds = fill(0, n)
    perm = shuffle ? randperm(n) : 1:n
    for (f, idx) in enumerate(val_idx)
        fidx = perm[idx]
        folds[fidx] .= f
    end
    return folds
end
