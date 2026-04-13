export prepare_data

function prepare_data(hm, data::KeyedArray; cfg = DataConfig(), kwargs...)
    predictors, forcings, targets = get_prediction_target_names(hm)
    # KeyedArray: use () syntax for views that are differentiable
    X_arr = Array(data(predictors))
    forcings_nt = NamedTuple([forcing => Array(data(forcing)) for forcing in forcings])
    targets_nt = NamedTuple([target => Array(data(target)) for target in targets])
    return ((X_arr, forcings_nt), targets_nt)
end

function prepare_data(hm::MultiNNHybridModel, data::KeyedArray; cfg = DataConfig(), kwargs...)
    predictors, forcings, targets = get_prediction_target_names(hm)
    # KeyedArray: use () syntax for views that are differentiable
    X_all = NamedTuple([name => Array(data(p)) for (name, p) in pairs(predictors)])
    forcings_nt = NamedTuple([forcing => Array(data(forcing)) for forcing in forcings])
    targets_nt = NamedTuple([target => Array(data(target)) for target in targets])
    return ((X_all, forcings_nt), targets_nt)
end

function prepare_data(hm, data::AbstractDimArray; kwargs...)
    predictors, forcings, targets = get_prediction_target_names(hm)
    # KeyedArray: use () syntax for views that are differentiable
    X_arr = data[variable = At(predictors)]
    forcings_nt = NamedTuple([forcing => data[variable = At(forcing)] for forcing in forcings])
    targets_nt = NamedTuple([target => data[variable = At(target)] for target in targets])
    # DimArray: use [] syntax (copies, but differentiable)
    return ((X_arr, forcings_nt), targets_nt)
end

function prepare_data(hm, data::DataFrame; array_type = :KeyedArray, drop_missing_rows = true)
    predictors, forcings, targets = get_prediction_target_names(hm)

    # all_predictor_cols = unique(vcat(values(predictors_forcing)...))
    col_to_select = unique([vcat(predictors...); forcings; targets])

    # subset to only the cols we care about
    sdf = data[!, col_to_select]

    mapcols(col -> replace!(col, missing => NaN), sdf; cols = names(sdf, Union{Missing, Real}))

    if drop_missing_rows
        # Separate predictor/forcing vs. target columns
        predforce_cols = setdiff(col_to_select, targets)

        # For each row, check if *any* predictor/forcing is missing
        mask_missing_predforce = map(row -> any(isnan, row), eachrow(sdf[:, predforce_cols]))

        # For each row, check if *at least one* target is present (i.e. not all missing)
        mask_at_least_one_target = map(row -> any(!isnan, row), eachrow(sdf[:, targets]))

        # Keep rows where predictors/forcings are *complete* AND there's some target present
        keep = .!mask_missing_predforce .& mask_at_least_one_target
        sdf = sdf[keep, col_to_select]
    end

    if array_type == :KeyedArray
        ds = to_keyedArray(Float32.(sdf))
    else
        ds = to_dimArray(Float32.(sdf))
    end
    return prepare_data(hm, ds)
end

function prepare_data(hm, data::Tuple; kwargs...)
    return data
end

"""
    prepare_data(hm, data::DataFrame; array_type=:KeyedArray, drop_missing_rows=true)
    prepare_data(hm, data::KeyedArray)
    prepare_data(hm, data::AbstractDimArray)
    prepare_data(hm, data::Tuple)

Prepare data for training by extracting predictor/forcing and target variables based on the hybrid model's configuration.

# Arguments:
- `hm`: The Hybrid Model
- `data`: The input data, which can be a DataFrame, KeyedArray, or DimensionalData array.
- `array_type`: (DataFrame only) Output array type: `:KeyedArray` (default) or `:DimArray`.
- `drop_missing_rows`: (DataFrame only) If `true` (default), drop rows where any predictor is NaN or all targets are NaN.

# Returns:
- If `data` is a DataFrame: a tuple of (predictors_forcing, targets) as KeyedArrays or DimArrays depending on `array_type`.
- If `data` is a KeyedArray: a tuple of (predictors_forcing, targets) as KeyedArrays.
- If `data` is an AbstractDimArray: a tuple of (predictors_forcing, targets) as DimArrays.
- If `data` is already a Tuple, it is returned as-is.
"""
function prepare_data end

"""
    get_prediction_target_names(hm)
Utility function to extract predictor/forcing and target names from a hybrid model.

# Arguments:
- `hm`: The Hybrid Model

Returns a tuple of (predictors_forcing, targets) names.
"""
function get_prediction_target_names(hm)
    targets = hm.targets
    predictors = hm.predictors
    forcings = hm.forcing

    if isempty(predictors)
        @warn "Note that you don't have predictors variables."
    end
    if isempty(forcings)
        @warn "Note that you don't have forcing variables."
    end
    if isempty(targets)
        @warn "Note that you don't have target names."
    end
    return predictors, forcings, targets
end
