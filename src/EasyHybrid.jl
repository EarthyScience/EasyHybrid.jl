"""
    EasyHybrid
    
EasyHybrid is a Julia package for hybrid machine learning models, combining neural networks and traditional statistical methods. It provides tools for data preprocessing, model training, and evaluation, making it easier to build and deploy hybrid models.
"""
module EasyHybrid

using AxisKeys: AxisKeys, KeyedArray, axiskeys, wrapdims
using CSV: CSV
using Chain: @chain
using ChainRulesCore: ChainRulesCore
using ComponentArrays: ComponentArrays, ComponentArray
using DataFrameMacros: DataFrameMacros, @transform
using DataFrames: DataFrames, DataFrame, GroupedDataFrame, Missing, coalesce, mapcols, select, missing, All
using DimensionalData: DimensionalData, AbstractDimArray, At, dims, groupby
using Downloads: Downloads
using Hyperopt: Hyperopt, Hyperoptimizer
using JLD2: JLD2, jldopen
using LuxCore: LuxCore
using Lux: Lux
using MLJ: partition
using MLUtils: MLUtils, DataLoader, kfolds, numobs, rpad, splitobs
using NCDatasets: NCDatasets, NCDataset, close, name
using OptimizationOptimisers: OptimizationOptimisers, AdamW, Adam, Optimisers
using PrettyTables: PrettyTables
using Printf: Printf, @sprintf
using ProgressMeter: ProgressMeter, Progress, next!
using Random: Random, AbstractRNG, randperm, randstring
using Reexport: @reexport
using Statistics: Statistics, mean, cor, quantile, var
using StyledStrings: StyledStrings, @styled_str
using Zygote: Zygote
using Static: False, True

@reexport begin
    import LuxCore
    using Lux: Lux, Dense, Chain, Dropout, relu, sigmoid, swish
    using Random
    using Statistics
    using DataFrames
    using CSV
    using OptimizationOptimisers: OptimizationOptimisers, Optimisers, Adam, AdamW, RMSProp
    using ComponentArrays: ComponentArrays, ComponentArray
end

include("macro_hybrid.jl")
include("utils/wrap_tuples.jl")
include("utils/io.jl")
include("utils/tools.jl")
include("models/models.jl")
include("utils/show_generic.jl")
include("utils/synthetic_test_data.jl")
include("utils/compute_loss_types.jl")
include("utils/show_loss_types.jl")
include("utils/compute_loss.jl")
include("utils/loss_fn.jl")
include("plotrecipes.jl")
include("train.jl")
include("utils/show_train.jl")
include("utils/helpers_for_HybridModel.jl")
include("utils/helpers_data_loading.jl")
include("tune.jl")
include("utils/helpers_cross_validation.jl")

end
