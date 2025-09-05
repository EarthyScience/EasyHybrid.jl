"""
    EasyHybrid
    
EasyHybrid is a Julia package for hybrid machine learning models, combining neural networks and traditional statistical methods. It provides tools for data preprocessing, model training, and evaluation, making it easier to build and deploy hybrid models.
"""
module EasyHybrid
import LuxCore: LuxCore.setup, LuxCore.AbstractLuxContainerLayer
import Flux
using DataFrames
using DataFrameMacros
using Chain: @chain
using CSV
using MLUtils
using AxisKeys
using MLJ: partition
using Random
using LuxCore
using ChainRulesCore
using Zygote
using Optimisers
using Statistics: mean, cor
using ProgressMeter
using Random
using JLD2
using StyledStrings
using Printf
using Reexport: @reexport
using Hyperopt

@reexport begin
    import LuxCore
    using Lux
    using Lux: Dense, Chain, Dropout, relu
    using Random
    using Statistics
    using DataFrames
    using CSV
    using Optimisers
    using OptimizationOptimisers
    using ComponentArrays
end

include("macro_hybrid.jl")
include("utils/wrap_tuples.jl")
include("utils/io.jl")
include("utils/tools.jl")
include("models/models.jl")
include("utils/show_generic.jl")
include("utils/synthetic_test_data.jl")
include("utils/logging_loss.jl")
include("utils/loss_fn.jl")
include("train.jl")
include("utils/show_train.jl")
include("utils/helpers_for_HybridModel.jl")
include("plotrecipes.jl")
include("utils/helpers_data_loading.jl")
include("tune.jl")

end
