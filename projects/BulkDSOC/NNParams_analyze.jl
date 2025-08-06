using Pkg
Pkg.activate("projects/BulkDSOC")
Pkg.develop(path=pwd())
Pkg.instantiate()

using AxisKeys
using Revise
using EasyHybrid
using Lux
using Optimisers
using GLMakie
using Random
using LuxCore
using CSV, DataFrames
using EasyHybrid.MLUtils
using Statistics
using Plots
using Flux
using NNlib

target_names = [:BD, :CF, :SOCconc, :SOCdensity]; #
# check stats function
clean_r2(x) = mean(y for y in x if !ismissing(y) && !isfinite(y))
std_r2(x)   = std(y for y in x if !ismissing(y) && !isfinite(y))
count_bad_r2(x) = count(y -> !ismissing(y) && !isfinite(y), x)

top_n = 20;

top_params_per_target = Dict{Symbol, DataFrame}()

for tgt in target_names
    # read and clean
    df = CSV.read(joinpath(@__DIR__, "./eval/parameter_search_$(string(tgt)).csv"),DataFrame, normalizenames=true)
    df = filter(:r2 => x -> isfinite(x), df)
    df = sort(df, :r2, rev=true);
    print(first(df, 20))
    
    # # check stats
    # sta = combine(groupby(df, [:hidden_layers]),
    # :r2 => clean_r2 => :r2_mean,
    # :r2 => std_r2   => :r2_std,
    # :r2 => count_bad_r2 => :num_bad);
    # sort(sta, :r2_mean, rev=true)

    # # check r2 distribution
    # df.index = 1:nrow(df);
    # Plots.plot(df.index, dff.r2,
    # xlabel = "Configuration Rank",
    # ylabel = "R2",
    # title = tgt,
    # legend = false,
    # marker = :circle,
    # line = :solid)

    # # best n combinations
    # top_params_per_target[tgt] = first(df, top_n)[!, [:hidden_layers, :batch_size, :learning_rate, :activation]]
end

# # to NamedTuples for easy intersection
# param_sets = [Set(eachrow(df)) for df in values(top_params_per_target)];
# # shared ones
# common_params = reduce(intersect, param_sets);
# foreach(p -> println(p), common_params)



