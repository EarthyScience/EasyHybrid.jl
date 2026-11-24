using EasyHybrid
using Test
using Random
using DimensionalData: DimArray, Dim
using DiskArrays: DiskArrays
using DiskArrays.TestTypes: AccessCountDiskArray, getindex_log
using MLUtils: numobs, getobs

# Synthetic Q10 respiration data stored as a (variable × observation) matrix, the
# layout used for lazy training (e.g. a YAXArray opened from Zarr).
function lazy_q10_matrix(n; rng = Random.Xoshiro(7))
    ta = 20.0f0 .* rand(rng, Float32, n) .- 5.0f0
    sw_pot = 800.0f0 .* rand(rng, Float32, n)
    dsw_pot = randn(rng, Float32, n)
    rb = 1.0f0 .+ 0.004f0 .* sw_pot
    reco = rb .* 1.5f0 .^ (0.1f0 .* (ta .- 15.0f0))
    site = Float32.(repeat(1:10, inner = cld(n, 10))[1:n])
    obsid = Float32.(1:n)
    reco[5] = NaN32                # unobserved target -> dropped
    sw_pot[17] = NaN32             # missing predictor -> dropped
    return permutedims(hcat(ta, sw_pot, dsw_pot, reco, site, obsid))
end

const LAZY_VARS = ["ta", "sw_pot", "dsw_pot", "reco", "site", "obsid"]   # Strings, as stored in Zarr

function RbQ10_lazy(; ta, Q10, rb, tref = 15.0f0)
    reco = rb .* Q10 .^ (0.1f0 .* (ta .- tref))
    return (; reco, Q10, rb)
end

lazy_q10_model() = constructHybridModel(
    [:sw_pot, :dsw_pot], [:ta], [:reco], RbQ10_lazy,
    (rb = (3.0f0, 0.0f0, 13.0f0), Q10 = (2.0f0, 1.0f0, 4.0f0)), [:rb], [:Q10];
    hidden_layers = [8], scale_nn_outputs = true,
)

lazy_cube(M; chunk = 200) = DimArray(
    AccessCountDiskArray(M; chunksize = (size(M, 1), chunk)),
    (Dim{:variable}(LAZY_VARS), Dim{:batch_size}(1:size(M, 2)))
)

@testset "Lazy data" begin
    n = 3000
    M = lazy_q10_matrix(n)
    hm = lazy_q10_model()

    @testset "detection" begin
        @test is_lazy_data(lazy_cube(M))
        @test !is_lazy_data(DimArray(M, (Dim{:variable}(LAZY_VARS), Dim{:batch_size}(1:n))))
        @test !is_lazy_data(M)
    end

    @testset "LazyHybridData reads" begin
        cube = lazy_cube(M)
        d = LazyHybridData(cube, hm)
        @test numobs(d) == n
        @test d.block_span == 200                       # one storage chunk per block
        @test LazyHybridData(cube, hm; block_size = 450).block_span == 600

        ((x, forcings), (y, mask)) = getobs(d, 1:20)
        @test size(x) == (2, 18)                        # rows 5 and 17 dropped
        @test x isa Matrix{Float32}
        @test keys(forcings) == (:ta,) && keys(y) == (:reco,)
        @test all(mask.reco)
        @test x[1, 1] == M[2, 1] && forcings.ta[1] == M[1, 1] && y.reco[1] == M[4, 1]
        # Unsorted indices keep their order
        ((x2, _), _) = getobs(d, [10, 3])
        @test x2[:, 1] == M[2:3, 10] && x2[:, 2] == M[2:3, 3]

        # Blocks never straddle a storage chunk
        for b in EasyHybrid._blocks(d)
            @test length(unique((b .- 1) .÷ 200)) == 1
        end
        @test sort(reduce(vcat, EasyHybrid._blocks(d))) == 1:n

        # Views starting mid-chunk stay aligned with the storage chunk grid
        vcube = DimArray(
            view(AccessCountDiskArray(M; chunksize = (size(M, 1), 200)), :, 151:n),
            (Dim{:variable}(LAZY_VARS), Dim{:batch_size}(1:(n - 150)))
        )
        dv = LazyHybridData(vcube, hm)
        @test dv.block_offset == 150
        for b in EasyHybrid._blocks(dv)
            @test length(unique((b .+ 150 .- 1) .÷ 200)) == 1
        end

        # Duplicated indices are read as given (not as a contiguous range)
        @test EasyHybrid._as_range([1, 1, 3]) == [1, 1, 3]
        ((xd, _), _) = getobs(d, [3, 1, 1])
        @test xd == M[2:3, [3, 1, 1]]

        # Models without predictors (global parameters only)
        dn = LazyHybridData(cube, (; predictors = Symbol[], forcing = [:ta], targets = [:reco]))
        ((xn, fn), _) = getobs(dn, 1:4)
        @test size(xn) == (0, 4) && fn.ta == M[1, 1:4]

        @test_throws ArgumentError LazyHybridData(cube, hm; block_size = 0)
        @test typeof(d.obs_cache[]) == Nothing
        @test eltype(typeof(d.obs_cache)) == Union{Nothing, NamedTuple{(:reco,), Tuple{Vector{Float32}}}}
    end

    @testset "LazyBatchLoader covers every observation once" begin
        cube = lazy_cube(M)
        probe = (; predictors = [:obsid], forcing = [:ta], targets = [:reco])
        d = LazyHybridData(cube, probe; shuffle_buffer = 400)
        for shuffle in (true, false)
            loader = LazyBatchLoader(d; batchsize = 64, shuffle, shuffle_buffer = 400, rng = Random.Xoshiro(1))
            batches = collect(loader)
            seen = Int.(reduce(vcat, [vec(first(first(b))) for b in batches]))
            @test sort(seen) == setdiff(1:n, [5])    # sw_pot is not an input here, so 17 stays
            @test all(size(first(first(b)), 2) == 64 for b in batches[1:(end - 1)])
            shuffle || @test seen == setdiff(1:n, [5])
        end
        @test LazyBatchLoader(d; batchsize = 8, shuffle_buffer = 400).group == 2
        @test LazyBatchLoader(d; batchsize = 8, shuffle = false).group == 1
    end

    @testset "train on lazy data matches the in-memory initial state" begin
        kw = (;
            nepochs = 3, batchsize = 128, opt = Adam(0.01), plotting = false,
            show_progress = false, save_training = false, random_seed = 3,
        )
        cube = lazy_cube(M)
        out_lazy = train(hm, cube; kw...)

        # In-memory reference: same rows, already without the dropped observations
        keep = setdiff(1:n, [5, 17])
        mem = DimArray(M[:, keep], (Dim{:variable}(Symbol.(LAZY_VARS)), Dim{:batch_size}(1:length(keep))))
        out_mem = train(hm, mem; kw...)

        @test out_lazy.train_history[1].mse.sum ≈ out_mem.train_history[1].mse.sum rtol = 1.0e-4
        @test out_lazy.val_history[1].mse.sum ≈ out_mem.val_history[1].mse.sum rtol = 1.0e-4
        @test isfinite(out_lazy.val_history[end].mse.sum)
        @test out_lazy.val_history[end].mse.sum < out_lazy.val_history[1].mse.sum
        @test size(out_lazy.train_obs_pred) == size(out_mem.train_obs_pred)
        @test size(out_lazy.val_obs_pred) == size(out_mem.val_obs_pred)

        # Nothing was read in one go: every disk read spans at most one storage chunk.
        log = getindex_log(parent(cube))
        @test !isempty(log)
        @test maximum(length(i[2]) for i in log) <= 200
    end

    @testset "splitting lazy data" begin
        cube = lazy_cube(M)
        cfg = EasyHybrid.DataConfig(split_by_id = :site)
        tr, va = EasyHybrid.lazy_splits(cube, hm, cfg)
        sites_tr = unique(M[5, tr.idx])
        sites_va = unique(M[5, va.idx])
        @test isempty(intersect(sites_tr, sites_va))
        @test numobs(tr) + numobs(va) == n - 2              # rows 5 and 17 are not usable

        # Proportions apply to the usable observations, as for in-memory data
        tr, va = EasyHybrid.prepare_training_splits(cube, hm, EasyHybrid.DataConfig(split_data_at = 0.75))
        @test tr isa LazyHybridData && va isa LazyHybridData
        usable = setdiff(1:n, [5, 17])
        @test [tr.idx; va.idx] == usable
        @test numobs(tr) == round(Int, 0.75 * length(usable)) || numobs(tr) == floor(Int, 0.75 * length(usable))
        @test_throws ArgumentError EasyHybrid.full_batch_data(tr, EasyHybrid.TrainConfig())

        # Force the in-memory path on a disk-backed array
        tr, va = EasyHybrid.prepare_training_splits(
            DimArray(M, (Dim{:variable}(Symbol.(LAZY_VARS)), Dim{:batch_size}(1:n))), hm,
            EasyHybrid.DataConfig(lazy = false)
        )
        @test tr isa InMemorySplit

        @test_throws ArgumentError EasyHybrid.lazy_splits(cube, hm, EasyHybrid.DataConfig(sequence_length = 5))
        # Auto mode falls back to in-memory for setups the lazy path does not support
        @test !EasyHybrid._use_lazy(cube, EasyHybrid.DataConfig(sequence_length = 5))
        @test EasyHybrid._use_lazy(cube, EasyHybrid.DataConfig())
        @test_throws ArgumentError EasyHybrid.prepare_training_splits(M, hm, EasyHybrid.DataConfig(lazy = true))
    end

    @testset "placeholder batch and history compaction" begin
        d = LazyHybridData(lazy_cube(M), hm)
        loader = LazyBatchLoader(d; batchsize = 8)
        ((x, _), _) = EasyHybrid._placeholder_batch(loader)
        @test size(x, 2) <= 8

        ŷ(k) = (; reco = fill(Float32(k), 10), Q10 = Float32[k], rb = fill(Float32(k), 10))
        snap(k) = EasyHybrid.EpochSnapshot((;), (;), ŷ(k), ŷ(k), k)
        cfg = EasyHybrid.TrainConfig(monitor_names = [:Q10])
        h = EasyHybrid.TrainingHistory(snap(0))
        push!(h.snapshots, snap(1))
        EasyHybrid.compact_history!(h, d, cfg)
        @test keys(h.snapshots[1].ŷ_train) == (:Q10,) && keys(h.snapshots[1].ŷ_val) == (:Q10,)
        @test h.snapshots[2].ŷ_train == ŷ(1)                  # latest keeps everything
        mem = EasyHybrid.InMemorySplit(zeros(Float32, 1, 2), (;), (; reco = zeros(Float32, 2)))
        h2 = EasyHybrid.TrainingHistory(snap(0)); push!(h2.snapshots, snap(1))
        EasyHybrid.compact_history!(h2, mem, cfg)
        @test h2.snapshots[1].ŷ_train == ŷ(0)                 # in-memory: unchanged
    end

    @testset "concatenating block predictions" begin
        ŷs = [(; reco = Float32[1, 2], Q10 = Float32[3], p = (; rb = Float32[4, 5])), (; reco = Float32[6, 7, 8], Q10 = Float32[3], p = (; rb = Float32[9, 10, 11]))]
        c = EasyHybrid._concat_obs(ŷs, [2, 3])
        @test c.reco == Float32[1, 2, 6, 7, 8]
        @test c.Q10 == Float32[3]
        @test c.p.rb == Float32[4, 5, 9, 10, 11]
        m = EasyHybrid._concat_obs([ones(Float32, 2, 2), zeros(Float32, 2, 3)], [2, 3])
        @test size(m) == (2, 5)
    end
end
