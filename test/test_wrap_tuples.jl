using Test
using EasyHybrid: WrappedTuples

@testset "WrappedTuples" begin
    vec = [(a=1, b=2.0), (a=3, b=4.0)]
    wt = WrappedTuples(vec)

    # Basic properties
    @test typeof(wt) <: AbstractVector{NamedTuple}
    @test size(wt) == (2,)
    @test length(wt) == 2

    # Indexing
    @test wt[1] == vec[1]
    @test wt[1:1] isa WrappedTuples
    @test wt[1:1].data == vec[1:1]

    # Iteration
    @test collect(wt) == vec

    # Index style
    @test IndexStyle(WrappedTuples) isa IndexLinear

    # Dot-access to fields
    @test wt.a == [1, 3]
    @test wt.b == [2.0, 4.0]

    # Keys and propertynames
    @test keys(wt) == propertynames(vec[1])
    pn = propertynames(wt)
    @test :data in pn && :a in pn && :b in pn

    # Matrix conversion (checks promotion and column layout)
    M = Matrix(wt)
    @test size(M) == (2, 2)
    @test M[1,1] == 1.0 && M[2,1] == 3.0 && M[1,2] == 2.0 && M[2,2] == 4.0

    # Missing field raises FieldError
    @test_throws FieldError wt.x

    # Slicing preserves behavior
    sub = wt[2:2]
    @test sub.a == [3]
    @test length(sub) == 1
end
