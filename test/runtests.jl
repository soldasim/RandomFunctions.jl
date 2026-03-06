using RandomFunctions
using Test
using Aqua

@testset "RandomFunctions.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(RandomFunctions)
    end
    # Write your tests here.
end
