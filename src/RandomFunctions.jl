module RandomFunctions

export sample
export FunctionData
export FunctionSampler, ComposedSampler
export BatchSampler

export GPSampler
export ClusteredSampler

using Distributions
using KernelFunctions
using LinearAlgebra
using Random

include("types.jl")
include("composed.jl")
include("batch.jl")
include("samplers/include.jl")

end # module
