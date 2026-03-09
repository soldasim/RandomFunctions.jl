module RandomFunctions

export sample
export FunctionData
export FunctionSampler, ComposedSampler

export GPSampler
export ClusteredSampler

using Distributions
using KernelFunctions
using LinearAlgebra
using Random

include("types.jl")
include("composed.jl")
include("gaussian_process.jl")
include("clustered.jl")

end
