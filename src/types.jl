
"""
    FunctionData(context_x, context_y, target_x, target_y)

A struct to hold the context and target data of sampled functions.

## Fields
- `context_x::AbstractArray{<:Real, 3}`: shape (x_dim, n_context_points, n_batch)
- `context_y::AbstractArray{<:Real, 3}`: shape (y_dim, n_context_points, n_batch)
- `target_x::AbstractArray{<:Real, 3}`: shape (x_dim, n_target_points, n_batch)
- `target_y::AbstractArray{<:Real, 3}`: shape (y_dim, n_target_points, n_batch)
"""
struct FunctionData
    context_x::AbstractArray{<:Real, 3}
    context_y::AbstractArray{<:Real, 3}
    target_x::AbstractArray{<:Real, 3}
    target_y::AbstractArray{<:Real, 3}

    function FunctionData(context_x, context_y, target_x, target_y)
        @assert size(context_x, 1) == size(target_x, 1) "Input dimensionality must match between context and target."
        @assert size(context_y, 1) == size(target_y, 1) "Output dimensionality must match between context and target."
        @assert size(context_x, 2) == size(context_y, 2) "Number of context points must match between inputs and outputs."
        @assert size(target_x, 2) == size(target_y, 2) "Number of target points must match between inputs and outputs."
        @assert size(context_x, 3) == size(context_y, 3) == size(target_x, 3) == size(target_y, 3) "Number of samples must match between context and target."
        new(context_x, context_y, target_x, target_y)
    end
end

x_dim(data::FunctionData) = size(data.context_x, 1)
y_dim(data::FunctionData) = size(data.context_y, 1)
n_context(data::FunctionData) = size(data.context_x, 2)
n_target(data::FunctionData) = size(data.target_x, 2)
n_batch(data::FunctionData) = size(data.context_x, 3)

"""
    FunctionSampler

An abstract type for sampling functions.

Each subtype of `FunctionSampler` should implement the following methods:
- `sample(::FunctionSampler, n_context::Int, n_target::Int, n_batch::Int) -> ::FunctionData`
- `x_dim(::FunctionSampler) -> Int`
- `y_dim(::FunctionSampler) -> Int`
"""
abstract type FunctionSampler end

"""
    InputSampler

An abstract type for sampling input points for functions.

Each subtype of `InputSampler` should implement the following method:
- `sample(::InputSampler, n_context::Int, n_target::Int, n_batch::Int) -> (x_context, x_target)`
- `x_dim(::InputSampler) -> Int`

See also [`FunctionData`](@ref) for the expected shapes of the input and output arrays.
"""
abstract type InputSampler end

"""
    OutputSampler

An abstract type for sampling output values for functions.

Each subtype of `OutputSampler` should implement the following method:
- `sample(::OutputSampler, x_context, x_target) -> (y_context, y_target)`
- `y_dim(::OutputSampler) -> Int`

See also [`FunctionData`](@ref) for the expected shapes of the input and output arrays.
"""
abstract type OutputSampler end

"""
    sample(sampler::FunctionSampler, n_context::Int, n_target::Int, n_batch::Int) -> FunctionData
    sample(sampler::InputSampler, n_context::Int, n_target::Int, n_batch::Int) -> (x_context, x_target)
    sample(sampler::OutputSampler, x_context, x_target) -> (y_context, y_target)

The `sample` function is the main interface for sampling functions, input points, and output values.

See also [`FunctionData`](@ref).
"""
function sample end

"""
    x_dim(::FunctionSampler) -> Int
    x_dim(::InputSampler) -> Int

Returns the dimensionality of the input space for a given sampler.

See also [`y_dim`](@ref).
"""
function x_dim end

"""
    y_dim(::FunctionSampler) -> Int
    y_dim(::OutputSampler) -> Int

Returns the dimensionality of the output space for a given sampler.

See also [`x_dim`](@ref).
"""
function y_dim end
