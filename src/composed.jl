
"""
    ComposedSampler <: FunctionSampler
    ComposedSampler(::InputSampler, ::OutputSampler)

A `FunctionSampler` composed of an `InputSampler` and an `OutputSampler`.
"""
struct ComposedSampler <: FunctionSampler
    input_sampler::InputSampler
    output_sampler::OutputSampler
end

function sample(sampler::ComposedSampler, n_context::Int, n_target::Int, n_samples::Int)
    x_context, x_target = sample(sampler.input_sampler, n_context, n_target, n_samples)
    y_context, y_target = sample(sampler.output_sampler, x_context, x_target)
    return FunctionData(x_context, y_context, x_target, y_target)
end

x_dim(sampler::ComposedSampler) = x_dim(sampler.input_sampler)
y_dim(sampler::ComposedSampler) = y_dim(sampler.output_sampler)
