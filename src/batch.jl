
"""
    BatchSampler(; kwargs...)

Samples `FunctionData` using the provided `FunctionSampler` with specified batch size
and random context ant target point counts drawn from predefined priors.

## Keywords
- `sampler::FunctionSampler`: The underlying function sampler to use for generating data.
- `n_batch::Int`: The number of function samples to generate in a batch (default: 32).
- `n_context_prior::DiscreteUnivariateDistribution`: Prior distribution for sampling the number of context points (default: `DiscreteUniform(8, 128)`).
- `n_target_prior::DiscreteUnivariateDistribution`: Prior distribution for sampling the number of target points (default: `Dirac(128)`).
"""
@kwdef struct BatchSampler
    sampler::FunctionSampler
    n_batch::Int = 32
    n_context_prior::DiscreteUnivariateDistribution = DiscreteUniform(8, 128)
    n_target_prior::DiscreteUnivariateDistribution = Dirac(128)
end

function sample(sampler::BatchSampler)
    n_context = rand(sampler.n_context_prior)
    n_target = rand(sampler.n_target_prior)
    
    return RandomFunctions.sample(sampler.sampler, n_context, n_target, sampler.n_batch)
end
