
"""
    GPSampler(; kwargs...)

A sampler for generating 1D Gaussian Process (GP) function outputs.

## Keywords
- `kernel::Kernel`: The base kernel function for the GP (default: `GaussianKernel()`)
- `length_scale_prior::UnivariateDistribution`: Prior distribution for sampling length scales 
  (default: `Uniform(0.1, 1.0)`). One length scale is sampled per input dimension.
- `amplitude_prior::UnivariateDistribution`: Prior distribution for sampling the signal amplitude 
  (default: `Uniform(0.1, 1.0)`)
- `noise_std_prior::UnivariateDistribution`: Prior distribution for sampling the noise standard 
  deviation (default: `Dirac(1e-6)`). Used for numerical stability.
"""
@kwdef struct GPSampler <: OutputSampler
	kernel::Kernel = GaussianKernel()
	length_scale_prior::UnivariateDistribution = Uniform(0.1, 1.0)
	amplitude_prior::UnivariateDistribution = Uniform(0.1, 1.0)
	noise_std_prior::UnivariateDistribution = Dirac(1e-6)
end

y_dim(::GPSampler) = 1

function sample(sampler::GPSampler, x_context::AbstractArray{<:Real, 3}, x_target::AbstractArray{<:Real, 3})
    inputs = cat(x_context, x_target; dims=2)  # (dim_x, n_context + n_target, n_batch)
    dim_x, n_points, n_batch = size(inputs)
    n_context = size(x_context, 2)
    n_target = size(x_target, 2)
    
    # Initialize output array
    outputs = zeros(1, n_points, n_batch)
    
    # For each sample
    for b in 1:n_batch
        # Sample hyperparameters from their priors
        length_scales = [rand(sampler.length_scale_prior) for _ in 1:dim_x]
        amplitude = rand(sampler.amplitude_prior)
        noise_std = rand(sampler.noise_std_prior)
        
        # Get input points for this sample
        X = inputs[:, :, b]  # (dim_x, n_points)
        
        # Build kernel with sampled hyperparameters
        k = amplitude^2 * (sampler.kernel ∘ ARDTransform(1 ./ length_scales))
        
        # Compute covariance matrix
        K = kernelmatrix(k, ColVecs(X))
        
        # Add noise to diagonal for numerical stability
        K += noise_std^2 * I
        
        # Sample from the GP (multivariate normal with zero mean)
        outputs[1, :, b] = rand(MvNormal(zeros(n_points), Symmetric(K)))
    end
    
    y_context = outputs[:, 1:n_context, :]
    y_target = outputs[:, n_context+1:end, :]
    return y_context, y_target
end
