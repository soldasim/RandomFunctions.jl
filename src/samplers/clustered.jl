
@kwdef struct ClusteredSampler <: InputSampler
    x_dim::Int = 1
    x_range::Tuple{Float64, Float64} = (-1.0, 1.0)
    n_clusters_prior::DiscreteUnivariateDistribution = DiscreteUniform(1, x_dim + 1)
    outlier_ratio_prior::UnivariateDistribution = Uniform(0.05, 0.2)
    relative_cluster_std_prior::UnivariateDistribution = Uniform(0.02, 0.1)
end

x_dim(sampler::ClusteredSampler) = sampler.x_dim

function sample(sampler::ClusteredSampler, n_context::Int, n_target::Int, n_batch::Int)
    x_min, x_max = sampler.x_range

    # Arrays use shape: (x_dim, num_points, n_batch)
    x_context = zeros(sampler.x_dim, n_context, n_batch)
    x_target = zeros(sampler.x_dim, n_target, n_batch)

    for b in 1:n_batch
        # Sample x points with clustering
        n_clusters = rand(sampler.n_clusters_prior)

        # Sample cluster standard deviations
        cluster_std = rand(sampler.relative_cluster_std_prior, n_clusters) .* (x_max - x_min)
        
        # Sample cluster centers uniformly, ensuring at least 2*std away from edges
        cluster_centers = zeros(sampler.x_dim, n_clusters)
        for c in 1:n_clusters
            for d in 1:sampler.x_dim
                min_center = x_min + 2 * cluster_std[c]
                max_center = x_max - 2 * cluster_std[c]
                if min_center > max_center
                    cluster_centers[d, c] = (min_center + max_center) / 2
                else
                    cluster_centers[d, c] = rand(Uniform(min_center, max_center))
                end
            end
        end

        # Decide how many points are outliers (context only)
        outlier_fraction = rand(sampler.outlier_ratio_prior)
        n_outliers = round(Int, outlier_fraction * n_context)
        n_clustered = n_context - n_outliers
        
        # Sample outlier points uniformly
        if n_outliers > 0
            x_context[:, 1:n_outliers, b] = rand(Uniform(x_min, x_max), sampler.x_dim, n_outliers)
        end
        
        # Sample clustered points from Gaussians around cluster centers
        for i in 1:n_clustered
            # Randomly assign to a cluster
            cluster_idx = rand(1:n_clusters)
            for d in 1:sampler.x_dim
                x_context[d, n_outliers + i, b] = clamp(
                    cluster_centers[d, cluster_idx] + randn() * cluster_std[cluster_idx],
                    x_min, x_max
                )
            end
        end

        # Sample target points using a Latin hypercube grid
		bin_width = (x_max - x_min) / n_target
		for d in 1:sampler.x_dim
			vals = x_min .+ ((0:n_target-1) .+ rand(n_target)) .* bin_width
			x_target[d, :, b] = vals[randperm(n_target)]
		end
    end

    return x_context, x_target
end
