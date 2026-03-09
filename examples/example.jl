using RandomFunctions
using CairoMakie
using KernelFunctions
using Distributions

# Set up the input sampler (ClusteredSampler)
# This will sample clustered input points in 2D space
input_sampler = ClusteredSampler(
    x_dim = 2,                                          # 2D input space
    x_range = (-2.0, 2.0),                              # x values from -2 to 2
    n_clusters_prior = DiscreteUniform(2, 4),           # 2-4 clusters per sample
    outlier_ratio_prior = Uniform(0.1, 0.2),            # 10-20% outliers
    relative_cluster_std_prior = Uniform(0.05, 0.15)    # cluster spread
)

# Set up the output sampler (GPSampler)
# This will generate smooth function outputs using a Gaussian Process
output_sampler = GPSampler(
    kernel = GaussianKernel(),                          # RBF kernel
    length_scale_prior = Uniform(0.2, 0.8),             # controls smoothness
    amplitude_prior = Uniform(0.5, 1.5),                # controls output scale
    noise_std_prior = Dirac(1e-6)                       # numerical stability
)

# Compose the samplers
sampler = ComposedSampler(input_sampler, output_sampler)

# Sample function data
n_context_points = 20   # number of context points per sample
n_target_points = 50    # number of target points per sample
n_samples = 4           # number of function samples

data = RandomFunctions.sample(sampler, n_context_points, n_target_points, n_samples)

# Create visualization
fig = Figure(size = (1200, 800))

# Plot each sample in a separate subplot
for i in 1:n_samples
    ax = Axis(fig[div(i-1, 2) + 1, mod(i-1, 2) + 1],
        xlabel = "x₁",
        ylabel = "x₂",
        title = "Sample $i",
        aspect = DataAspect()
    )
    
    # Extract data for this sample
    ctx_x1 = data.context_x[1, :, i]  # context x1 coordinates
    ctx_x2 = data.context_x[2, :, i]  # context x2 coordinates
    ctx_y = data.context_y[1, :, i]   # context y values (outputs)
    tgt_x1 = data.target_x[1, :, i]   # target x1 coordinates
    tgt_x2 = data.target_x[2, :, i]   # target x2 coordinates
    tgt_y = data.target_y[1, :, i]    # target y values (outputs)
    
    # Plot target points with color representing function output
    scatter!(ax, tgt_x1, tgt_x2,
        color = tgt_y, colormap = :viridis, markersize = 12,
        label = "Target points")
    
    # Plot context points (the clustered observations) with larger markers
    scatter!(ax, ctx_x1, ctx_x2,
        color = ctx_y, colormap = :viridis, markersize = 18,
        marker = :circle, strokewidth = 2, strokecolor = :red,
        label = "Context points")
    
    # Add legend and colorbar to first subplot
    if i == 1
        axislegend(ax, position = :lt)
        Colorbar(fig[div(i-1, 2) + 1, 3], label = "Output (y)", colormap = :viridis)
    end
end

# Display the figure
display(fig)

# Save the figure
save("example_output.png", fig)
println("Figure saved to example_output.png")


