# NeRF Studio splatfacto init vs Gaussian Splatting simple-knn-cuda

After reviewing the `distCUDA2` implementation and the CUDA kernel in `simple_knn.cu`, I can now provide a more informed comparison of the scales calculation in the two code snippets.

In code 1: SplatFactoModel in NeRF Studio

```python
distances, _ = self.k_nearest_sklearn(means.data, 3)
distances = torch.from_numpy(distances)
avg_dist = distances.mean(dim=-1, keepdim=True)
scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
```

The `scales` are calculated by finding the average distance to the 3 nearest neighbors for each point using a CPU-based implementation (`k_nearest_sklearn`).

In code 2: [simple-knn-cuda](https://gitlab.inria.fr/bkerbl/simple-knn)

```python
dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
```

The `scales` are calculated using the `distCUDA2` function, which internally calls the `SimpleKNN::knn` CUDA kernel. This kernel computes an approximate mean distance for each point by:

1. Sorting the points using a Morton code-based spatial partitioning.
2. Dividing the points into boxes of size `BOX_SIZE` (1024).
3. For each point, finding the 3 nearest neighbors within a small neighborhood (neighboring boxes).
4. Computing the mean of these 3 nearest neighbor distances as an approximation of the mean distance.

So while both snippets involve computing scales based on distances between points and applying a logarithm, code 1 uses a CPU-based exact nearest neighbor implementation, while code 2 uses an approximate GPU-based approach for efficiency.

The GPU-based approach trades off some accuracy for significant performance gains by leveraging spatial partitioning and parallelism. The approximation quality likely depends on the point cloud distribution and the value of `BOX_SIZE`.

In summary, the underlying concept of computing scales from distances is the same, but the implementations differ in their approach (exact CPU-based vs. approximate GPU-based) and accuracy/performance trade-offs.
