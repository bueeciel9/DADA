import torch

def compute_voxel_density(points, voxel_size, coordinate_min, coordinate_max):
    """
    Compute the density of each voxel.
    
    :param points: (torch.Tensor) Point cloud data, shape (B, N, 3)
    :param voxel_size: (float) The size of each voxel
    :param coordinate_min: (list or tuple) The minimum coordinate values
    :param coordinate_max: (list or tuple) The maximum coordinate values
    :return: (torch.Tensor) The voxel density, shape (B, H, W, D)
    """
    
    B, N, _ = points.shape
    coordinate_range = [max_val - min_val for min_val, max_val in zip(coordinate_min, coordinate_max)]
    grid_size = [int(coordinate_range[i] / voxel_size) for i in range(3)]
    H, W, D = grid_size
    
    # Compute the voxel indices for each point
    voxel_indices = ((points - torch.tensor(coordinate_min, device=points.device).view(1, 1, 3)) / voxel_size).long()
    
    # Compute the voxel density
    voxel_density = torch.zeros((B, H, W, D), device=points.device)
    for b in range(B):
        for n in range(N):
            h, w, d = voxel_indices[b, n]
            if 0 <= h < H and 0 <= w < W and 0 <= d < D:
                voxel_density[b, h, w, d] += 1
                
    return voxel_density
# Now, you need to integrate the voxel density calculation into the VOTR model. Open the pcdet/models/dense_heads/votr_head.py file and import the compute_voxel_density function:


from .voxel_density import compute_voxel_density
# In the same file, locate the forward function in the VOTRHead class. Calculate the voxel density at the beginning of the function, right after the batch_size calculation:

voxel_density = compute_voxel_density(xyz, self.voxel_size, self.coordinate_min, self.coordinate_max)
# Modify the model architecture to include the voxel density information during the convolution phase. You can concatenate the density tensor to the input tensor for each convolutional layer, or use a more complex approach to combine the two. For example:


# Concatenate the density tensor to the input tensor
input_features = torch.cat((input_features, voxel_density.view(batch_size, -1, self.grid_size[0], self.grid_size[1], self.grid_size[2])), dim=1)
# Re-train the VOTR model with the modified architecture to incorporate the voxel density information.
