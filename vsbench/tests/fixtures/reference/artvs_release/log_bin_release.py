# Transcribed from the ART-VS release (git 9b1208c, MIT, "Alessandro Scherl and the ART-VS
# authors"): catkin_ws/ibvs/src/visual_servoing/features/multi_backbone_extractor.py,
# MultiBackboneExtractor._log_bin, lines 852-912. The method body is kept line for line;
# `self.num_patches` and `self.device` became the arguments `num_patches` and `device`, and
# the one-time "[BINNING]" print (lines 869-874) was removed. See PROVENANCE.md.
import torch


def log_bin_release(x: torch.Tensor, num_patches, hierarchy: int = 1, device="cpu") -> torch.Tensor:
    """
    Create a log-binned descriptor for AM-RADIO and other models.
    Adapted from DINOv2 implementation for multi-backbone use.

    Args:
        x: tensor of features. Has shape [B, 1, num_patches, feature_dim].
        hierarchy: how many bin hierarchies to use.

    Returns:
        Binned features of shape [B, 1, num_patches, binned_feature_dim]
    """

    B = x.shape[0]
    num_bins = 1 + 8 * hierarchy

    # Remove the singleton dimension and rearrange: [B, num_patches, feature_dim] -> [B, feature_dim, H, W]
    bin_x = x.squeeze(1)  # [B, num_patches, feature_dim]
    bin_x = bin_x.permute(0, 2, 1)  # [B, feature_dim, num_patches]
    bin_x = bin_x.reshape(B, bin_x.shape[1], num_patches[0], num_patches[1])
    # Now: [B, feature_dim, H, W]

    sub_desc_dim = bin_x.shape[1]  # feature_dim

    avg_pools = []
    # Compute bins of all sizes for all spatial locations
    for k in range(0, hierarchy):
        # avg pooling with kernel 3**k x 3**k
        win_size = 3 ** k
        avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
        avg_pools.append(avg_pool(bin_x))

    bin_x = torch.zeros((B, sub_desc_dim * num_bins, num_patches[0], num_patches[1])).to(device)

    for y in range(num_patches[0]):
        for x_coord in range(num_patches[1]):
            part_idx = 0
            # Fill all bins for a spatial location (y, x_coord)
            for k in range(0, hierarchy):
                kernel_size = 3 ** k
                for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                    for j in range(x_coord - kernel_size, x_coord + kernel_size + 1, kernel_size):
                        if i == y and j == x_coord and k != 0:
                            continue
                        if 0 <= i < num_patches[0] and 0 <= j < num_patches[1]:
                            bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x_coord] = avg_pools[k][:, :, i, j]
                        else:  # Handle padding
                            temp_i = max(0, min(i, num_patches[0] - 1))
                            temp_j = max(0, min(j, num_patches[1] - 1))
                            bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x_coord] = avg_pools[k][:, :, temp_i, temp_j]
                        part_idx += 1

    # Reshape back to sequence format: [B, 1, num_patches, binned_feature_dim]
    bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
    return bin_x
