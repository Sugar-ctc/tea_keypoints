import torch
import torch.nn.functional as F
import numpy as np
def compute_hog_features(prior):
    b, c, h, w = prior.shape  


    prior_gray = torch.zeros((b, h, w), dtype=torch.float32, device=prior.device)
    prior_gray = 0.2989 * prior[:, 0] + 0.5870 * prior[:, 1] + 0.1140 * prior[:, 2]  # (b, h, w)
    sobel_x = torch.tensor([[1, 0, -1],
                             [2, 0, -2],
                             [1, 0, -1]], dtype=torch.float32, device=prior.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1],
                             [0, 0, 0],
                             [-1, -2, -1]], dtype=torch.float32, device=prior.device).view(1, 1, 3, 3)

    gx = F.conv2d(prior_gray.unsqueeze(1), sobel_x, padding=1)  # (b, 1, h, w)
    gy = F.conv2d(prior_gray.unsqueeze(1), sobel_y, padding=1)  # (b, 1, h, w)

    magnitude = torch.sqrt(gx**2 + gy**2)
    angle = torch.atan2(gy, gx) * (180 / np.pi) % 180  # (b, 1, h, w)

    bins = torch.arange(0, 180, 20, device=prior.device)  # [0, 20, 40, ..., 160]
    bin_count = bins.size(0)


    bin_index = (angle / 20).floor().long()  # (b, 1, h, w)
    bin_index = torch.clamp(bin_index, 0, bin_count - 1)  

    grid_size = 4
    linenum = h // grid_size
    cownum = w // grid_size
    histograms = torch.zeros((b, linenum * cownum, bin_count), device=prior.device)

    for j in range(linenum):
        for k in range(cownum):
            x_start = j * grid_size
            x_end = (j + 1) * grid_size
            y_start = k * grid_size
            y_end = (k + 1) * grid_size

            mag_cell = magnitude[:, 0, x_start:x_end, y_start:y_end]  # (b, 1, grid_size, grid_size)
            ang_cell = bin_index[:, 0, x_start:x_end, y_start:y_end]  # (b, 1, grid_size, grid_size)

            # Flatten to (b, grid_size * grid_size)
            mag_cell_flat = mag_cell.reshape(b, -1)  # (b, grid_size * grid_size)
            ang_cell_flat = ang_cell.reshape(b, -1)  # (b, grid_size * grid_size)

            for bin_num in range(bin_count):
                weight = (ang_cell_flat == bin_num).float()
                histograms[:, j * cownum + k, bin_num] = torch.sum(mag_cell_flat * weight, dim=1)

    return histograms  
