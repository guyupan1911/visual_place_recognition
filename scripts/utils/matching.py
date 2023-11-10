import torch
import numpy as np

def match_descriptors(descriptors_1, descriptors_2):
    if descriptors_1.shape[0] == 0 or descriptors_2.shape[0] == 0:
        return np.zeros((0, 3))

    # send data to GPU
    descriptors1_torch = torch.from_numpy(descriptors_1).to('cuda')
    descriptors2_torch = torch.from_numpy(descriptors_2).to('cuda')
    # make sure its double (because CUDA tensors only supports floating-point)
    descriptors1_torch = descriptors1_torch.float()
    descriptors2_torch = descriptors2_torch.float()
    # sanity check
    simmilarity_matrix = descriptors1_torch @ descriptors2_torch.t()
    scores = torch.max(simmilarity_matrix, dim=1)[0]
    nearest_neighbor_idx_1vs2 = torch.max(simmilarity_matrix, dim=1)[1]
    nearest_neighbor_idx_2vs1 = torch.max(simmilarity_matrix, dim=0)[1]
    ids1 = torch.arange(0, simmilarity_matrix.shape[0], device=descriptors1_torch.device)
    # cross check
    mask = ids1 == nearest_neighbor_idx_2vs1[nearest_neighbor_idx_1vs2]

    # lowe's ratio check
    # mask2 = simmilarity_matrix.topk(2, dim=1)[0][:, 1] < \
    #         simmilarity_matrix.topk(2, dim=1)[0][:, 0] * 0.7
    # mask = torch.logical_and(mask, mask2)

    matches_torch = torch.stack(
        [ids1[mask].type(torch.float), nearest_neighbor_idx_1vs2[mask].type(torch.float), scores[mask]]).t()
    # retrieve data back from GPU
    matches = matches_torch.data.cpu().numpy()
    matches = matches.astype(np.float)
    return matches
