import torch
import torch.nn as nn
import torch.nn.functional as F

from knn_cuda import KNN
from torch_cluster import fps as fps_cluster
from torch_cluster import knn as knn_cluster
from torch_cluster import grid_cluster
from torch_scatter import scatter_add, scatter_mean


def fps(data, number, batch=None, random_start=False, return_idx=False):
    '''
    Furthest point sampling. 

        data B N 3
        number int
        batch B*N 1 (batch index for each point in the flattened input)
    '''
    # torch geometric batching version
    bs, n_pts, pt_dim = data.shape[0], data.shape[1], data.shape[2]
    ratio = float(number / n_pts * 1.0)

    # flatten data, supposing we have the batch tensor to index
    fps_idx = fps_cluster(data.reshape(-1, pt_dim), batch=batch, ratio=ratio, random_start=random_start)
    fps_data = data.reshape(-1, pt_dim)[fps_idx].reshape(bs, -1, pt_dim)
    if return_idx:
        out_idx = fps_idx.reshape(bs, -1) % n_pts
        return fps_data, out_idx
    else:
        return fps_data


def knn_interpolate(x, pos_x, pos_y, batch_x=None, batch_y=None, k=3):
    """
    from torch_cluster
    """
    with torch.no_grad():
        assign_index = knn_cluster(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y)
        y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    y = scatter_add(x[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0))
    y = y / scatter_add(weights, y_idx, dim=0, dim_size=pos_y.size(0))

    return y


class FPSDownSample:
    def __init__(self, num_group):
        self.num_group = num_group

    def forward(self, xyz, return_idx=False):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape

        # fps the centers out

        batch_idx = torch.arange(0, batch_size) # ints from 0 to batch_size
        batch_idx = batch_idx[:, None].repeat(1, num_points) # expand to have a batch index for each point
        batch_idx = batch_idx.view(-1).long().cuda() # flatten 

        # pass in points to group, number of groups, and batch index of the point cloud (will be flattened in fps)
        out_fps = fps(xyz, self.num_group, batch=batch_idx, return_idx=return_idx) # B G 3

        downsampled_idx = None
        if return_idx:
            downsampled_xyz, downsampled_idx = out_fps
        else:
            downsampled_xyz = out_fps
        
        if return_idx:
            return downsampled_xyz, downsampled_idx
        else:
            return downsampled_xyz


def fps_downsample(xyz, num_group, return_idx=False):
    '''
        input: B N 3
        ---------------------------
        output: B G M 3
        center : B G 3
    '''
    batch_size, num_points, _ = xyz.shape

    # fps the centers out
    batch_idx = torch.arange(0, batch_size) # ints from 0 to batch_size
    batch_idx = batch_idx[:, None].repeat(1, num_points) # expand to have a batch index for each point
    batch_idx = batch_idx.view(-1).long().cuda() # flatten 

    # pass in points to group, number of groups, and batch index of the point cloud (will be flattened in fps)
    out_fps = fps(xyz, num_group, batch=batch_idx, return_idx=return_idx) # B G 3

    downsampled_idx = None
    if return_idx:
        downsampled_xyz, downsampled_idx = out_fps
    else:
        downsampled_xyz = out_fps
    
    if return_idx:
        return downsampled_xyz, downsampled_idx
    else:
        return downsampled_xyz


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)
        self.batch_idx = None


    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape

        # fps the centers out
        batch_idx = torch.arange(0, batch_size) # ints from 0 to batch_size
        batch_idx = batch_idx[:, None].repeat(1, num_points) # expand to have a batch index for each point
        batch_idx = batch_idx.reshape(-1).long().cuda() # flatten 

        # pass in points to group, number of groups, and batch index of the point cloud (will be flattened in fps)
        center = fps(xyz, self.num_group, batch=batch_idx) # B G 3

        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).reshape(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.reshape(-1)
        neighborhood = xyz.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

