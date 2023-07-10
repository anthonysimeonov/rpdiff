import torch
import torch.nn as nn

from rpdiff.utils.torch3d_util import matrix_to_quaternion
from rpdiff.utils.torch_scatter_utils import FPSDownSample
from typing import List, Union, Tuple


class LocalAbstractPolicy(nn.Module):
    """
    Base class for our policies that operate on local point features. Input
    processing and output processing are the same, the only thing that differs
    between child classes are the mechanism for processing the input point features
    (consisting of the 3D coordinates of the shape pair and their corresponding
    per-point features from a potentially pre-trained encoder)
    """
    def __init__(self, n_pts: int, pn_pts: int=None, cn_pts: int=None, fixed_scaling: float=None):
        super().__init__()
        self.n_pts = n_pts
        self.pn_pts = pn_pts
        self.cn_pts = cn_pts
        self.onehot_a = torch.Tensor([[[1, 0]]]).float().cuda()
        self.onehot_b = torch.Tensor([[[0, 1]]]).float().cuda()

        self.pcd_emb = None  # Make sure this is implemented in the child class (linear projection)
        self.des_emb = None  # Make sure this is implemented in the child class (linear projection)

        if pn_pts is None:
            pn_pts = n_pts
        if cn_pts is None:
            cn_pts = n_pts

        self.fps_ds = FPSDownSample(n_pts)
        self.fps_ds_p = FPSDownSample(pn_pts)
        self.fps_ds_c = FPSDownSample(cn_pts)

        self.predict_offset = False
        self.out_trans_offset = None
        self.fixed_scaling = fixed_scaling

    def set_predict_offset(self, predict_offset):
        self.predict_offset = predict_offset

    def initialize_weights(self):
        # initialization

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.out_param, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _build_output_heads(self, pooled_dim: int, hidden_dim: int):
        """
        Creates the output 2-layer MLPs used for predicting translation and rotation
        
        Args:
            pooled_dim (int): Input dimensionality to these output heads (same as dimensionality
                of the pooled feature after processing the point features)
            hidden_dim (int): Hidden dim for these output heads
        """
        self.out_trans = nn.Sequential(nn.Linear(pooled_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        self.out_vec1 = nn.Sequential(nn.Linear(pooled_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3)) 
        self.out_vec2 = nn.Sequential(nn.Linear(pooled_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))

        if self.predict_offset:
            self.out_trans_offset = nn.Sequential(nn.Linear(pooled_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))
        
    def process_model_input(self, model_input: dict) -> Tuple[dict]:
        """
        Pre-processes the input point cloud and per-point features. Consists of scaling the 3D point
        clouds to be roughly ~unit bounding box scale each, adding the mean offsets to make them 
        in the correct position relative to each other (with a convention for centering the pair by
        the mean of the parent point cloud), and then combining all of these features first in the 
        feature dimension, and then in the point dimension
        
        Args:
            model_input (dict): Keys (Values):
                "parent/child_start_pcd" B x N x 3 (centered point clouds),
                "parent/child_start_pcd_mean" B x 3 (mean of the point clouds at their original positions
                    in the world frame)
                "parent/child_point_latent" B x N x D (per-point features from feat-encoder)

        Output:
            dict: Keys (Values): "parent/child" B x N x 3 (scaled and shifted point clouds)
            dict: Keys (Values): "parent/child" B x N x h_dim (projected feats and point clouds, concated into 
                new per-point features)
        """

        # centerd 3D point clouds
        ppcd_cent = model_input['parent_start_pcd']
        cpcd_cent = model_input['child_start_pcd']

        # offset using pre-computed means
        p2c_offset = model_input['parent_start_pcd_mean'] - model_input['child_start_pcd_mean']

        B, N = ppcd_cent.shape[0], ppcd_cent.shape[1]
        Np = ppcd_cent.shape[1]
        Nc = cpcd_cent.shape[1]

        # scale up to approximate unit bounding box, for each obj
        # vn_child_scaling = 1 / ((cpcd_cent.max(1)[0].max(-1)[0] - cpcd_cent.min(1)[0].min(-1)[0]) / 2.0)
        vn_parent_scaling = 1 / ((ppcd_cent.max(1)[0].max(-1)[0] - ppcd_cent.min(1)[0].min(-1)[0]) / 2.0)
        if self.fixed_scaling is not None:
            vn_parent_scaling = torch.Tensor([self.fixed_scaling]).repeat(B).float().cuda()
        ppcd_cent = ppcd_cent * vn_parent_scaling[:, None, None].repeat(1, Np, 1)
        cpcd_cent = cpcd_cent * vn_parent_scaling[:, None, None].repeat(1, Nc, 1)
        
        # add offset to the respective point clouds for relative positions between objects to show up, scaling the offset appropriately too
        cpcd = cpcd_cent - (p2c_offset[:, None, :].repeat(1, cpcd_cent.shape[1], 1) * vn_parent_scaling[:, None, None].repeat(1, Nc, 1))
        ppcd = ppcd_cent

        # util.meshcat_pcd_show(self.mc_vis, ppcd[0].detach().cpu().numpy(), (255, 0, 0), 'scene/process/ppcd')
        # util.meshcat_pcd_show(self.mc_vis, cpcd[0].detach().cpu().numpy(), (0, 0, 255), 'scene/process/cpcd')

        # feature-wise concatenation - [3D coord embedding, feature encoder embedding]
        parent_pcd_emb = self.pcd_emb(ppcd)
        child_pcd_emb = self.pcd_emb(cpcd)
        parent_latent_emb = self.des_emb(ppcd_cent) # TODO - where a separate point cloud encoder/featurizer for each individual shape could go (for now "dummy features" from the 3D points)
        child_latent_emb = self.des_emb(cpcd_cent)

        # append one-hot for object A/B identity
        des_full_parent_coords = torch.cat([parent_pcd_emb, parent_latent_emb, self.onehot_a.repeat((B, Np, 1))], dim=-1)
        des_full_child_coords = torch.cat([child_pcd_emb, child_latent_emb, self.onehot_b.repeat((B, Nc, 1))], dim=-1)
        
        # downsample with random permutation (voxel or fps downsampling would be better...)
        des = dict(parent=des_full_parent_coords, child=des_full_child_coords)
        pcd = dict(parent=ppcd, child=cpcd)

        return des, pcd

    def process_model_output(self, pooled_h: torch.FloatTensor) -> dict:
        B = pooled_h.shape[0]

        out_trans = self.out_trans(pooled_h)
        out_vec1 = self.out_vec1(pooled_h)
        out_vec2 = self.out_vec2(pooled_h)

        if out_vec1.ndim == 3:
            nq = out_vec1.shape[1]

        if out_vec1.ndim == 3:
            # b = z1/||z1||, a = (z2 - <b, z2>b)/||z2||, makes a and b orthonormal
            vec1 = torch.nn.functional.normalize(out_vec1, dim=-1)
            vec2 = (out_vec2 - (vec1 * out_vec2).sum(-1).view(B, -1, 1) * vec1) / torch.norm(out_vec2, dim=-1).view(B, -1, 1)
            vec2 = torch.nn.functional.normalize(vec2, dim=-1)
            vec3 = torch.cross(vec1, vec2, dim=-1)

            out_rot_mat = torch.stack([vec1, vec2, vec3], dim=-1)
            out_quat = matrix_to_quaternion(out_rot_mat)
            out_quat_unnorm = out_quat
        else:
            nq = 1
            vec1 = torch.nn.functional.normalize(out_vec1, dim=-1)
            vec2 = (out_vec2 - (vec1 * out_vec2).sum(-1).view(B, -1) * vec1) / torch.norm(out_vec2, dim=-1).view(B, -1)
            vec2 = torch.nn.functional.normalize(vec2, dim=-1)
            vec3 = torch.cross(vec1, vec2, dim=-1)

            out_rot_mat = torch.stack([vec1, vec2, vec3], dim=-1)
            out_quat = matrix_to_quaternion(out_rot_mat)
            out_quat_unnorm = out_quat

        model_output = dict(
            trans=out_trans, 
            unnorm_quat=out_quat_unnorm, 
            quat=out_quat, 
            rot_mat=out_rot_mat)

        if self.predict_offset:
            out_trans_offset = self.out_trans_offset(pooled_h)
            out_trans_offset = torch.nn.functional.normalize(out_trans_offset, dim=-1)
            model_output['trans_offset'] = out_trans_offset

        return model_output


class LocalAbstractSuccessClassifier(LocalAbstractPolicy):
    def __init__(self, n_pts: int, pn_pts: int=None, cn_pts: int=None, 
                 sigmoid: bool=True, fixed_scaling: float=None):
        super().__init__(n_pts=n_pts, pn_pts=pn_pts, cn_pts=cn_pts, fixed_scaling=fixed_scaling)
        self.sigmoid = sigmoid

    def _build_output_heads(self, pooled_dim: int, hidden_dim: int):
        """
        Creates the output 2-layer MLPs used for predicting translation and rotation
        
        Args:
            pooled_dim (int): Input dimensionality to these output heads (same as dimensionality
                of the pooled feature after processing the point features)
            hidden_dim (int): Hidden dim for these output heads
        """
        self.out_success = nn.Sequential(nn.Linear(pooled_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def process_model_output(self, pooled_h: torch.FloatTensor) -> dict:
        #out_success = self.out_success(pooled_h)
        if self.sigmoid:
            out_success = torch.sigmoid(self.out_success(pooled_h))
        else:
            out_success = self.out_success(pooled_h)

        model_output = dict(
            success=out_success)

        return model_output

