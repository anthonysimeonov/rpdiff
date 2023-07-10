import os.path as osp
import torch
import torch.nn as nn

from rpdiff.model.scene_encoder import LocalPoolPointnet
from rpdiff.utils.mesh_util import three_util


class CoarseAffordanceVoxelRot(nn.Module):
    def __init__(self, feat_dim, rot_grid_dim, 
                 padding, scene_encoder_kwargs, voxel_reso_grid,
                 hidden_dim=256, softmax_out=False, 
                 euler_rot=False, euler_bins_per_axis=72,  mc_vis=None):
        super().__init__()

        self.scene_encoder = LocalPoolPointnet(
            dim=feat_dim,
            padding=padding,
            grid_resolution=voxel_reso_grid,
            mc_vis=mc_vis,
            **scene_encoder_kwargs)
        # self.scene_encoder = MinkLocalPoolPointnet(
        #     dim=feat_dim,
        #     padding=padding,
        #     grid_resolution=voxel_reso_grid,
        #     mc_vis=mc_vis,
        #     **scene_encoder_kwargs)
        
        self.voxel_reso_grid = voxel_reso_grid
        self.aff_feat_dim = scene_encoder_kwargs.c_dim

        self.affordance_head = nn.Sequential(
            nn.Linear(self.aff_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        
        self.euler_rot = euler_rot
        self.euler_bins_per_axis = euler_bins_per_axis
        if self.euler_rot:
            self.rot_affordance_head = nn.Sequential(
                nn.Linear(self.aff_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.euler_bins_per_axis*3))
        else:
            self.rot_affordance_head = nn.Sequential(
                nn.Linear(self.aff_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, int(rot_grid_dim)))

        self.onehot_a = torch.Tensor([[[1, 0]]]).float().cuda()
        self.onehot_b = torch.Tensor([[[0, 1]]]).float().cuda()

        self.softmax_out = softmax_out
        self.softmax = nn.Softmax(dim=1)
        # self.softmax3d = SpatialSoftmax3D(32, 32, 32, 1)

    def encode_scene_fea_grid(self, p, c=None):
        # concatenate additional features, assuming this is just the parent object

        B, N = p.shape[0], p.shape[1]
        p = torch.cat([p, self.onehot_a.repeat((B, N, 1))], dim=-1)
        if c is not None:
            # do same for child object if it's included
            Nc = c.shape[1]
            c = torch.cat([c, self.onehot_b.repeat((B, Nc, 1))], dim=-1)
            p = torch.cat([p, c], dim=1)

        fea_grid = self.scene_encoder(p)
        return fea_grid

    def voxel_affordance(self, x):
        bs = x.shape[0]
        x_vox = self.affordance_head(x)

        out_vox = x_vox.reshape(bs, -1)

        if self.softmax_out:
            # out = self.softmax(out)
            out_vox = self.softmax(out_vox)

        model_output = dict(
            voxel_affordance=out_vox,
        )

        return model_output

    def rot_affordance(self, rot_model_input, mean_child=True, global_feat=False, global_feat_type='pc_pts'):
        ppcd = rot_model_input['parent_start_pcd_sn']
        cpcd = rot_model_input['child_start_pcd_sn']

        B = ppcd.shape[0] 
        Np, Nc = ppcd.shape[1], cpcd.shape[1]
        ppcd = torch.cat([ppcd, self.onehot_a.repeat((B, Np, 1))], dim=-1)
        cpcd = torch.cat([cpcd, self.onehot_b.repeat((B, Nc, 1))], dim=-1)

        p_full = torch.cat((ppcd, cpcd), dim=1)

        fea_grid = self.scene_encoder(p_full)
        
        fea_grid = fea_grid['grid'].permute(0, 2, 3, 4, 1)
        flat_fea_grid = fea_grid.reshape(B, -1, self.aff_feat_dim)
        
        cpcd_mean_raster_index = three_util.coordinate2index(
            three_util.normalize_3d_coordinate(torch.mean(cpcd[:, :, :3], dim=1).reshape(B, -1, 3)),
            self.voxel_reso_grid,
            '3d').squeeze()

        cpcd_raster_index = three_util.coordinate2index(
            three_util.normalize_3d_coordinate(cpcd[:, :, :3].reshape(B, -1, 3)),
            self.voxel_reso_grid,
            '3d').squeeze()

        ppcd_raster_index = three_util.coordinate2index(
            three_util.normalize_3d_coordinate(ppcd[:, :, :3].reshape(B, -1, 3)),
            self.voxel_reso_grid,
            '3d').squeeze()

        full_pcd_raster_index = torch.cat((ppcd_raster_index, cpcd_raster_index), dim=-1)
        
        if global_feat:
            if global_feat_type == 'pc_pts':
                global_fea = flat_fea_grid.gather(dim=1, index=full_pcd_raster_index.reshape(B, -1, 1).repeat((1, 1, self.aff_feat_dim))).mean(1)
            else:
                global_fea = flat_fea_grid.mean(1)
        else:
            if mean_child:
                global_fea = flat_fea_grid.gather(dim=1, index=cpcd_mean_raster_index.reshape(B, -1, 1).repeat((1, 1, self.aff_feat_dim))).reshape(B, -1)
            else:
                global_fea = flat_fea_grid.gather(dim=1, index=cpcd_raster_index.reshape(B, -1, 1).repeat((1, 1, self.aff_feat_dim))).mean(1)

        # out_vox = self.affordance_head(fea_grid).reshape(B, -1)
        # top_voxel_inds = torch.max(out_vox, 1)[1]  # B x 1
        # index = top_voxel_inds.reshape(B, -1, 1).repeat((1, 1, self.aff_feat_dim))
        # global_fea = flat_fea_grid.gather(dim=1, index=index).reshape(B, -1)

        # fea_grid = fea_grid['grid'].permute(0, 2, 3, 4, 1).reshape(B, -1, self.aff_feat_dim)
        # global_fea = fea_grid.mean(1)
        
        out_rot = self.rot_affordance_head(global_fea)

        if self.softmax_out:
            # out = self.softmax(out)
            out_rot = self.softmax(out_rot)

        model_output = dict(
            rot_affordance=out_rot)
        #     child_mean_raster_index=cpcd_mean_raster_index
        # )

        return model_output

    def forward(self, x):
        bs = x.shape[0]
        x_rot = self.rot_affordance(x)
        x_vox = self.affordance(x)

        # out = torch.sigmoid(x).reshape(bs, -1)
        out_rot = x_rot.reshape(bs, -1)
        out_vox = x_vox.reshape(bs, -1)

        if self.softmax_out:
            # out = self.softmax(out)
            out_rot = self.softmax(out_rot)
            out_vox = self.softmax(out_vox)

        model_output = dict(
            voxel_affordance=out_vox,
            rot_affordance=out_rot
        )

        return model_output

