import torch
import torch.nn as nn

from rpdiff.model.transformer import nsm_transformer
from rpdiff.model.policy_feat_encoder import LocalAbstractPolicy, LocalAbstractSuccessClassifier
from rpdiff.utils import util
from rpdiff.utils.config_util import AttrDict
from rpdiff.utils.torch_util import maxpool, meanpool, SinusoidalPosEmb

import trimesh
import numpy as np
from meshcat import Visualizer
from typing import List, Union, Tuple


class NSMTransformerSingleTransformationRegression(LocalAbstractPolicy):
    def __init__(self, 
                 feat_dim: int, 
                 in_dim: int=3, 
                 hidden_dim: int=256, 
                 n_heads: int=4, 
                 drop_p: float=0.0, 
                 n_blocks: int=2, 
                 n_pts: int=64, pn_pts: int=None, cn_pts: int=None, 
                 n_queries: int=2, 
                 pooling: str='mean', 
                 predict_offset: bool=False, 
                 bidir: bool=False, 
                 use_timestep_emb: bool=False, 
                 max_timestep: int=None, 
                 timestep_pool_method: str='meanpool',
                 mc_vis: Visualizer=None):
        """
        Args:
            feat_dim (int): Dimensionality of the input features (typically 3D for point clouds, or 5D for point clouds A and B - with one-hot)
            in_dim (int): Dimensionality of the first layer of our policy, after encoding
                the point cloud and the point cloud features (from the potentially pre-trained encoder).
                The very first operation is to project 3D -> in_dim/2 and feat_dim -> in_dim/2
                and concatenate these to provide to our main module
            hidden_dim (int): Internal dimensionality of our main modules
            n_heads (int): Number of heads for multi-head self-attention
            drop_p (float): Between 0.0 and 1.0, probability of dropout
            n_blocks (int): Number of multi-head self-attention blocks to use in the transformer
            n_pts (int): Number of points to downsample to, for each shape (so total number of
                points will be 2*n_pts)
            pn_pts (int): Number of points to downsample parent/scene to
            cn_pts (int): Number of points to downsample child/object to
            n_queries (int): Number of query tokens to use for output (unused currently, deafult 1 output)
            pooling (str): 'mean' or 'max', for how we pool the output features from the transformer
            bidir (bool): If True, compute object-scene cross attention in both directions
            use_timestep_emb (bool): If True, also condition on timestep/iteration embedding
            max_timestep (int): Value to clip the maximum timestep to
            timestep_pool_method (str): 'meanpool' or 'concat'
            mc_vis (Visualizer): Meshcat interface for debugging visualization
        """
        super().__init__(n_pts=n_pts, pn_pts=pn_pts, cn_pts=cn_pts)
        self.hidden_dim = hidden_dim
        self.in_dim = hidden_dim
        self.n_pts = n_pts
        self.pn_pts = pn_pts
        self.cn_pts = cn_pts
        self.mc_vis = mc_vis
        
        # input projections
        self.pcd_emb = nn.Linear(3, int(self.in_dim/2))
        self.des_emb = nn.Linear(feat_dim, int(self.in_dim/2))

        # one query per output transformation
        self.n_queries = n_queries
        self.bidir = bidir

        cfg = AttrDict(dict())
        cfg.model = AttrDict(dict())
        cfg.model.num_heads = n_heads
        cfg.model.num_blocks = n_blocks
        cfg.model.pc_feat_dim = self.in_dim + 2
        # cfg.model.pc_feat_dim = 3
        cfg.model.transformer_feat_dim = hidden_dim 

        self.transformer = nsm_transformer.Transformer(cfg)

        self.set_predict_offset(predict_offset)
        
        # 2*hidden_dim because we combine the pooled output feats with the global 'cls' token feat
        # self._build_output_heads(pooled_dim=2*hidden_dim, hidden_dim=hidden_dim)
        # self._build_output_heads(pooled_dim=2*(hidden_dim + 2), hidden_dim=hidden_dim)
        self._build_output_heads(pooled_dim=hidden_dim + 2, hidden_dim=hidden_dim)

        self.pool = meanpool if pooling == 'mean' else maxpool
        self.perm = None

        self.per_point_h = None
        
        self.pos_emb_max_timestep = max_timestep
        self.pos_emb = SinusoidalPosEmb(dim=self.hidden_dim + 2, max_pos=self.pos_emb_max_timestep)
        # self.timestep_emb_proj = nn.Sequential(nn.Linear(hidden_dim+2, hidden_dim+2), nn.LayerNorm(hidden_dim+2))
        self.timestep_emb_proj = nn.Linear(hidden_dim+2, hidden_dim+2)
        # self.timestep_emb_proj = nn.Sequential(nn.Linear(hidden_dim+2, hidden_dim+2), nn.ReLU(), nn.Linear(hidden_dim+2, hidden_dim+2))
        
        self.use_timestep_emb = use_timestep_emb
        if timestep_pool_method == 'meanpool':
            self.pool_with_var = self.pool_with_var_meanpool
        elif timestep_pool_method == 'concat':
            self.pool_with_var = self.pool_with_var_concat

            # we have to re-build the output heads in this case, since we are concatenating (dim will be double)
            if self.use_timestep_emb:
                self._build_output_heads(pooled_dim=2*(hidden_dim + 2), hidden_dim=hidden_dim)
        else:
            pass

    def set_pos_emb_max_timestep(self, max_timestep: int):
        self.pos_emb_max_timestep = max_timestep
        self.pos_emb = SinusoidalPosEmb(dim=self.hidden_dim + 2, max_pos=self.pos_emb_max_timestep)

    def pool_with_var(self, *args, **kwargs):
        return

    def pool_with_var_meanpool(self, pooled_h: torch.FloatTensor, 
                               new_var: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        out_pooled_h = torch.mean(torch.stack(
            [
                pooled_h, 
                new_var
            ], 1), 1)
        return out_pooled_h

    def pool_with_var_concat(self, pooled_h: torch.FloatTensor, 
                             new_var: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        out_pooled_h = torch.cat((pooled_h, new_var), dim=-1)
        return out_pooled_h

    def forward(self, model_input: dict, *args, **kwargs) -> dict:
        n_pts = self.n_pts

        # centerd 3D point clouds
        ppcd_cent = model_input['parent_start_pcd']
        cpcd_cent = model_input['child_start_pcd']
        B, N = ppcd_cent.shape[0], ppcd_cent.shape[1]
        Np = ppcd_cent.shape[1]
        Nc = cpcd_cent.shape[1]

        # process inputs
        des_dict, pcd_dict = self.process_model_input(model_input)
        ppcd, cpcd = pcd_dict['parent'], pcd_dict['child']
        des_full_parent_coords, des_full_child_coords = des_dict['parent'], des_dict['child']
        
        # downsample with random permutation (voxel or fps downsampling would be better...)
        ppcd, perm1 = self.fps_ds_p.forward(ppcd, return_idx=True)
        cpcd, perm2 = self.fps_ds_c.forward(cpcd, return_idx=True)
        pcd_full = torch.cat([ppcd, cpcd], dim=1)

        # # can uncomment here to visualize point clouds inside forward pass (debugging)
        # util.meshcat_pcd_show(self.mc_vis, ppcd[0].detach().cpu().numpy(), (255, 0, 0), 'scene/process/ppcd_perm')
        # util.meshcat_pcd_show(self.mc_vis, cpcd[0].detach().cpu().numpy(), (0, 0, 255), 'scene/process/cpcd_perm')
        # import trimesh
        # sz = 0.05
        # for i, pt in enumerate(ppcd[0]):
        #     sph = trimesh.creation.uv_sphere(sz).apply_translation(pt.detach().cpu().numpy())
        #     util.meshcat_trimesh_show(self.mc_vis, f'scene/process/sph/ppcd_perm_sph_{i}', sph, (255, 0, 0))

        # for i, pt in enumerate(cpcd[0]):
        #     sph = trimesh.creation.uv_sphere(sz).apply_translation(pt.detach().cpu().numpy())
        #     util.meshcat_trimesh_show(self.mc_vis, f'scene/process/sph/cpcd_perm_sph_{i}', sph, (0, 0, 255))

        # print('here after getting des and pcd (perm)')
        # from IPython import embed; embed()

        des_full_parent_coords = torch.gather(des_full_parent_coords, dim=1, index=perm1[:, :, None].repeat((1, 1, des_full_parent_coords.shape[-1])))
        des_full_child_coords = torch.gather(des_full_child_coords, dim=1, index=perm2[:, :, None].repeat((1, 1, des_full_child_coords.shape[-1])))
        des_full = torch.cat([des_full_parent_coords, des_full_child_coords], dim=1)

        # print(f'des_full.shape: {des_full.shape}, pcd_full.shape: {pcd_full.shape}')

        if 'timestep_emb' in model_input and self.use_timestep_emb:
            des_full_child_coords = torch.cat((des_full_child_coords, model_input['timestep_emb'].reshape(B, 1, -1)), dim=1)

        # process per-point features and pool
        # per_point_h1 = self.transformer(des_full_parent_coords.transpose(2, 1), des_full_child_coords.transpose(2, 1)).transpose(2, 1)
        per_point_h2 = self.transformer(des_full_child_coords.transpose(2, 1), des_full_parent_coords.transpose(2, 1)).transpose(2, 1)
        # per_point_h = self.transformer(des_full, pcd_full)

        # per_point_h = torch.cat((per_point_h1, per_point_h2), dim=-1)
        per_point_h = per_point_h2

        pooled_h = self.pool(per_point_h, dim=1, keepdim=True)  # B x 1 x D
        
        if 'timestep_emb' in model_input and self.use_timestep_emb:
            pooled_h = self.pool_with_var(
                pooled_h, 
                self.timestep_emb_proj(model_input['timestep_emb']).reshape(B, 1, -1)
            )

        # process model output predictions
        model_output = self.process_model_output(pooled_h)

        if 'save_attn' in kwargs:
            self.ppcd_ds = ppcd
            self.cpcd_ds = cpcd
            self.pdes_ds = des_full_parent_coords
            self.cdes_ds = des_full_child_coords

        return model_output


class NSMTransformerSingleTransformationRegressionCVAE(LocalAbstractPolicy):
    def __init__(self, 
                 feat_dim: int, 
                 in_dim: int=3, 
                 hidden_dim: int=256, 
                 n_heads: int=4, 
                 drop_p: float=0.0, 
                 n_blocks: int=2, 
                 n_pts: int=64, pn_pts: int=None, cn_pts: int=None, 
                 n_queries: int=2, 
                 pooling: str='mean', 
                 predict_offset: bool=False, 
                 bidir: bool=False, 
                 latent_dim: int=None, 
                 residual_latent: bool=False, 
                 residual_tf_enc: bool=False, 
                 tf_pool_method: str='meanpool', 
                 latent_pool_method: str='meanpool', 
                 mc_vis: Visualizer=None):
        """
        Args:
            feat_dim (int): Dimensionality of the input features (typically 3D for point clouds, or 5D for point clouds A and B - with one-hot)
            in_dim (int): Dimensionality of the first layer of our policy, after encoding
                the point cloud and the point cloud features (from the potentially pre-trained encoder).
                The very first operation is to project 3D -> in_dim/2 and feat_dim -> in_dim/2
                and concatenate these to provide to our main module
            hidden_dim (int): Internal dimensionality of our main modules
            n_heads (int): Number of heads for multi-head self-attention
            drop_p (float): Between 0.0 and 1.0, probability of dropout
            n_blocks (int): Number of multi-head self-attention blocks to use in the transformer
            n_pts (int): Number of points to downsample to, for each shape (so total number of
                points will be 2*n_pts)
            pn_pts (int): Number of points to downsample parent/scene to
            cn_pts (int): Number of points to downsample child/object to
            n_queries (int): Number of query tokens to use for output (unused currently, deafult 1 output)
            pooling (str): 'mean' or 'max', for how we pool the output features from the transformer
            bidir (bool): If True, compute object-scene cross attention in both directions
            latent_dim (int): Dimensionality of CVAE latent space
            residual_latent (bool): If True, incorporate latent via residual connection in decoder
            residual_tf_enc (bool): If True, incorporate transform via residual connection in encoder
            tf_pool_method (str): 'meanpool' or 'concat'
            latent_pool_method (str): 'meanpool' or 'concat'
            mc_vis (Visualizer): Meshcat interface for debugging viz
        """
        super().__init__(n_pts=n_pts, pn_pts=pn_pts, cn_pts=cn_pts)
        self.hidden_dim = hidden_dim
        self.in_dim = hidden_dim
        self.n_pts = n_pts
        self.pn_pts = pn_pts
        self.cn_pts = cn_pts
        self.mc_vis = mc_vis
        
        # input projections
        self.pcd_emb = nn.Linear(3, int(self.in_dim/2))
        self.des_emb = nn.Linear(feat_dim, int(self.in_dim/2))

        # one query per output transformation
        self.n_queries = n_queries
        self.bidir = bidir

        cfg = AttrDict(dict())
        cfg.model = AttrDict(dict())
        cfg.model.num_heads = n_heads
        cfg.model.num_blocks = n_blocks
        cfg.model.pc_feat_dim = self.in_dim + 2
        # cfg.model.pc_feat_dim = 3
        cfg.model.transformer_feat_dim = hidden_dim 

        # self.transformer = nsm_transformer.Transformer(cfg)
        self.transformer_enc = nsm_transformer.Transformer(cfg)
        self.transformer_dec = nsm_transformer.Transformer(cfg)

        if latent_dim is None:
            enc_mlp_dim = hidden_dim + 2
        else:
            enc_mlp_dim = latent_dim
        
        enc_mlp_dim = hidden_dim + 2
        self.tf_proj = nn.Linear(7, enc_mlp_dim)
        # self.tf_proj = nn.Sequential(nn.Linear(7, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 
        self.enc_mu_head = nn.Sequential(nn.Linear(enc_mlp_dim, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 
        self.enc_logvar_head = nn.Sequential(nn.Linear(enc_mlp_dim, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 

        # self.enc_mu_head_rot = nn.Sequential(nn.Linear(enc_mlp_dim, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 
        # self.enc_logvar_head_rot = nn.Sequential(nn.Linear(enc_mlp_dim, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 
        # self.rot_proj = nn.Linear(4, enc_mlp_dim)
        # self.enc_mu_head_trans = nn.Sequential(nn.Linear(enc_mlp_dim, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 
        # self.enc_logvar_head_trans = nn.Sequential(nn.Linear(enc_mlp_dim, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 
        # self.trans_proj = nn.Linear(3, enc_mlp_dim)

        self.residual_latent = residual_latent
        self.residual_tf_enc = residual_tf_enc

        self.set_predict_offset(predict_offset)
        
        # 2*hidden_dim because we combine the pooled output feats with the global 'cls' token feat
        # self._build_output_heads(pooled_dim=2*hidden_dim, hidden_dim=hidden_dim)
        # self._build_output_heads(pooled_dim=2*(hidden_dim + 2), hidden_dim=hidden_dim)
        self._build_output_heads(pooled_dim=hidden_dim + 2, hidden_dim=hidden_dim)

        self.pool = meanpool if pooling == 'mean' else maxpool
        self.perm = None

        self.per_point_h = None

        self.eval_sample = False  # if True, will NOT encode during forward -- instead, will just sample z latent from unimodal Gaussian

        if tf_pool_method == 'meanpool':
            assert enc_mlp_dim == self.hidden_dim + 2, 'Dimensions must match for meanpool!'
            self.pool_tf = self.pool_with_var_meanpool
        elif tf_pool_method == 'concat':
            self.pool_tf = self.pool_with_var_concat
            
            # have to rebuild the encoder heads, since dim will be double
            # self.enc_mu_head = nn.Sequential(nn.Linear(2*enc_mlp_dim, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 
            # self.enc_logvar_head = nn.Sequential(nn.Linear(2*enc_mlp_dim, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 
            self.enc_mu_head = nn.Sequential(nn.Linear(hidden_dim + 2 + enc_mlp_dim, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 
            self.enc_logvar_head = nn.Sequential(nn.Linear(hidden_dim + 2 + enc_mlp_dim, enc_mlp_dim, enc_mlp_dim), nn.ReLU(), nn.Linear(enc_mlp_dim, enc_mlp_dim)) 
        else:
            pass

        if latent_pool_method == 'meanpool':
            assert enc_mlp_dim == self.hidden_dim + 2, 'Dimensions must match for meanpool!'
            self.pool_latent = self.pool_with_var_meanpool
        elif latent_pool_method == 'concat':
            self.pool_latent = self.pool_with_var_concat
            
            # have to rebuild the output heads, since dim will be double
            if self.residual_latent:
                self._build_output_heads(pooled_dim=hidden_dim + 2 + enc_mlp_dim, hidden_dim=hidden_dim)
                # self._build_output_heads(pooled_dim=2*(hidden_dim + 2), hidden_dim=hidden_dim)
        else:
            pass

    def set_eval_sample(self, eval_sample: bool):
        self.eval_sample = eval_sample

    def pool_latent(self, *args, **kwargs):
        return

    def pool_tf(self, *args, **kwargs):
        return

    def pool_with_var_meanpool(self, pooled_h: torch.FloatTensor, 
                               new_var: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        out_pooled_h = torch.mean(torch.stack(
            [
                pooled_h, 
                new_var
            ], 1), 1)
        return out_pooled_h

    def pool_with_var_concat(self, pooled_h: torch.FloatTensor, 
                             new_var: torch.FloatTensor, *args, **kwargs) -> torch.FloatTensor:
        out_pooled_h = torch.cat((pooled_h, new_var), dim=-1)
        return out_pooled_h

    def get_pp_feats(self, model_input: dict, *args, **kwargs) -> Tuple[torch.FloatTensor]:
        n_pts = self.n_pts

        # centerd 3D point clouds
        ppcd_cent = model_input['parent_start_pcd']
        cpcd_cent = model_input['child_start_pcd']
        B, N = ppcd_cent.shape[0], ppcd_cent.shape[1]
        Np = ppcd_cent.shape[1]
        Nc = cpcd_cent.shape[1]

        # process inputs
        des_dict, pcd_dict = self.process_model_input(model_input)
        ppcd, cpcd = pcd_dict['parent'], pcd_dict['child']
        des_full_parent_coords, des_full_child_coords = des_dict['parent'], des_dict['child']
        
        # downsample with random permutation (voxel or fps downsampling would be better...)
        ppcd, perm1 = self.fps_ds_p.forward(ppcd, return_idx=True)
        cpcd, perm2 = self.fps_ds_c.forward(cpcd, return_idx=True)
        pcd_full = torch.cat([ppcd, cpcd], dim=1)

        # util.meshcat_pcd_show(self.mc_vis, ppcd[0].detach().cpu().numpy(), (255, 0, 0), 'scene/process/ppcd_perm')
        # util.meshcat_pcd_show(self.mc_vis, cpcd[0].detach().cpu().numpy(), (0, 0, 255), 'scene/process/cpcd_perm')
        # import trimesh
        # sz = 0.05
        # for i, pt in enumerate(ppcd[0]):
        #     sph = trimesh.creation.uv_sphere(sz).apply_translation(pt.detach().cpu().numpy())
        #     util.meshcat_trimesh_show(self.mc_vis, f'scene/process/sph/ppcd_perm_sph_{i}', sph, (255, 0, 0))

        # for i, pt in enumerate(cpcd[0]):
        #     sph = trimesh.creation.uv_sphere(sz).apply_translation(pt.detach().cpu().numpy())
        #     util.meshcat_trimesh_show(self.mc_vis, f'scene/process/sph/cpcd_perm_sph_{i}', sph, (0, 0, 255))

        # print('here after getting des and pcd (perm)')
        # from IPython import embed; embed()

        des_full_parent_coords = torch.gather(des_full_parent_coords, dim=1, index=perm1[:, :, None].repeat((1, 1, des_full_parent_coords.shape[-1])))
        des_full_child_coords = torch.gather(des_full_child_coords, dim=1, index=perm2[:, :, None].repeat((1, 1, des_full_child_coords.shape[-1])))
        des_full = torch.cat([des_full_parent_coords, des_full_child_coords], dim=1)

        return des_full_parent_coords, des_full_child_coords

    def reparameterize(self, mu: torch.FloatTensor, logvar: torch.FloatTensor) -> torch.FloatTensor:
        std_dev = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std_dev)
        z = mu + std_dev * eps
        return z

    def encode(self, model_input: dict, 
               des_full_parent_coords: torch.FloatTensor, des_full_child_coords: torch.FloatTensor, 
               *args, **kwargs) -> dict:
        ppcd_cent = model_input['parent_start_pcd']
        cpcd_cent = model_input['child_start_pcd']
        B = ppcd_cent.shape[0]
        Np = ppcd_cent.shape[1]
        Nc = cpcd_cent.shape[1]

        # if 'rot_trans' in model_input.keys():
        #     if model_input['rot_trans'] = 'rot':
        #         enc_mu_head = self.enc_mu_head_rot
        #         enc_logvar_head = self.enc_logvar_head_rot
        #         tf_proj = self.rot_proj
        #         tf_to_proj = model_input['tf'][:, 3:]
        #     elif model_input['rot_trans'] = 'trans':
        #         enc_mu_head = self.enc_mu_head_trans
        #         enc_logvar_head = self.enc_logvar_head_trans
        #         tf_proj = self.trans_proj
        #         tf_to_proj = model_input['tf'][:, :3]
        #     else:
        #         pass

        # concatenate the latent variable into the point set
        tf_emb = self.tf_proj(model_input['tf'])  # translation + quaternion
        # tf_emb = tf_proj(tf_to_proj)  # translation + quaternion
        des_full_child_coords = torch.cat((des_full_child_coords, tf_emb.reshape(B, 1, -1)), dim=1)
        # des_full_parent_coords = torch.cat((des_full_parent_coords, tf_emb.reshape(B, 1, -1)), dim=-1)

        per_point_h = self.transformer_enc(des_full_child_coords.transpose(2, 1), des_full_parent_coords.transpose(2, 1)).transpose(2, 1)
        pooled_h = self.pool(per_point_h, dim=1, keepdim=True)  # B x 1 x D

        if self.residual_tf_enc:
            pooled_h = self.pool_tf(
                pooled_h, 
                tf_emb.reshape(B, 1, -1)
            )

        z_mu = self.enc_mu_head(pooled_h)
        z_logvar = self.enc_logvar_head(pooled_h)
        # z_mu = enc_mu_head(pooled_h)
        # z_logvar = enc_logvar_head(pooled_h)

        z = self.reparameterize(z_mu, z_logvar)

        model_output = {}
        model_output['z'] = z
        model_output['z_mu'] = z_mu
        model_output['z_logvar'] = z_logvar

        # if 'rot_trans' in model_input.keys():
        #     if model_input['rot_trans'] = 'rot':
        #         model_output['z_mu_rot'] = z_mu
        #         model_output['z_logvar_rot'] = z_logvar
        #     elif model_input['rot_trans'] = 'trans':
        #         model_output['z_mu_trans'] = z_mu
        #         model_output['z_logvar_trans'] = z_logvar
        #     else:
        #         pass

        return model_output

    def decode(self, model_input: dict, encoder_output: dict, 
               des_full_parent_coords: torch.FloatTensor, des_full_child_coords: torch.FloatTensor, 
               *args, **kwargs) -> dict:
        ppcd_cent = model_input['parent_start_pcd']
        cpcd_cent = model_input['child_start_pcd']
        B = ppcd_cent.shape[0]
        Np = ppcd_cent.shape[1]
        Nc = cpcd_cent.shape[1]

        # concatenate the latent variable into the point set
        z = encoder_output['z'] 
        des_full_child_coords = torch.cat((des_full_child_coords, z.reshape(B, 1, -1)), dim=1)
        # des_full_parent_coords = torch.cat((des_full_parent_coords, z.reshape(B, 1, -1)), dim=-1)

        per_point_h = self.transformer_dec(des_full_child_coords.transpose(2, 1), des_full_parent_coords.transpose(2, 1)).transpose(2, 1)
        pooled_h = self.pool(per_point_h, dim=1, keepdim=True)  # B x 1 x D

        if self.residual_latent:
            pooled_h = self.pool_latent(
                pooled_h, 
                z.reshape(B, 1, -1)
            )

        # process model output predictions
        model_output = self.process_model_output(pooled_h)

        for k, v in encoder_output.items():
            model_output[k] = v

        return model_output

    def forward(self, model_input: dict, *args, **kwargs) -> dict:

        des_full_parent_coords, des_full_child_coords = self.get_pp_feats(model_input)
        
        if self.eval_sample:
            B = des_full_parent_coords.shape[0]
            h_dim = des_full_parent_coords.shape[-1]
            encoder_output = {}
            encoder_output['z'] = torch.randn((B, h_dim)).float().cuda()
        else:
            encoder_output = self.encode(model_input, des_full_parent_coords, des_full_child_coords)
        model_output = self.decode(model_input, encoder_output, des_full_parent_coords, des_full_child_coords)

        return model_output


class NSMTransformerSingleSuccessClassifier(LocalAbstractSuccessClassifier):
    def __init__(self, 
                 feat_dim: int, 
                 in_dim: int=3, 
                 hidden_dim: int=256, 
                 n_heads: int=4, 
                 drop_p: float=0.0, 
                 n_blocks: int=2, 
                 n_pts: int=64, pn_pts: int=None, cn_pts: int=None, 
                 pooling: str='mean', 
                 sigmoid: bool=True, 
                 fixed_scaling: float=None, 
                 bidir: bool=False, 
                 n_queries: int=1,
                 mc_vis: Visualizer=None):
        """
        Args:
            feat_dim (int): Dimensionality of the input features (typically 3D for point clouds, or 5D for point clouds A and B - with one-hot)
            in_dim (int): Dimensionality of the first layer of our policy, after encoding
                the point cloud and the point cloud features (from the potentially pre-trained encoder).
                The very first operation is to project 3D -> in_dim/2 and feat_dim -> in_dim/2
                and concatenate these to provide to our main module
            hidden_dim (int): Internal dimensionality of our main modules
            n_heads (int): Number of heads for multi-head self-attention
            drop_p (float): Between 0.0 and 1.0, probability of dropout
            n_blocks (int): Number of multi-head self-attention blocks to use in the transformer
            n_pts (int): Number of points to downsample to, for each shape (so total number of
                points will be 2*n_pts)
            pn_pts (int): Number of points to downsample parent/scene to
            cn_pts (int): Number of points to downsample child/object to
            pooling (str): 'mean' or 'max', for how we pool the output features from the transformer
            sigmoid (bool): If True, pass through a final sigmoid at the output
            fixed_scaling (float): Manually scale the objects based on pre-computed scale value
            bidir (bool): If True, compute object-scene cross attention in both directions
            n_queries (int): Number of query tokens to use for output (unused currently, deafult 1 output)
            mc_vis (Visualizer): Meshcat interface for debugging visualization
        """
        super().__init__(n_pts=n_pts, pn_pts=pn_pts, cn_pts=cn_pts, sigmoid=sigmoid)
        self.hidden_dim = hidden_dim
        self.in_dim = hidden_dim
        self.n_pts = n_pts
        self.pn_pts = pn_pts
        self.cn_pts = cn_pts
        self.mc_vis = mc_vis
        
        # input projections
        self.pcd_emb = nn.Linear(3, int(self.in_dim/2))
        self.des_emb = nn.Linear(feat_dim, int(self.in_dim/2))

        self.bidir = bidir

        cfg = AttrDict(dict())
        cfg.model = AttrDict(dict())
        cfg.model.num_heads = n_heads
        cfg.model.num_blocks = n_blocks
        cfg.model.pc_feat_dim = self.in_dim + 2
        # cfg.model.pc_feat_dim = 3
        cfg.model.transformer_feat_dim = hidden_dim 

        self.transformer = nsm_transformer.Transformer(cfg)

        # 2*hidden_dim because we combine the pooled output feats with the global 'cls' token feat
        # self._build_output_heads(pooled_dim=2*hidden_dim, hidden_dim=hidden_dim)
        # self._build_output_heads(pooled_dim=2*(hidden_dim + 2), hidden_dim=hidden_dim)
        
        if self.bidir:
            self._build_output_heads(pooled_dim=2*(hidden_dim + 2), hidden_dim=hidden_dim)
        else:
            self._build_output_heads(pooled_dim=hidden_dim + 2, hidden_dim=hidden_dim)

        self.pool = meanpool if pooling == 'mean' else maxpool
        self.perm = None

        self.per_point_h = None

    def forward(self, model_input: dict, *args, **kwargs) -> dict:
        n_pts = self.n_pts

        # centerd 3D point clouds
        ppcd_cent = model_input['parent_start_pcd']
        cpcd_cent = model_input['child_start_pcd']
        B, N = ppcd_cent.shape[0], ppcd_cent.shape[1]
        Np = ppcd_cent.shape[1]
        Nc = cpcd_cent.shape[1]

        # process inputs
        des_dict, pcd_dict = self.process_model_input(model_input)
        ppcd, cpcd = pcd_dict['parent'], pcd_dict['child']
        des_full_parent_coords, des_full_child_coords = des_dict['parent'], des_dict['child']
        
        # downsample with random permutation (voxel or fps downsampling would be better...)
        ppcd, perm1 = self.fps_ds_p.forward(ppcd, return_idx=True)
        cpcd, perm2 = self.fps_ds_c.forward(cpcd, return_idx=True)
        pcd_full = torch.cat([ppcd, cpcd], dim=1)

        # util.meshcat_pcd_show(self.mc_vis, ppcd[0].detach().cpu().numpy(), (255, 0, 0), 'scene/process/ppcd_perm')
        # util.meshcat_pcd_show(self.mc_vis, cpcd[0].detach().cpu().numpy(), (0, 0, 255), 'scene/process/cpcd_perm')
        # import trimesh
        # sz = 0.05
        # for i, pt in enumerate(ppcd[0]):
        #     sph = trimesh.creation.uv_sphere(sz).apply_translation(pt.detach().cpu().numpy())
        #     util.meshcat_trimesh_show(self.mc_vis, f'scene/process/sph/ppcd_perm_sph_{i}', sph, (255, 0, 0))

        # for i, pt in enumerate(cpcd[0]):
        #     sph = trimesh.creation.uv_sphere(sz).apply_translation(pt.detach().cpu().numpy())
        #     util.meshcat_trimesh_show(self.mc_vis, f'scene/process/sph/cpcd_perm_sph_{i}', sph, (0, 0, 255))

        # print('here after getting des and pcd (perm)')
        # from IPython import embed; embed()

        des_full_parent_coords = torch.gather(des_full_parent_coords, dim=1, index=perm1[:, :, None].repeat((1, 1, des_full_parent_coords.shape[-1])))
        des_full_child_coords = torch.gather(des_full_child_coords, dim=1, index=perm2[:, :, None].repeat((1, 1, des_full_child_coords.shape[-1])))
        des_full = torch.cat([des_full_parent_coords, des_full_child_coords], dim=1)

        # print(f'des_full.shape: {des_full.shape}, pcd_full.shape: {pcd_full.shape}')
        
        per_point_h2 = self.transformer(des_full_child_coords.transpose(2, 1), des_full_parent_coords.transpose(2, 1)).transpose(2, 1)
        if self.bidir:
        # process per-point features and pool
            per_point_h1 = self.transformer(des_full_parent_coords.transpose(2, 1), des_full_child_coords.transpose(2, 1)).transpose(2, 1)
        # per_point_h = self.transformer(des_full, pcd_full)

        if self.bidir:
            per_point_h = torch.cat((per_point_h1, per_point_h2), dim=-1)
        else:
            per_point_h = per_point_h2

        pooled_h = self.pool(per_point_h, dim=1, keepdim=True)  # B x 1 x D

        # process model output predictions
        model_output = self.process_model_output(pooled_h)

        return model_output
