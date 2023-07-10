import torch
import torch.nn as nn
import time
import numpy as np

from trimesh import creation as trimesh_creation

from rpdiff.utils import util
from rpdiff.utils.torch3d_util import matrix_to_quaternion, matrix_to_euler_angles

from rpdiff.training.train_util import adjust_learning_rate, get_grad_norm, euler_disc_pred_to_rotmat, get_linear_warmup_cosine_decay_sched
from rpdiff.training.pred_util import get_rotation_grid_index_torch, get_euler_onehot_torch

from rpdiff.utils.config_util import AttrDict
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from meshcat import Visualizer
from typing import List, Tuple, Union, Callable

MC_SIZE=0.007


def train_iter_coarse_aff(
            coarse_aff_mi: dict,
            coarse_aff_gt: dict,
            coarse_aff_model: nn.Module,
            aff_optimizer: Optimizer,
            aff_loss_fn: Callable,
            args: AttrDict,
            voxel_grid_pts: torch.Tensor, reso_grid: int,
            rot_mat_grid: torch.Tensor, rot_grid_bins: int,
            it: int, current_epoch: float,
            logger: SummaryWriter, refine_pred: bool=False,
            mc_vis: Visualizer=None):

    # setup visualization, clear from last
    mc_iter_name = 'scene/train_iter_aff'
    if mc_vis is not None:
        mc_vis[mc_iter_name].delete()
    start_time = time.time()

    bs = coarse_aff_mi['parent_start_pcd_sn'].shape[0]
    n_child_pts = coarse_aff_mi['child_start_pcd_sn'].shape[1]
    db_idx = 0

    #######################################################################
    # Encode parent voxel grid (scene-normalized)

    if args.loss.coarse_aff.rot_only_on_refine:
        if refine_pred:
            parent_feature_grid = coarse_aff_model.encode_scene_fea_grid(
                p=coarse_aff_mi['parent_start_pcd_sn'],
                c=coarse_aff_mi['child_start_pcd_sn'])
        else:
            parent_feature_grid = coarse_aff_model.encode_scene_fea_grid(coarse_aff_mi['parent_start_pcd_sn'])
    else:
        parent_feature_grid = coarse_aff_model.encode_scene_fea_grid(coarse_aff_mi['parent_start_pcd_sn'])
    parent_feature_grid = parent_feature_grid['grid'].permute(0, 2, 3, 4, 1)  # features in last dimension

    # parent_feature_grid = coarse_aff_model.encode_scene_fea_grid(coarse_aff_mi['parent_start_pcd_sn'])
    # parent_feature_grid = parent_feature_grid['grid'].permute(0, 2, 3, 4, 1)  # features in last dimension

    if args.debug:
        # visualize the scene-normalized input parent point cloud
        util.meshcat_pcd_show(
            mc_vis,
            coarse_aff_mi['parent_start_pcd_sn'][db_idx].detach().cpu().numpy(),
            (255, 0, 0),
            f'{mc_iter_name}/pre_voxel_aff/parent_start_pcd_sn',
            size=MC_SIZE
        )
        util.meshcat_pcd_show(
            mc_vis,
            coarse_aff_mi['child_start_pcd_sn'][db_idx].detach().cpu().numpy(),
            (0, 0, 255),
            f'{mc_iter_name}/pre_voxel_aff/child_start_pcd_sn',
            size=MC_SIZE
        )

    # predict best voxel position
    voxel_aff_model_output = coarse_aff_model.voxel_affordance(parent_feature_grid)

    #######################################################################
    # Apply this prediction to the child objects
    top_voxel_inds = torch.max(voxel_aff_model_output['voxel_affordance'], 1)[1]  # B x 1
    top_voxel_coord_sn = voxel_grid_pts[top_voxel_inds]  # B x 3, scene-normalized (unit-cube)

    if args.debug:
        # visualize the top-k voxels from affordance model
        vals, inds = torch.topk(voxel_aff_model_output['voxel_affordance'][db_idx], k=10)
        sm = nn.Softmax(dim=0)
        sm_vals = sm(voxel_aff_model_output['voxel_affordance'][db_idx])

        # vals, inds = torch.topk(latent_norm, k=100)
        # vals, inds = torch.topk(-1.0*latent_norm, k=100)

        # show the raster points for reference
        util.meshcat_pcd_show(
            mc_vis,
            voxel_grid_pts.detach().cpu().numpy(),
            (0, 0, 0),
            f'{mc_iter_name}/post_voxel_aff/voxel_grid_pts',
            size=MC_SIZE/3.0
        )

        mc_vis['scene/voxel_grid'].delete()
        sz_base = 1.1/reso_grid
        for idx, i in enumerate(inds):
            pt = voxel_grid_pts.detach().cpu().numpy()[i]
            aff_value = voxel_aff_model_output['voxel_affordance'][0][i]
            box = trimesh_creation.box([sz_base]*3).apply_translation(pt)
            print(f'Softmax value: {sm_vals[i]}')
            util.meshcat_trimesh_show(mc_vis, f'{mc_iter_name}/post_voxel_aff/{i}', box, opacity=0.3)

        # show this with the ground truth best voxel
        util.meshcat_pcd_show(
            mc_vis,
            coarse_aff_gt['child_centroid_voxel_coord'][db_idx].detach().cpu().numpy().reshape(1, 3),
            (0, 255, 0),
            f'{mc_iter_name}/post_voxel_aff/ground_truth_voxel_coord',
            size=MC_SIZE*5
        )

    # translate the child point cloud to this position
    # voxel_position = torch.from_numpy(top_voxel_coord_sn).float().cuda()
    # child_pcd_delta = voxel_position - coarse_aff_mi['child_start_pcd_mean_sn']  # B x 3
    child_pcd_delta = top_voxel_coord_sn - coarse_aff_mi['child_start_pcd_mean_sn']  # B x 3
    child_pcd_delta = child_pcd_delta.reshape(bs, 1, 3).repeat((1, n_child_pts, 1)) 

    child_start_pcd_sn_post_trans = coarse_aff_mi['child_start_pcd_sn'] + child_pcd_delta
    parent_start_pcd_sn_post_trans = coarse_aff_mi['parent_start_pcd_sn']

    coarse_aff_rot_mi = dict(
        parent_start_pcd_sn=parent_start_pcd_sn_post_trans,
        child_start_pcd_sn=child_start_pcd_sn_post_trans)

    #######################################################################
    # Re-encode new joint parent/child point cloud and predict best rotation

    if args.debug:
        # visualize the parent and child point clouds before passing to rot affordance
        util.meshcat_pcd_show(
            mc_vis,
            coarse_aff_rot_mi['parent_start_pcd_sn'][db_idx].detach().cpu().numpy(),
            (255, 0, 0),
            f'{mc_iter_name}/pre_rot_aff/parent_start_pcd_sn',
            size=MC_SIZE
        )
        util.meshcat_pcd_show(
            mc_vis,
            coarse_aff_rot_mi['child_start_pcd_sn'][db_idx].detach().cpu().numpy(),
            (0, 0, 255),
            f'{mc_iter_name}/pre_rot_aff/child_start_pcd_sn',
            size=MC_SIZE*1.5
        )

    # rot_aff_model_output = coarse_aff_model.rot_affordance(coarse_aff_rot_mi, mean_child=args.model.coarse_aff.rot_feat_mean_child)

    if args.loss.coarse_aff.rot_only_on_refine:
        if refine_pred:
            rot_aff_model_output = coarse_aff_model.rot_affordance(
                coarse_aff_rot_mi, 
                mean_child=args.model.coarse_aff.rot_feat_mean_child, 
                global_feat=args.model.coarse_aff.rot_feat_global, 
                global_feat_type=args.model.coarse_aff.rot_feat_global_type)
        else:
            # create index corresponding to identity rotation 
            identity_rot_mats = torch.eye(3)[None, :, :].repeat((bs, 1, 1)).float().cuda()
            rot_aff_model_output = {}
            if coarse_aff_model.euler_rot:
                onehot_euler_dict_t = get_euler_onehot_torch(matrix_to_euler_angles(identity_rot_mats, 'ZYX'), bins_per_axis=args.data.euler_bins_per_axis)
                identity_rot_euler_idx = torch.cat(list(onehot_euler_dict_t.values()), dim=-1).reshape(bs, args.data.euler_bins_per_axis*3)
                rot_aff_model_output['rot_affordance'] = identity_rot_euler_idx
            else:
                identity_rot_idx = get_rotation_grid_index_torch(identity_rot_mats, rot_mat_grid)
                rot_aff_model_output['rot_affordance'] = torch.nn.functional.one_hot(identity_rot_idx.long(), num_classes=rot_grid_bins).float()
            rot_aff_model_output['force_out_rot_mat'] = identity_rot_mats
    else:
            rot_aff_model_output = coarse_aff_model.rot_affordance(
                coarse_aff_rot_mi, 
                mean_child=args.model.coarse_aff.rot_feat_mean_child, 
                global_feat=args.model.coarse_aff.rot_feat_global, 
                global_feat_type=args.model.coarse_aff.rot_feat_global_type)

    if args.debug:
        # apply the predicted rotation and visualize the resulting child point cloud
        
        # get the top rotation matrices
        if coarse_aff_model.euler_rot:
            top_rot_mats = euler_disc_pred_to_rotmat(
                rot_aff_model_output['rot_affordance'], 
                bins_per_axis=coarse_aff_model.euler_bins_per_axis)
        else:
            top_rot_inds = torch.max(rot_aff_model_output['rot_affordance'], 1)[1]  # B x 1
            top_rot_mats = rot_mat_grid[top_rot_inds]  # B x 3 x 3
        # top_rot_mats = rot_mat_grid[top_rot_inds.detach().cpu().numpy()]  # B x 3 x 3
        # top_rot_mats = torch.from_numpy(top_rot_mats).float().cuda()

        # apply output rotation to object point cloud 

        # center child point cloud
        child_start_pcd_sn = coarse_aff_rot_mi['child_start_pcd_sn']
        child_start_pcd_sn_mean = torch.mean(child_start_pcd_sn, 1).reshape(bs, 1, 3).repeat((1, n_child_pts, 1))
        child_start_pcd_sn_cent = child_start_pcd_sn - child_start_pcd_sn_mean

        # rotate
        child_rot_pcd_sn_cent = torch.bmm(top_rot_mats, child_start_pcd_sn_cent.transpose(1, 2)).transpose(2, 1).contiguous()

        # translate back
        child_final_pcd_sn = child_rot_pcd_sn_cent + child_start_pcd_sn_mean

        # show prediction
        util.meshcat_pcd_show(
            mc_vis,
            coarse_aff_rot_mi['parent_start_pcd_sn'][db_idx].detach().cpu().numpy(),
            (255, 0, 0),
            f'{mc_iter_name}/post_rot_aff/parent_start_pcd_sn',
            size=MC_SIZE
        )
        util.meshcat_pcd_show(
            mc_vis,
            child_final_pcd_sn[db_idx].detach().cpu().numpy(),
            (0, 255, 255),
            f'{mc_iter_name}/post_rot_aff/child_final_pcd_sn',
            size=MC_SIZE*1.5
        )

        # # show voxel containing the feature we used to compute the rotation
        # cpcd_mean_grid_pts = voxel_grid_pts[rot_aff_model_output['child_mean_raster_index']]
        # util.meshcat_pcd_show(
        #     mc_vis,
        #     cpcd_mean_grid_pts[db_idx].detach().cpu().numpy().reshape(1, 3),
        #     (255, 0, 255),
        #     f'{mc_iter_name}/post_rot_aff/child_mean_grid_pt_global_feat',
        #     size=MC_SIZE*7.5
        # )

        # show ground truth
        util.meshcat_pcd_show(
            mc_vis,
            coarse_aff_gt['child_final_pcd_sn'][db_idx].detach().cpu().numpy(),
            (0, 255, 0),
            f'{mc_iter_name}/post_rot_aff/child_final_pcd_sn_gt',
            size=MC_SIZE*1.5
        )

        # compute ground truth
        # rotate
        if coarse_aff_model.euler_rot:
            rot_mats_gt = euler_disc_pred_to_rotmat(
                coarse_aff_gt['euler_onehot'], 
                bins_per_axis=coarse_aff_model.euler_bins_per_axis)
        else:
            rot_mats_gt = rot_mat_grid[coarse_aff_gt['rot_mat_grid_index'].long()]
        child_rot_pcd_sn_cent_gt = torch.bmm(rot_mats_gt, child_start_pcd_sn_cent.transpose(1, 2)).transpose(2, 1).contiguous()

        # translate back
        child_final_pcd_sn_gt = child_rot_pcd_sn_cent_gt + child_start_pcd_sn_mean

        # show ground truth
        util.meshcat_pcd_show(
            mc_vis,
            child_final_pcd_sn_gt[db_idx].detach().cpu().numpy(),
            (0, 255, 0),
            f'{mc_iter_name}/post_rot_aff/child_final_pcd_sn_gt_computed',
            size=MC_SIZE*1.5
        )

        from IPython import embed; embed()

    #######################################################################
    # Combine model outputs and compute loss
    coarse_aff_model_output = {}
    coarse_aff_model_output['voxel_affordance'] = voxel_aff_model_output['voxel_affordance']
    coarse_aff_model_output['rot_affordance'] = rot_aff_model_output['rot_affordance']
    if 'force_out_rot_mat' in rot_aff_model_output:
        coarse_aff_model_output['force_out_rot_mat'] = rot_aff_model_output['force_out_rot_mat']

    loss_dict = aff_loss_fn(
        coarse_aff_model_output, coarse_aff_gt, 
        reso=reso_grid, rot_reso=rot_grid_bins)

    # loss = 0
    # for k, v in loss_dict.items():
    #     loss += v
    loss_voxel = loss_dict['voxel_affordance']
    loss_rot = loss_dict['rot_affordance']

    # loss = loss_voxel + loss_rot

    # check if we should be considering the rot affordance output
    if args.loss.coarse_aff.rot_only_on_refine:
        if refine_pred:
            loss = loss_voxel + loss_rot
        else:
            loss = loss_voxel
    else:
        loss = loss_voxel + loss_rot

    #######################################################################
    # Gradient step, log, and return
    
    if args.optimizer.coarse_aff.use_schedule:
        sched_args = args.optimizer.schedule
        # if args.optimizer.coarse_aff.schedule is not None:
        if args.optimizer.coarse_aff.get('schedule') is not None:
            if args.optimizer.coarse_aff.schedule is not None:
                sched_args = args.optimizer.coarse_aff.schedule

        opt_type = args.optimizer.coarse_aff.type
        sched_args.lr = args.optimizer[opt_type].lr
        sched_args.epochs = args.experiment.num_iterations * bs / args.experiment.dataset_length
        
        adj_lr = adjust_learning_rate(aff_optimizer, current_epoch, sched_args) 
        lr = aff_optimizer.param_groups[0]['lr'] 

    aff_optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(coarse_aff_model.parameters(), 0.5)
    grad_norm = get_grad_norm(coarse_aff_model) 
    aff_optimizer.step()
    
    if it % args.experiment.log_interval == 0 and args.experiment.train.out_log_coarse_aff:
        string = f'[Coarse Affordance (Refine: {refine_pred})] Iteration: {it} (Epoch: {int(current_epoch)}) '
        for loss_name, loss_val in loss_dict.items():
            string += f'{loss_name}: {loss_val.mean().item():.6f} '
            logger.add_scalar(loss_name, loss_val.mean().item(), it)

            end_time = time.time()
            total_duration = end_time - start_time


        string += f' Grad norm : {grad_norm:.4f}'
        if args.optimizer.coarse_aff.use_schedule:
            string += f' LR : {lr:.7f}'
        string += f' Duration: {total_duration:.4f}'
        print(string)

    out_dict = {}
    out_dict['loss'] = loss_dict
    out_dict['model_output'] = coarse_aff_model_output

    return out_dict

def train_iter_refine_pose(
            pose_refine_mi: dict,
            pose_refine_gt: dict,
            pose_refine_model: nn.Module,
            pr_optimizer: Optimizer,
            pr_loss_fn: Callable,
            args: AttrDict,
            it: int, current_epoch: float,
            logger: SummaryWriter,
            mc_vis: Visualizer=None):

    mc_iter_name = 'scene/train_iter_pose'
    if mc_vis is not None:
        mc_vis[mc_iter_name].delete()
    start_time = time.time()

    bs = pose_refine_mi['parent_start_pcd'].shape[0]
    n_child_pts = pose_refine_mi['child_start_pcd'].shape[1]
    db_idx = 0

    #######################################################################
    # Prepare for refinement predictions
    if args.debug:
        util.meshcat_pcd_show(
            mc_vis, 
            pose_refine_mi['parent_start_pcd'][db_idx].detach().cpu().numpy(), 
            (255, 0, 0), 
            f'{mc_iter_name}/pcd_feat_mi/parent_point_cloud', 
            size=MC_SIZE)
        util.meshcat_pcd_show(
            mc_vis, 
            pose_refine_mi['child_start_pcd'][db_idx].detach().cpu().numpy(), 
            (0, 0, 255), 
            f'{mc_iter_name}/pcd_feat_mi/child_point_cloud', 
            size=MC_SIZE)

    #######################################################################
    # Make rotation refinement prediction
    if args.debug:
        p_pcd_viz_cent = pose_refine_mi['parent_start_pcd'][db_idx].detach().cpu().numpy()
        c_pcd_viz_cent = pose_refine_mi['child_start_pcd'][db_idx].detach().cpu().numpy() 

        util.meshcat_pcd_show(
            mc_vis, 
            p_pcd_viz_cent,
            (255, 0, 0), 
            f'{mc_iter_name}/rot_pose_refine_mi/parent_start_pcd', 
            size=MC_SIZE)
        util.meshcat_pcd_show(
            mc_vis, 
            c_pcd_viz_cent,
            (0, 0, 255), 
            f'{mc_iter_name}/rot_pose_refine_mi/child_start_pcd', 
            size=MC_SIZE)

        util.meshcat_pcd_show(
            mc_vis, 
            (p_pcd_viz_cent + pose_refine_mi['parent_start_pcd_mean'][db_idx].detach().cpu().numpy()), 
            (255, 0, 0), 
            f'{mc_iter_name}/rot_pose_refine_mi/parent_pcd_uncent', 
            size=MC_SIZE)
        util.meshcat_pcd_show(
            mc_vis, 
            (c_pcd_viz_cent + pose_refine_mi['child_start_pcd_mean'][db_idx].detach().cpu().numpy()),
            (0, 0, 255), 
            f'{mc_iter_name}/rot_pose_refine_mi/child_pcd_uncent', 
            size=MC_SIZE)

    # if training a VAE, include the pose target in the input for the encoder
    if 'vae' in args.model.refine_pose.type:
        trans_to_enc = pose_refine_gt['trans'].float().cuda()
        rot_mat_label = pose_refine_gt['rot_mat'].float().cuda()
        quat_to_enc = matrix_to_quaternion(rot_mat_label)
        tf_to_enc = torch.cat((trans_to_enc, quat_to_enc), dim=-1)
        pose_refine_mi['tf'] = tf_to_enc
        pose_refine_mi['rot_trans'] = 'rot'

    if args.data.refine_pose.diffusion_steps:
        # position embedding for the timestep
        timestep_emb = pose_refine_model.pos_emb(pose_refine_mi['diffusion_timestep'])
        pose_refine_mi['timestep_emb'] = timestep_emb

    rot_model_output_raw = pose_refine_model(pose_refine_mi, rot=True)

    # apply output rotation to object point cloud for translation prediction
    child_start_pcd_original = pose_refine_mi['child_start_pcd'].clone().detach()

    rot_idx = torch.zeros(rot_model_output_raw['rot_mat'].shape[0]).long().cuda()

    rot_model_output = {}
    rot_model_output['rot_mat'] = torch.gather(rot_model_output_raw['rot_mat'], dim=1, index=rot_idx[:, None, None, None].repeat(1, 1, 3, 3)).reshape(bs, 3, 3)
    rot_model_output['quat'] = matrix_to_quaternion(rot_model_output['rot_mat'])

    # apply output rotation to object point cloud for translation prediction
    child_pcd_final_pred = torch.bmm(
        rot_model_output['rot_mat'], 
        pose_refine_mi['child_start_pcd'].transpose(1, 2)).transpose(2, 1).contiguous()  # flip to B x 3 x N, back to B x N x 3
    child_pcd_rot = child_pcd_final_pred.clone().detach()

    if args.debug:
        util.meshcat_pcd_show(
            mc_vis, 
            pose_refine_mi['child_start_pcd'][db_idx].detach().cpu().numpy(), 
            (255, 255, 0), 
            f'{mc_iter_name}/rot_policy_pred/child_pre_rot_cent', 
            size=MC_SIZE)

        util.meshcat_pcd_show(
            mc_vis, 
            child_pcd_rot[db_idx].detach().cpu().numpy(), 
            (255, 255, 0), 
            f'{mc_iter_name}/rot_policy_pred/child_post_rot_cent', 
            size=MC_SIZE)

        c_pcd_mean = pose_refine_mi['child_start_pcd_mean'].reshape(-1, 1, 3).repeat(1, child_pcd_rot.shape[1], 1) 
        c_pcd_trans_gt = pose_refine_gt['trans'].reshape(-1, 1, 3).repeat(1, child_pcd_rot.shape[1], 1).cuda() 

        viz_rot_wf = child_pcd_rot + c_pcd_mean + c_pcd_trans_gt
        util.meshcat_pcd_show(
            mc_vis, 
            viz_rot_wf[db_idx].detach().cpu().numpy(), 
            (0, 255, 0), 
            f'{mc_iter_name}/rot_policy_pred/child_post_rot_wf_gt',
            size=MC_SIZE)

        rot_frame_wf = np.eye(4)
        rot_frame_wf[:-1, :-1] = rot_model_output['rot_mat'][db_idx].detach().cpu().numpy()
        start_rot_frame = np.eye(4)
        start_rot_frame[:-1, :-1] = np.linalg.inv(pose_refine_gt['rot_mat'][db_idx].detach().cpu().numpy().squeeze())
        final_rot_frame_wf = np.matmul(rot_frame_wf, start_rot_frame)
        final_rot_frame_wf[:-1, -1] = pose_refine_mi['child_start_pcd_mean'].detach().cpu().numpy()[db_idx].squeeze()
        util.meshcat_frame_show(
            mc_vis, 
            f'{mc_iter_name}/rot_policy_pred/rot_mat_frame', 
            final_rot_frame_wf)

    pose_refine_mi['child_start_pcd'] = child_pcd_rot

    if args.debug:
        util.meshcat_pcd_show(
            mc_vis, 
            pose_refine_mi['child_start_pcd'][db_idx].detach().cpu().numpy(), 
            (255, 0, 255), 
            f'{mc_iter_name}/trans_pose_refine_mi/child_start_pcd', 
            size=MC_SIZE)

    #######################################################################
    # After re-encoding the rotated shape, make translation refinement prediction
    
    # if using VAE, provide the target tf for encoder (now, with rotation applied)
    if 'vae' in args.model.refine_pose.type:
        trans_to_enc = pose_refine_gt['trans'].float().cuda()
        rot_mat_label = pose_refine_gt['rot_mat'].float().cuda()

        rot_mat_label_new = torch.bmm(
            rot_mat_label,
            torch.inverse(rot_model_output['rot_mat']))

        quat_to_enc = matrix_to_quaternion(rot_mat_label_new)
        tf_to_enc = torch.cat((trans_to_enc, quat_to_enc), dim=-1)
        pose_refine_mi['tf'] = tf_to_enc
        pose_refine_mi['rot_trans'] = 'trans'

    # make translation prediction
    trans_model_output_raw = pose_refine_model(pose_refine_mi, stop=False)

    trans_idx = torch.zeros(trans_model_output_raw['trans'].shape[0]).long().cuda()
    
    trans_model_output = {}
    trans_model_output['trans'] = torch.gather(trans_model_output_raw['trans'], dim=1, index=trans_idx[:, None, None].repeat(1, 1, 3)).reshape(bs, 3)

    pose_refine_model_output = {}
    pose_refine_model_output['trans_raw'] = trans_model_output_raw['trans']
    pose_refine_model_output['trans'] = trans_model_output['trans']
    pose_refine_model_output['quat_raw'] = rot_model_output_raw['quat']
    pose_refine_model_output['rot_mat_raw'] = rot_model_output_raw['rot_mat']
    pose_refine_model_output['quat'] = rot_model_output['quat']
    pose_refine_model_output['rot_mat'] = rot_model_output['rot_mat']
    pose_refine_model_output['unnorm_quat'] = pose_refine_model_output['quat']

    # if using VAE, provide the target tf for encoder (now, with rotation applied)
    if 'vae' in args.model.refine_pose.type:
        pose_refine_model_output['z_mu_rot'] = rot_model_output_raw['z_mu']
        pose_refine_model_output['z_logvar_rot'] = rot_model_output_raw['z_logvar']
        pose_refine_model_output['z_mu_trans'] = trans_model_output_raw['z_mu']
        pose_refine_model_output['z_logvar_trans'] = trans_model_output_raw['z_logvar']

    # apply output transformation to object point cloud (for chamfer loss + visualization)
    child_pcd_pred_rot = torch.bmm(pose_refine_model_output['rot_mat'], child_start_pcd_original.transpose(1, 2)).transpose(2, 1)
    child_pcd_pred_rot = child_pcd_pred_rot + pose_refine_mi['child_start_pcd_mean'].reshape(-1, 1, 3).repeat(1, child_pcd_pred_rot.shape[1], 1) 
    child_pcd_final_pred = child_pcd_pred_rot + pose_refine_model_output['trans'].reshape(-1, 1, 3).repeat(1, child_pcd_pred_rot.shape[1], 1) 
    pose_refine_model_output['child_pcd_final_pred'] = child_pcd_final_pred

    if args.debug:
        util.meshcat_pcd_show(
            mc_vis, 
            child_pcd_pred_rot[db_idx].detach().cpu().numpy(), 
            (255, 128, 128), 
            f'{mc_iter_name}/trans_policy_pred_model_output/child_pcd_rot_pre_trans_wf', 
            size=MC_SIZE)
        util.meshcat_pcd_show(
            mc_vis, 
            pose_refine_model_output['child_pcd_final_pred'][db_idx].detach().cpu().numpy(), 
            (0, 255, 0), 
            f'{mc_iter_name}/trans_policy_pred_model_output/child_pcd_rot_post_trans_wf_final_pred', 
            size=MC_SIZE)

        # also show the ground truth, for reference
        child_pcd_rot_gt = torch.bmm(pose_refine_gt['rot_mat'], child_start_pcd_original.transpose(1, 2)).transpose(2, 1)

        gt_trans = pose_refine_gt['trans'].reshape(-1, 1, 3).repeat((1, child_pcd_rot_gt.shape[1], 1)) 
        gt_mean = pose_refine_mi['child_start_pcd_mean'].reshape(-1, 1, 3).repeat((1, child_pcd_rot_gt.shape[1], 1)) 
        child_pcd_rot_trans_gt = child_pcd_rot_gt + gt_trans + gt_mean 

        util.meshcat_pcd_show(
            mc_vis, 
            child_pcd_rot_trans_gt[db_idx].detach().cpu().numpy(), 
            (0, 255, 255), 
            f'{mc_iter_name}/ground_truth/child_final_pcd', 
            size=MC_SIZE)
        util.meshcat_pcd_show(
            mc_vis, 
            pose_refine_gt['parent_final_pcd'][db_idx].detach().cpu().numpy(), 
            (0, 0, 0), 
            f'{mc_iter_name}/ground_truth/parent_final_pcd', 
            size=MC_SIZE)
        from IPython import embed; embed()

    if args.experiment.train.predict_offset:
        # update mean for pose refine model input offset prediction
        pose_refine_mi['child_start_pcd_mean'] = torch.mean(pose_refine_model_output['child_pcd_final_pred'], dim=1)
        if pose_refine_mi.get('parent_start_pcd_offset') is not None:
            pose_refine_mi['parent_start_pcd'] = pose_refine_mi['parent_start_pcd_offset']
        trans_offset_model_output_raw = pose_refine_model(pose_refine_mi, stop=False)
        offset_ind = 0  # force the first index for unimodal offset prediction
        offset_idx = (torch.ones((bs)) * offset_ind).long().cuda()

        trans_offset_out = torch.gather(trans_offset_model_output_raw['trans_offset'], dim=1, index=offset_idx[:, None, None].repeat(1, 1, 3)).reshape(bs, 3)
        pose_refine_model_output['trans_offset_raw'] = trans_offset_model_output_raw['trans_offset']
        pose_refine_model_output['trans_offset'] = trans_offset_out

        if args.debug:

            pred_mean = pose_refine_mi['child_start_pcd_mean'].reshape(-1, 1, 3)
            pred_offset_pos = pred_mean + pose_refine_model_output['trans_offset'].reshape(-1, 1, 3) * 0.25

            util.meshcat_pcd_show(
                mc_vis, 
                pred_offset_pos[db_idx].detach().cpu().numpy(), 
                (255, 0, 255), 
                f'{mc_iter_name}/trans_offset_policy_pred_model_output/pred_offset_pos', 
                size=MC_SIZE*10)

            gt_mean = pose_refine_gt['child_final_pcd_mean'].reshape(-1, 1, 3)
            gt_offset_pos = gt_mean + pose_refine_gt['trans_offset'].reshape(-1, 1, 3) * 0.25

            util.meshcat_pcd_show(
                mc_vis, 
                gt_offset_pos[db_idx].detach().cpu().numpy(), 
                (255, 0, 0), 
                f'{mc_iter_name}/ground_truth/offset_pos', 
                size=MC_SIZE*10)

            from IPython import embed; embed()

    #######################################################################
    # Combine model outputs and compute loss
    loss_dict = pr_loss_fn(pose_refine_model_output, pose_refine_gt) 

    loss_trans = loss_dict['trans']
    loss_rot = loss_dict['rot']
    loss_chamf = loss_dict['chamf']

    # waff = 1.0
    if args.loss.refine_pose.chamf_only:
        loss = loss_chamf
    elif args.loss.refine_pose.trans_rot_only:
        loss = loss_trans + loss_rot
    else:
        loss = loss_trans + loss_rot + loss_chamf

    # if using VAE, provide the target tf for encoder (now, with rotation applied)
    if 'vae' in args.model.refine_pose.type and 'kl' in args.loss.refine_pose.type:
        rot_kl_loss = loss_dict['rot_kl']
        trans_kl_loss = loss_dict['trans_kl']
        
        rot_kl_w = args.loss.kl_div.rot_kl_weight
        trans_kl_w = args.loss.kl_div.trans_kl_weight
        
        if args.loss.kl_div.anneal_rot:
            if args.loss.kl_div.anneal_rot_total_epochs is None:
                total_epochs = args.experiment.num_iterations * bs / args.experiment.dataset_length
            else:
                total_epochs = args.loss.kl_div.anneal_rot_total_iters * bs / args.experiment.dataset_length
                # total_epochs = args.loss.kl_div.anneal_rot_total_epochs

            rot_kl_rate = 1 - get_linear_warmup_cosine_decay_sched(
                epoch=current_epoch,
                warmup_epochs=args.loss.kl_div.anneal_rot_warmup_epochs,
                total_epochs=total_epochs,
            )

            rot_kl_w = rot_kl_w * rot_kl_rate

        if args.loss.kl_div.anneal_trans:
            if args.loss.kl_div.anneal_trans_total_epochs is None:
                total_epochs = args.experiment.num_iterations * bs / args.experiment.dataset_length
            else:
                total_epochs = args.loss.kl_div.anneal_trans_total_iters * bs / args.experiment.dataset_length
                # total_epochs = args.loss.kl_div.anneal_trans_total_epochs

            trans_kl_rate = 1 - get_linear_warmup_cosine_decay_sched(
                epoch=current_epoch,
                warmup_epochs=args.loss.kl_div.anneal_trans_warmup_epochs,
                total_epochs=total_epochs,
            )

            trans_kl_w = trans_kl_w * trans_kl_rate

        loss = loss + rot_kl_w*rot_kl_loss + trans_kl_w*trans_kl_loss

    #######################################################################
    # Gradient step, log, and return

    if args.optimizer.refine_pose.use_schedule:
        sched_args = args.optimizer.schedule
        # if args.optimizer.refine_pose.schedule is not None:
        if args.optimizer.refine_pose.get('schedule') is not None:
            if args.optimizer.refine_pose.schedule is not None:
                sched_args = args.optimizer.refine_pose.schedule

        opt_type = args.optimizer.refine_pose.type
        sched_args.lr = args.optimizer[opt_type].lr
        sched_args.epochs = args.experiment.num_iterations * bs / args.experiment.dataset_length

        adj_lr = adjust_learning_rate(pr_optimizer, current_epoch, sched_args) 
        lr = pr_optimizer.param_groups[0]['lr'] 

    pr_optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(pose_refine_model.parameters(), 0.5)
    grad_norm = get_grad_norm(pose_refine_model) 
    pr_optimizer.step()

    if it % args.experiment.log_interval == 0 and args.experiment.train.out_log_refine_pose:
        string = f'[Pose Refinement] Iteration: {it} (Epoch: {int(current_epoch)}) '
        for loss_name, loss_val in loss_dict.items():
            string += f'{loss_name}: {loss_val.mean().item():.6f} '
            logger.add_scalar(loss_name, loss_val.mean().item(), it)

            end_time = time.time()
            total_duration = end_time - start_time

        string += f' Grad norm : {grad_norm:.4f}'
        if args.optimizer.refine_pose.use_schedule:
            string += f' LR : {lr:.7f}'

        if 'vae' in args.model.refine_pose.type and 'kl' in args.loss.refine_pose.type:
            string += f' Trans KL weight : {trans_kl_w:.2f}, Rot KL weight: {rot_kl_w:.2f}'

        string += f' Duration: {total_duration:.4f}'

        print(string)

    out_dict = {}
    out_dict['loss'] = loss_dict
    out_dict['model_output'] = pose_refine_model_output

    return out_dict

def train_iter_success(
            success_mi: dict,
            success_gt: dict,
            success_model: nn.Module,
            succ_optimizer: Optimizer,
            succ_loss_fn: Callable,
            args: AttrDict,
            it: int, current_epoch: float,
            logger: SummaryWriter,
            mc_vis: Visualizer=None):

    mc_iter_name = 'scene/train_iter_success'
    if mc_vis is not None:
        mc_vis[mc_iter_name].delete()
    start_time = time.time()

    bs = success_mi['parent_start_pcd'].shape[0]
    n_child_pts = success_mi['child_start_pcd'].shape[1]
    db_idx = 0

    #######################################################################
    # Prepare for success predictions
    if args.debug:
        util.meshcat_pcd_show(
            mc_vis, 
            success_mi['parent_start_pcd'][db_idx].detach().cpu().numpy(), 
            (255, 0, 0), 
            f'{mc_iter_name}/pcd_feat_mi/parent_point_cloud', 
            size=MC_SIZE)
        util.meshcat_pcd_show(
            mc_vis, 
            success_mi['child_start_pcd'][db_idx].detach().cpu().numpy(), 
            (0, 0, 255), 
            f'{mc_iter_name}/pcd_feat_mi/child_point_cloud', 
            size=MC_SIZE)

    if False:
        # Get extra point cloud features if we need them
        parent_pcd_feat_mi = dict(point_cloud=success_mi['parent_start_pcd'])
        child_pcd_feat_mi = dict(point_cloud=success_mi['child_start_pcd'])

        parent_local_latent = feat_model_p.extract_local_latent(parent_pcd_feat_mi, new=True)  # assumes we have already centered based on the parent
        parent_global_latent = feat_model_p.extract_global_latent(parent_pcd_feat_mi, new=False)  # assumes we have already centered based on the parent

        child_local_latent = feat_model_c.extract_local_latent(child_pcd_feat_mi, new=True)  # assumes we have already centered based on the child
        child_global_latent = feat_model_c.extract_global_latent(child_pcd_feat_mi, new=False)  # assumes we have already centered based on the child

        success_mi['parent_point_latent'] = parent_local_latent
        success_mi['child_point_latent'] = child_local_latent
        success_mi['parent_global_latent'] = parent_global_latent
        success_mi['child_global_latent'] = child_global_latent

    #######################################################################
    # Predict success
    if args.debug:
        offset = np.array([0.4, 0.0, 0.0])
        for jj in range(bs):
            parent_pcd_cent = success_mi['parent_start_pcd'][jj].detach().cpu().numpy()
            parent_pcd_mean = success_mi['parent_start_pcd_mean'][jj].detach().cpu().numpy()
            util.meshcat_pcd_show(
                mc_vis, 
                (offset * jj - bs * offset) + parent_pcd_cent + parent_pcd_mean,
                (255, 0, 0),
                f'{mc_iter_name}/success_mi/parent_start_pcd_{jj}', 
                size=MC_SIZE)

            child_pcd_cent = success_mi['child_start_pcd'][jj].detach().cpu().numpy()
            child_pcd_mean = success_mi['child_start_pcd_mean'][jj].detach().cpu().numpy()
            util.meshcat_pcd_show(
                mc_vis, 
                (offset * jj - bs * offset) + child_pcd_cent + child_pcd_mean,
                (0, 0, 255),
                f'{mc_iter_name}/success_mi/child_start_pcd_{jj}', 
                size=MC_SIZE)

    success_model_output = success_model(success_mi)

    if args.debug:
        for jj in range(bs):
            print(f'Success model input: {jj}, Pred Success: {torch.sigmoid(success_model_output["success"][jj]).item()}, GT Sucess: {success_gt["success"][jj].item()}')
        from IPython import embed; embed()

    #######################################################################
    # Combine model outputs and compute loss
    loss_dict = succ_loss_fn(success_model_output, success_gt) 
    loss_succ = loss_dict['success']

    loss = loss_succ

    if args.loss.success.use_aux_multi_query_loss:
        aux_loss_list = []
        for aux_idx in range(success_model_output['success_aux'].shape[2]):
            sc_aux_out = {}
            sc_aux_out['success'] = success_model_output['success_aux'][:, :, aux_idx].reshape(bs, 1, 1)

            aux_loss_dict = succ_loss_fn(sc_aux_out, success_gt) 
            aux_loss_succ = aux_loss_dict['success']
            aux_loss = aux_loss_succ
            aux_loss_list.append(aux_loss)
       
        n_aux = len(aux_loss_dict)
        for idx, al in enumerate(aux_loss_list[:-1]):
            loss_dict[f'aux {idx}'] = al / n_aux
            loss = loss + al 

    #######################################################################
    # Gradient step, log, and return

    if args.optimizer.success.use_schedule:
        sched_args = args.optimizer.schedule
        # if args.optimizer.success.schedule is not None:
        if args.optimizer.success.get('schedule') is not None:
            if args.optimizer.success.schedule is not None:
                sched_args = args.optimizer.success.schedule

        opt_type = args.optimizer.success.type
        sched_args.lr = args.optimizer[opt_type].lr
        sched_args.epochs = args.experiment.num_iterations * bs / args.experiment.dataset_length

        adj_lr = adjust_learning_rate(succ_optimizer, current_epoch, sched_args) 
        lr = succ_optimizer.param_groups[0]['lr'] 

    succ_optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_norm_(success_model.parameters(), 0.5)
    grad_norm = get_grad_norm(success_model) 
    succ_optimizer.step()

    if it % args.experiment.log_interval == 0 and args.experiment.train.out_log_success:
        string = f'[Success Classifier] Iteration: {it} (Epoch: {int(current_epoch)}) '
        for loss_name, loss_val in loss_dict.items():
            string += f'{loss_name}: {loss_val.mean().item():.6f} '
            logger.add_scalar(loss_name, loss_val.mean().item(), it)

            end_time = time.time()
            total_duration = end_time - start_time


        string += f' Grad norm : {grad_norm:.4f}'
        if args.optimizer.success.use_schedule:
            string += f' LR : {lr:.7f}'
        string += f' Duration: {total_duration:.4f}'
        print(string)

    out_dict = {}
    out_dict['loss'] = loss_dict
    out_dict['model_output'] = success_model_output

    return out_dict
