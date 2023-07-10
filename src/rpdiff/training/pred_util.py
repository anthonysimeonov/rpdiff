# From: https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
import numpy as np
import torch
from torch_scatter import scatter_mean

from rpdiff.utils import util
from rpdiff.utils.torch3d_util import matrix_to_quaternion, quaternion_to_matrix, matrix_to_euler_angles
from rpdiff.utils.mesh_util import three_util
from rpdiff.utils.torch_scatter_utils import fps_downsample
from rpdiff.utils import batch_pcd_util
from rpdiff.training.train_util import crop_pcd_batch, pad_flat_pcd_batch


MC_SIZE=0.007


def detach_dict(in_dict):
    out_dict = {}
    for k, v in in_dict.items():
        if isinstance(v, torch.Tensor):
            out_dict[k] = v.detach()
        elif isinstance(v, dict):
            out_dict[k] = {k2: v2.detach() for k2, v2 in v.items()}
        else:
            pass
    return out_dict


def get_rotation_grid_index_torch(rot_mat, rot_grid):
    invmat = torch.inverse(rot_mat)
    outmat = torch.matmul(invmat[None, :], rot_grid[:, None])
    d0, d1 = outmat.shape[0], outmat.shape[1]
    outnorm = torch.linalg.matrix_norm(torch.eye(3)[None, None, :, :].repeat((d0, d1, 1, 1)).float().cuda() - outmat)
    min_idx = torch.argmin(outnorm, 0)
    return min_idx


def get_euler_onehot_torch(euler_angles, bins_per_axis):
    bs = euler_angles.shape[0]

    onehot_dict = {}
    ks = ['x', 'y', 'z']
    
    euler_rot_disc = torch.linspace(-np.pi, np.pi, bins_per_axis).reshape(1, -1).repeat((bs, 1)).float().cuda()

    for i, k in enumerate(ks):
        min_idx = torch.argmin(torch.sqrt((euler_angles[:, i].reshape(bs, 1).repeat((1, bins_per_axis)) - euler_rot_disc)**2), dim=1)
        onehot_dict[k] = torch.full((bs, bins_per_axis), 0.).float().cuda().scatter_(1, min_idx[:, None], 1.0)
    
    return onehot_dict


def coarse_aff_to_refine_pose(
        coarse_aff_mi, coarse_aff_mo, coarse_aff_gt, 
        refine_pose_mi, refine_pose_gt, 
        rot_grid, voxel_grid, args, mc_vis=None):

    child_start_pcd_original = coarse_aff_mi['child_start_pcd']
    bs = child_start_pcd_original.shape[0]
    n_child_pts = child_start_pcd_original.shape[1]

    # if args.debug:
    #     util.meshcat_pcd_show(
    #         mc_vis, 
    #         child_start_pcd_original[db_idx].detach().cpu().numpy(),
    #         (0, 0, 255),
    #         f'{mc_pred_name}/child_start_pcd_original_cent',
    #         size=MC_SIZE)

    # get predicted rotation and position
    out_rot_aff = coarse_aff_mo['rot_affordance']
    out_rot_idx = torch.argmax(out_rot_aff, dim=1)
    out_rot_mat = rot_grid[out_rot_idx]

    out_voxel_aff = coarse_aff_mo['voxel_affordance']
    out_voxel_idx = torch.argmax(out_voxel_aff, dim=1)
    out_voxel_pos = voxel_grid[out_voxel_idx]
    out_voxel_pos_wf = out_voxel_pos / coarse_aff_mi['scene_scale'][0] + coarse_aff_mi['scene_mean']
    
    # apply output transformation to object point cloud 
    child_pcd_pred_rot = torch.bmm(out_rot_mat, child_start_pcd_original.transpose(1, 2)).transpose(2, 1)
    child_pcd_final_pred = child_pcd_pred_rot + out_voxel_pos_wf.reshape(-1, 1, 3).repeat((1, n_child_pts, 1))

    child_final_pred_mean = torch.mean(child_pcd_final_pred.detach().clone(), axis=1)
    child_start_mean = child_final_pred_mean
    child_final_mean = refine_pose_gt['child_final_pcd_mean'].float().cuda()

    # update the ground truth
    refine_pose_gt['trans'] = child_final_mean - child_start_mean
    refine_pose_gt['rot_mat'] = torch.bmm(coarse_aff_gt['rot_mat'], torch.inverse(out_rot_mat.detach().clone()))
    refine_pose_gt['child_start_pcd_mean'] = child_start_mean

    # update the input
    # # child_pcd_final_pred_cent = child_pcd_final_pred.detach().clone() - child_final_mean[:, None, :].repeat((1, n_child_pts, 1))
    # child_pcd_final_pred_cent = child_pcd_final_pred.detach().clone() - child_final_pred_mean[:, None, :].repeat((1, n_child_pts, 1))
    # # refine_pose_mi['child_start_pcd_mean'] = child_final_mean
    # refine_pose_mi['child_start_pcd_mean'] = child_start_mean
    # # refine_pose_mi['child_start_pcd'] = child_pcd_final_pred_cent
    # refine_pose_mi['child_start_pcd'] = child_pcd_pred_rot

    child_pcd_final_pred_cent = child_pcd_final_pred.detach().clone() - child_final_pred_mean[:, None, :].repeat((1, n_child_pts, 1))
    refine_pose_mi['child_start_pcd_mean'] = child_final_pred_mean
    refine_pose_mi['child_start_pcd'] = child_pcd_final_pred_cent

    # get parent crop if needed
    if args.data.parent_crop or args.data.refine_pose.parent_crop:
        # parent_pcd_to_crop = coarse_aff_mi['parent_start_pcd_sn']
        parent_pcd_to_crop = refine_pose_gt['parent_final_pcd'].float().cuda()
        # parent_pcd_to_crop = fps_downsample(refine_pose_gt['parent_final_pcd'], min(n_child_pts * bs, refine_pose_gt['parent_final_pcd'].shape[1]))
        out_parent_pcd_cropped = crop_pcd_batch(parent_pcd_to_crop, child_start_mean, box_length=args.data.refine_pose.crop_box_length)
        parent_pcd_cropped, parent_batch_idx, n_parent_pts_per_crop = out_parent_pcd_cropped

        mean_batch_idx = parent_batch_idx.reshape(-1, 1).repeat((1, 3)).long()
        parent_crop_mean_batch = scatter_mean(parent_pcd_cropped, mean_batch_idx, dim=0)

        for b in range(child_start_mean.shape[0]):
            b_idxs = torch.where(parent_batch_idx == b)[0]
            parent_pcd_cropped[b_idxs] -= parent_crop_mean_batch[b].reshape(1, 3).repeat((b_idxs.shape[0], 1))

        if args.data.refine_pose.get('parent_crop_same_n') is not None:
            if args.data.refine_pose.parent_crop_same_n:
                parent_pcd_cropped = pad_flat_pcd_batch(parent_pcd_cropped, parent_batch_idx, n_parent_pts_per_crop, n_child_pts, pcd_default=parent_pcd_to_crop)

        refine_pose_mi['parent_start_pcd'] = parent_pcd_cropped
        refine_pose_mi['parent_start_pcd_mean'] = parent_crop_mean_batch.reshape(-1, 3) 
            
    return refine_pose_mi, refine_pose_gt


def coarse_aff_to_success(
        coarse_aff_mi, coarse_aff_mo, coarse_aff_gt, 
        success_mi,  success_gt, 
        rot_grid, voxel_grid, args, mc_vis=None): 

    child_start_pcd_original = coarse_aff_mi['child_start_pcd']
    bs = child_start_pcd_original.shape[0]
    n_child_pts = child_start_pcd_original.shape[1]

    # get predicted rotation and position
    out_rot_aff = coarse_aff_mo['rot_affordance']

    if np.random.random() > 0.5:
        out_rot_idx = torch.argmax(out_rot_aff, dim=1)
    else:
        out_rot_idx = torch.randint(out_rot_aff.shape[1], (bs,)).long().cuda()
    out_rot_mat = rot_grid[out_rot_idx]

    out_voxel_aff = coarse_aff_mo['voxel_affordance']
    
    if np.random.random() > 0.5:
        out_voxel_idx = torch.argmax(out_voxel_aff, dim=1)
    else:
        out_voxel_idx = torch.randint(out_voxel_aff.shape[1], size=(bs,)).long().cuda()

    out_voxel_pos = voxel_grid[out_voxel_idx]
    out_voxel_pos_wf = out_voxel_pos / coarse_aff_mi['scene_scale'][0] + coarse_aff_mi['scene_mean']
    
    # apply output transformation to object point cloud 
    child_pcd_pred_rot = torch.bmm(out_rot_mat, child_start_pcd_original.transpose(1, 2)).transpose(2, 1)
    child_pcd_final_pred = child_pcd_pred_rot + out_voxel_pos_wf.reshape(-1, 1, 3).repeat((1, n_child_pts, 1))

    child_final_pred_mean = torch.mean(child_pcd_final_pred.detach().clone(), axis=1)
    child_start_mean = child_final_pred_mean
    child_final_mean = success_gt['child_final_pcd_mean'].float().cuda()

    # update the ground truth
    success_gt['trans'] = child_final_mean - child_start_mean
    success_gt['rot_mat'] = torch.bmm(coarse_aff_gt['rot_mat'], torch.inverse(out_rot_mat.detach().clone()))
    success_gt['child_start_pcd_mean'] = child_start_mean

    # update the input
    # child_pcd_final_pred_cent = child_pcd_final_pred.detach().clone() - child_final_mean[:, None, :].repeat((1, n_child_pts, 1))
    child_pcd_final_pred_cent = child_pcd_final_pred.detach().clone() - child_final_pred_mean[:, None, :].repeat((1, n_child_pts, 1))
    success_mi['child_start_pcd_mean'] = child_final_mean
    success_mi['child_start_pcd'] = child_pcd_final_pred_cent

    success_mi['child_start_pcd_mean'] = torch.cat([success_mi['child_start_pcd_mean'], success_gt['child_final_pcd_mean'].float().cuda()], dim=0)
    success_mi['parent_start_pcd_mean'] = torch.cat([success_mi['parent_start_pcd_mean'], success_gt['parent_final_pcd_mean']], dim=0).float().cuda()

    child_start_final_pcd_mean = success_gt['child_final_pcd_mean'].reshape(-1, 1, 3).repeat((1, n_child_pts, 1)).float().cuda()
    success_mi['child_start_pcd'] = torch.cat(
        [success_mi['child_start_pcd'], 
         success_gt['child_final_pcd'].float().cuda() - child_start_final_pcd_mean], dim=0)

    parent_final_pcd_mean = success_gt['parent_final_pcd_mean'].reshape(-1, 1, 3).repeat((1, n_child_pts, 1)).float().cuda()
    success_mi['parent_start_pcd'] = (success_gt['parent_final_pcd'].float().cuda() - parent_final_pcd_mean).repeat((2, 1, 1))

    # succ = torch.cat([succ, torch.ones(success_gt['child_final_pcd'].shape[0]).cuda()], dim=0)
    succ = torch.cat([
        torch.zeros(success_gt['child_final_pcd'].shape[0]),
        torch.ones(success_gt['child_final_pcd'].shape[0])], dim=0).long().cuda()
    success_gt['success'] = succ

    return success_mi, success_gt


def refine_pose_to_success(
        refine_pose_mi, refine_pose_mo, refine_pose_gt, 
        success_mi, success_gt, 
        args, mc_vis=None): 

    child_start_pcd_original = refine_pose_mi['child_start_pcd']
    bs = child_start_pcd_original.shape[0]
    n_child_pts = child_start_pcd_original.shape[1]
    nq  = refine_pose_mo['trans_raw'].shape[1]

    # get predicted rotation and position
    out_rot_aff = refine_pose_mo['rot_multi_query_affordance']

    if np.random.random() > 0.5:
        # use closest to the ground truth 
        rot_mat_label = refine_pose_gt['rot_mat'].float().cuda()
        quat_output = refine_pose_mo['quat_raw'].detach()
        # nq = quat_output.shape[1]
        quat_label = matrix_to_quaternion(rot_mat_label)[:, None, :].repeat(1, nq, 1)
        quat_scalar_prod = torch.sum(quat_output * quat_label, axis=-1)
        quat_dist = 1 - torch.pow(quat_scalar_prod, 2)

        # out_rot_idx = torch.argmax(refine_pose_mo['rot_multi_query_affordance'], 1).reshape(-1)
        out_rot_idx = torch.argmin(quat_dist, 1)
    else:
        out_rot_idx = torch.randint(out_rot_aff.shape[1], (bs,)).long().cuda()
    out_rot_mat = torch.gather(refine_pose_mo['rot_mat_raw'], dim=1, index=out_rot_idx[:, None, None, None].repeat(1, 1, 3, 3)).reshape(bs, 3, 3)
    out_quat = matrix_to_quaternion(out_rot_mat)

    trans_output = refine_pose_mo['trans_raw'].detach()
    trans_label = refine_pose_gt['trans'].float().cuda()
    trans_loss = torch.norm(trans_label[:, None, :].repeat(1, nq, 1) - trans_output, p=2, dim=-1)

    if np.random.random() > 0.5:
        trans_idx = torch.argmin(trans_loss, 1)
    else:
        trans_idx = torch.randint(0, nq, (bs,)).cuda()

    out_trans = torch.gather(refine_pose_mo['trans_raw'], dim=1, index=trans_idx[:, None, None].repeat(1, 1, 3)).reshape(bs, 3)
    out_trans = out_trans.reshape(-1, 1, 3).repeat((1, n_child_pts, 1))

    child_start_pcd_mean = refine_pose_mi['child_start_pcd_mean'].reshape(bs, 1, 3).repeat((1, n_child_pts, 1))
    
    # apply output transformation to object point cloud 
    child_pcd_pred_rot = torch.bmm(out_rot_mat, child_start_pcd_original.transpose(1, 2)).transpose(2, 1)
    child_pcd_final_pred = child_pcd_pred_rot + out_trans + child_start_pcd_mean

    child_final_pred_mean = torch.mean(child_pcd_final_pred.detach().clone(), axis=1)
    child_start_mean = child_final_pred_mean
    child_final_mean = success_gt['child_final_pcd_mean'].float().cuda()

    # update the ground truth
    success_gt['trans'] = child_final_mean - child_start_mean
    success_gt['rot_mat'] = torch.bmm(refine_pose_gt['rot_mat'], torch.inverse(out_rot_mat.detach().clone()))
    success_gt['child_start_pcd_mean'] = child_start_mean

    # update the input
    # child_pcd_final_pred_cent = child_pcd_final_pred.detach().clone() - child_final_mean[:, None, :].repeat((1, n_child_pts, 1))
    child_pcd_final_pred_cent = child_pcd_final_pred.detach().clone() - child_final_pred_mean[:, None, :].repeat((1, n_child_pts, 1))
    success_mi['child_start_pcd_mean'] = child_final_mean
    success_mi['child_start_pcd'] = child_pcd_final_pred_cent

    success_mi['child_start_pcd_mean'] = torch.cat([success_mi['child_start_pcd_mean'], success_gt['child_final_pcd_mean'].float().cuda()], dim=0)
    success_mi['parent_start_pcd_mean'] = torch.cat([success_mi['parent_start_pcd_mean'], success_gt['parent_final_pcd_mean']], dim=0).float().cuda()

    child_start_final_pcd_mean = success_gt['child_final_pcd_mean'].reshape(-1, 1, 3).repeat((1, n_child_pts, 1)).float().cuda()
    success_mi['child_start_pcd'] = torch.cat(
        [success_mi['child_start_pcd'], 
         success_gt['child_final_pcd'].float().cuda() - child_start_final_pcd_mean], dim=0)

    parent_final_pcd_mean = success_gt['parent_final_pcd_mean'].reshape(-1, 1, 3).repeat((1, n_child_pts, 1)).float().cuda()
    success_mi['parent_start_pcd'] = (success_gt['parent_final_pcd'].float().cuda() - parent_final_pcd_mean).repeat((2, 1, 1))

    # succ = torch.cat([succ, torch.ones(success_gt['child_final_pcd'].shape[0]).cuda()], dim=0)
    succ = torch.cat([
        torch.zeros(success_gt['child_final_pcd'].shape[0]),
        torch.ones(success_gt['child_final_pcd'].shape[0])], dim=0).long().cuda()
    success_gt['success'] = succ

    return success_mi, success_gt


def coarse_aff_to_coarse_aff(
        coarse_aff_mi, coarse_aff_mo, coarse_aff_gt, 
        rot_grid, voxel_grid_pred, voxel_grid, voxel_reso_grid, 
        args, mc_vis=None):
    child_start_pcd_original = coarse_aff_mi['child_start_pcd']
    parent_start_pcd_original = coarse_aff_mi['parent_start_pcd_dense']
    # parent_start_pcd_original = coarse_aff_mi['parent_start_pcd']
    bs = child_start_pcd_original.shape[0]
    n_child_pts = child_start_pcd_original.shape[1]
    n_parent_pts = parent_start_pcd_original.shape[1]

    parent_start_pcd_original_wf = parent_start_pcd_original + coarse_aff_mi['parent_start_pcd_mean'].reshape(-1, 1, 3).repeat((1, n_parent_pts, 1))

    # get predicted rotation and position
    out_rot_aff = coarse_aff_mo['rot_affordance']

    if 'force_out_rot_mat' in coarse_aff_mo:
        out_rot_mat = coarse_aff_mo['force_out_rot_mat']
    else:
        out_rot_idx = torch.argmax(out_rot_aff, dim=1)
        out_rot_mat = rot_grid[out_rot_idx]

    out_voxel_aff = coarse_aff_mo['voxel_affordance']
    gt_voxel_idx = coarse_aff_gt['child_centroid_voxel_index'].long()
    if np.random.random() > (1 - args.data.coarse_aff.c2f.use_pred_prob):
        out_voxel_idx = torch.argmax(out_voxel_aff, dim=1)
    else:
        out_voxel_idx = gt_voxel_idx
    out_voxel_pos = voxel_grid_pred[out_voxel_idx]
    out_voxel_pos_wf = out_voxel_pos / coarse_aff_mi['scene_scale'][0] + coarse_aff_mi['scene_mean']
    
    # apply output transformation to object point cloud 
    child_pcd_pred_rot = torch.bmm(out_rot_mat, child_start_pcd_original.transpose(1, 2)).transpose(2, 1)
    child_pcd_final_pred = child_pcd_pred_rot + out_voxel_pos_wf.reshape(-1, 1, 3).repeat((1, n_child_pts, 1))

    child_final_pred_mean = torch.mean(child_pcd_final_pred.detach().clone(), axis=1)
    child_start_mean = child_final_pred_mean
    child_final_mean = torch.mean(coarse_aff_gt['child_final_pcd'], 1).float().cuda()
    
    # print('here in pred util')
    # from IPython import embed; embed()
    # get parent crop if needed
    if args.data.coarse_aff.c2f.parent_crop:

        parent_pcd_to_crop = parent_start_pcd_original_wf
        # if False:
        try:
            out_parent_pcd_cropped = crop_pcd_batch(parent_pcd_to_crop, child_start_mean, box_length=args.data.coarse_aff.c2f.crop_box_length)
            parent_pcd_cropped, parent_batch_idx, n_parent_pts_per_crop = out_parent_pcd_cropped

            # TODO: handle cases where we crop and there are too few points? 
            mean_batch_idx = parent_batch_idx.reshape(-1, 1).repeat((1, 3)).long()
            parent_crop_mean_batch = scatter_mean(parent_pcd_cropped, mean_batch_idx, dim=0)
            
            parent_pcd_cropped_cent = parent_pcd_cropped.clone().detach()
            for b in range(child_start_mean.shape[0]):
                b_idxs = torch.where(parent_batch_idx == b)[0]
                parent_pcd_cropped_cent[b_idxs] -= parent_crop_mean_batch[b].reshape(1, 3).repeat((b_idxs.shape[0], 1))

            if args.data.coarse_aff.c2f.get('parent_crop_same_n') is not None:
                if args.data.coarse_aff.c2f.parent_crop_same_n:
                    parent_pcd_cropped_cent = pad_flat_pcd_batch(parent_pcd_cropped_cent, parent_batch_idx, n_parent_pts_per_crop, n_child_pts, pcd_default=parent_pcd_to_crop).float().cuda()
                    parent_pcd_cropped = pad_flat_pcd_batch(parent_pcd_cropped, parent_batch_idx, n_parent_pts_per_crop, n_child_pts, pcd_default=parent_pcd_to_crop).float().cuda()
        
            # get new scene normalization params
            # scene_mean = parent_crop_mean_batch.reshape(-1, 3)
            pmean = parent_crop_mean_batch.reshape(-1, 3)
            cmean = child_start_mean
            # scene_mean = cmean
            scene_mean = torch.mean(torch.stack([pmean, cmean], 0), 0)
            
            pcd_full = torch.cat((parent_pcd_cropped, child_pcd_final_pred), dim=1)
            pfull_max = torch.max(pcd_full, dim=1)[0]
            pfull_min = torch.min(pcd_full, dim=1)[0]
            scene_scale = 1.0 / torch.max(pfull_max - pfull_min, dim=-1)[0]
            # scene_scale = 1.0 / (args.data.coarse_aff.c2f.crop_box_length * 2.0)
        except (RuntimeError, IndexError) as e:
        # else:
            print(f'[C2C pred util] Exception: {e}')
            print('here')
            from IPython import embed; embed()

            parent_pcd_cropped = fps_downsample(parent_pcd_to_crop, n_child_pts)
            # parent_pcd_cropped = parent_pcd_to_crop
            parent_crop_mean_batch = torch.mean(parent_pcd_cropped, dim=1)
            parent_pcd_cropped_cent = parent_pcd_cropped - parent_crop_mean_batch.reshape(-1, 1, 3).repeat((1, n_child_pts, 1))

            pmean = parent_crop_mean_batch.reshape(-1, 3)
            cmean = child_start_mean
            # scene_mean = cmean
            scene_mean = torch.mean(torch.stack([pmean, cmean], 0), 0)
            
            pcd_full = torch.cat((parent_pcd_cropped, child_pcd_final_pred), dim=1)
            pfull_max = torch.max(pcd_full, dim=1)[0]
            pfull_min = torch.min(pcd_full, dim=1)[0]
            scene_scale = 1.0 / torch.max(pfull_max - pfull_min, dim=-1)[0]

            # scene_mean = coarse_aff_mi['scene_mean']
            # scene_scale = coarse_aff_mi['scene_scale']
    else:
        # get new scene normalization params
        scene_mean = coarse_aff_mi['scene_mean']
        scene_scale = coarse_aff_mi['scene_scale']
    
    # get new scene-normalized point clouds
    # parent_pcd_scene_norm = parent_pcd_cropped * scene_scale
    parent_pcd_scene_norm = (parent_pcd_cropped - scene_mean.reshape(-1, 1, 3).repeat((1, n_child_pts, 1))) * scene_scale[:, None, None]
    child_start_pcd_scene_norm = (child_pcd_final_pred - scene_mean.reshape(-1, 1, 3).repeat((1, n_child_pts, 1))) * scene_scale[:, None, None]
    child_final_pcd_scene_norm = (coarse_aff_gt['child_final_pcd'] - scene_mean.reshape(-1, 1, 3).repeat((1, n_child_pts, 1))) * scene_scale[:, None, None]

    child_pcd_final_pred_cent = child_pcd_final_pred.detach().clone() - child_final_pred_mean[:, None, :].repeat((1, n_child_pts, 1))

    # update the new model input
    coarse_aff_mi_new = {}
    coarse_aff_mi_new['parent_start_pcd_sn'] = parent_pcd_scene_norm.detach()
    coarse_aff_mi_new['child_start_pcd_sn'] = child_start_pcd_scene_norm
    coarse_aff_mi_new['parent_start_pcd'] = parent_pcd_cropped_cent
    coarse_aff_mi_new['child_start_pcd'] = child_pcd_final_pred_cent
    coarse_aff_mi_new['parent_start_pcd_mean_sn'] = torch.mean(parent_pcd_scene_norm, 1)
    coarse_aff_mi_new['child_start_pcd_mean_sn'] = torch.mean(child_start_pcd_scene_norm, 1)
    coarse_aff_mi_new['parent_start_pcd_mean'] = parent_crop_mean_batch
    coarse_aff_mi_new['child_start_pcd_mean'] = child_final_pred_mean
    coarse_aff_mi_new['scene_mean'] = scene_mean
    # coarse_aff_mi_new['scene_scale'] = torch.Tensor([1.0 / (args.data.coarse_aff.c2f.crop_box_length * 2.0)]).repeat(bs).float().cuda()  
    coarse_aff_mi_new['scene_scale'] = scene_scale

    # update the ground truth, based on new scene normalization params
    new_cpcd_final_mean_raster_index = three_util.coordinate2index(
        three_util.normalize_3d_coordinate(torch.mean(child_final_pcd_scene_norm, 1).reshape(-1, 1, 3)),
        voxel_reso_grid, '3d').squeeze()
    new_cpcd_final_mean_raster_coord = voxel_grid[new_cpcd_final_mean_raster_index]

    if args.debug:
        db_idx = 0

        mc_iter_name = 'scene/c2c_pred_util'

        # show this with the ground truth best voxel
        util.meshcat_pcd_show(
            mc_vis,
            child_final_pcd_scene_norm[db_idx].detach().cpu().numpy(),
            (0, 255, 0),
            f'{mc_iter_name}/post_voxel_aff/child_final_pcd_scene_norm',
            size=MC_SIZE
        )

        util.meshcat_pcd_show(
            mc_vis,
            voxel_grid.detach().cpu().numpy(),
            (0, 0, 0),
            f'{mc_iter_name}/post_voxel_aff/voxel_grid_pts',
            size=MC_SIZE/3.0
        )

        # show this with the ground truth best voxel
        util.meshcat_pcd_show(
            mc_vis,
            new_cpcd_final_mean_raster_coord[db_idx].detach().cpu().numpy().reshape(1, 3),
            (0, 255, 0),
            f'{mc_iter_name}/post_voxel_aff/ground_truth_voxel_coord',
            size=MC_SIZE*5
        )

        # print('here with voxel coord in pred_util c2c') 
        # from IPython import embed; embed()

    new_small_rotmat = torch.bmm(coarse_aff_gt['rot_mat'], torch.inverse(out_rot_mat.detach().clone()))
    new_rotmat_grid_index = get_rotation_grid_index_torch(new_small_rotmat, rot_grid)
        
    euler_bins_per_axis = args.data.euler_bins_per_axis
    new_small_rotmat_euler = matrix_to_euler_angles(new_small_rotmat, 'ZYX')
    new_euler_onehot = get_euler_onehot_torch(new_small_rotmat_euler, euler_bins_per_axis)

    new_trans = child_final_mean - child_start_mean

    # update the ground truth
    coarse_aff_gt_new = {}
    coarse_aff_gt_new['child_centroid_voxel_index'] = new_cpcd_final_mean_raster_index
    coarse_aff_gt_new['child_centroid_voxel_coord'] = new_cpcd_final_mean_raster_coord
    coarse_aff_gt_new['trans'] = new_trans
    coarse_aff_gt_new['rot_mat'] = new_small_rotmat
    coarse_aff_gt_new['rot_mat_grid_index'] = new_rotmat_grid_index
    coarse_aff_gt_new['euler_onehot'] = new_euler_onehot
    coarse_aff_gt_new['child_final_pcd'] = coarse_aff_gt['child_final_pcd']
    coarse_aff_gt_new['child_final_pcd_sn'] = child_final_pcd_scene_norm

    coarse_aff_mi_new = detach_dict(coarse_aff_mi_new)
    coarse_aff_gt_new = detach_dict(coarse_aff_gt_new)
            
    return coarse_aff_mi_new, coarse_aff_gt_new


def refine_pose_to_refine_pose(refine_pose_mi, refine_pose_gt, args, mc_vis=None): 
    raise NotImplementedError

