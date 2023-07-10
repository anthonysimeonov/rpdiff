import copy
import numpy as np
import torch

import trimesh
import matplotlib.pyplot as plt

from rpdiff.utils import util
from rpdiff.utils.torch_util import angle_axis_to_rotation_matrix, transform_pcd_torch
from rpdiff.utils.torch3d_util import matrix_to_quaternion, quaternion_to_matrix, matrix_to_axis_angle
from rpdiff.utils.torch_scatter_utils import FPSDownSample


def single_shot_regression(mc_vis, parent_pcd, child_pcd, rpdiff_dict, policy_model, 
                           viz=False, *args, **kwargs):
    """
    Directly predict the relative transformation in one shot
    """
    parent_rpdiff = rpdiff_dict['parent']
    child_rpdiff = rpdiff_dict['child']
    
    rix1 = np.random.permutation(parent_pcd.shape[0])
    rix2 = np.random.permutation(child_pcd.shape[0])
    parent_pcd = parent_pcd[rix1[:1500]]
    child_pcd = child_pcd[rix2[:1500]]

    prpdiff_pcd, prpdiff_coords = copy.deepcopy(util.center_pcd(parent_pcd)), copy.deepcopy(util.center_pcd(child_pcd, ref_pcd=parent_pcd))
    crpdiff_pcd, crpdiff_coords = copy.deepcopy(util.center_pcd(child_pcd)), copy.deepcopy(util.center_pcd(parent_pcd, ref_pcd=child_pcd))

    # obtain descriptor values for the input (parent RPDIFF evaluated at child pcd)
    parent_rpdiff_mi = {}
    parent_rpdiff_mi['point_cloud'] = torch.from_numpy(prpdiff_pcd).float().cuda().reshape(1, -1, 3)
    parent_rpdiff_mi['coords'] = torch.from_numpy(prpdiff_coords).float().cuda().reshape(1, -1, 3)

    # obtain descriptor values for the input (child RPDIFF evaluated at parent pcd)
    child_rpdiff_mi = {}
    child_rpdiff_mi['point_cloud'] = torch.from_numpy(crpdiff_pcd).float().cuda().reshape(1, -1, 3)
    child_rpdiff_mi['coords'] = torch.from_numpy(crpdiff_coords).float().cuda().reshape(1, -1, 3)

    parent_latent = parent_rpdiff.model.extract_latent(parent_rpdiff_mi).detach()  # assumes we have already centered based on the parent
    parent_rpdiff_child_desc = parent_rpdiff.model.forward_latent(parent_latent, parent_rpdiff_mi['coords']).detach()

    child_latent = child_rpdiff.model.extract_latent(child_rpdiff_mi).detach()  # assumes we have already centered based on the child
    child_rpdiff_parent_desc = child_rpdiff.model.forward_latent(child_latent, child_rpdiff_mi['coords']).detach()

    # prepare inputs to the policy
    policy_mi = {}
    policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud']
    policy_mi['child_start_pcd'] = child_rpdiff_mi['point_cloud']
    # policy_mi['parent_start_pcd_mean'] = torch.mean(parent_rpdiff_mi['point_cloud'], axis=1)
    # policy_mi['child_start_pcd_mean'] = torch.mean(child_rpdiff_mi['point_cloud'], axis=1)
    policy_mi['parent_start_pcd_mean'] = torch.from_numpy(np.mean(parent_pcd, axis=0)).float().cuda().reshape(1, -1, 3)
    policy_mi['child_start_pcd_mean'] = torch.from_numpy(np.mean(child_pcd, axis=0)).float().cuda().reshape(1, -1, 3)
    policy_mi['parent_rpdiff_child_desc'] = parent_rpdiff_child_desc
    policy_mi['child_rpdiff_parent_desc'] = child_rpdiff_parent_desc
    policy_mi['parent_latent'] = parent_latent
    policy_mi['child_latent'] = child_latent

    model_output = policy_model(policy_mi)
    
    out_rot_mat = model_output['rot_mat'].detach().cpu().numpy().reshape(3, 3)
    out_trans = model_output['trans'].detach().cpu().numpy().reshape(1, 3)

    tf1 = np.eye(4); tf1[:-1, -1] = -1.0 * np.mean(child_pcd, axis=0)
    tf2 = np.eye(4); tf2[:-1, :-1] = out_rot_mat
    tf3 = np.eye(4); tf3[:-1, -1] = np.mean(child_pcd, axis=0) + out_trans
    out_tf = np.matmul(tf3, np.matmul(tf2, tf1))

    print('Out Trans: ', out_trans)
    print('Out Rot: ', out_rot_mat)
    print('Out TF: ', out_tf)

    return out_tf


def iterative_regression(mc_vis, parent_pcd, child_pcd, rpdiff_dict, policy_model, 
                         viz=False, n_iters=10, return_all_child_pcds=False, *args, **kwargs):
    """
    Iteratively predict the relative transformation in one shot,
    applying each prediction to the shape and feeding the transformed
    shape back in as input for another step
    """
    out_tf_list = []
    out_tf_full = np.eye(4)
    out_child_pcd_list = []
    current_child_pcd = copy.deepcopy(child_pcd)
    for it in range(n_iters):
        out_tf = single_shot_regression(mc_vis, parent_pcd, current_child_pcd, rpdiff_dict, policy_model)
        current_child_pcd = util.transform_pcd(current_child_pcd, out_tf)
        out_child_pcd_list.append(current_child_pcd)
        out_tf_list.append(out_tf)

        out_tf_full = np.matmul(out_tf, out_tf_full)

    if return_all_child_pcds:
        return out_tf_full, current_child_pcd
    else:
        return out_tf_full


def energy_optimization(mc_vis, parent_pcd, child_pcd, rpdiff_dict, policy_model, 
                        viz=False, n_iters=10, return_all_child_pcds=False, *args, **kwargs):
    """
    Optimize the transformation with respect to a learned energy function,
    run a few times and pick the one with the best overall cost
    """
    raise NotImplementedError


def single_shot_regression_feat_combine(mc_vis, parent_pcd, child_pcd, feat_model, policy_model, 
                           viz=False, *args, **kwargs):
    """
    Directly predict the relative transformation in one shot
    """
    rix1 = np.random.permutation(parent_pcd.shape[0])
    rix2 = np.random.permutation(child_pcd.shape[0])
    parent_pcd = parent_pcd[rix1[:1500]]
    child_pcd = child_pcd[rix2[:1500]]

    prpdiff_pcd = copy.deepcopy(util.center_pcd(parent_pcd)) 
    crpdiff_pcd = copy.deepcopy(util.center_pcd(child_pcd)) 

    # obtain descriptor values for the input (parent)
    parent_rpdiff_mi = {}
    parent_rpdiff_mi['point_cloud'] = torch.from_numpy(prpdiff_pcd).float().cuda().reshape(1, -1, 3)

    # obtain descriptor values for the input (child)
    child_rpdiff_mi = {}
    child_rpdiff_mi['point_cloud'] = torch.from_numpy(crpdiff_pcd).float().cuda().reshape(1, -1, 3)

    model_dict = None
    if isinstance(policy_model, dict):
        model_dict = policy_model
        policy_model = model_dict['trans']
        rot_policy_model = model_dict['rot']
    else:
        rot_policy_model = policy_model

    policy_model = policy_model.eval()
    rot_policy_model = rot_policy_model.eval()

    if isinstance(feat_model, dict):
        feat_model_p = feat_model['parent']
        feat_model_c = feat_model['child']
    else:
        feat_model_p = feat_model_c = feat_model

    parent_local_latent = feat_model_p.extract_local_latent(parent_rpdiff_mi, new=True).detach()
    parent_global_latent = feat_model_p.extract_global_latent(parent_rpdiff_mi).detach()

    child_local_latent = feat_model_c.extract_local_latent(child_rpdiff_mi, new=True).detach()
    child_global_latent = feat_model_c.extract_global_latent(child_rpdiff_mi).detach()

    # prepare inputs to the policy
    policy_mi = {}
    policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud']
    policy_mi['child_start_pcd'] = child_rpdiff_mi['point_cloud']
    policy_mi['parent_start_pcd_mean'] = torch.from_numpy(np.mean(parent_pcd, axis=0)).float().cuda().reshape(1, 3)
    policy_mi['child_start_pcd_mean'] = torch.from_numpy(np.mean(child_pcd, axis=0)).float().cuda().reshape(1, 3)

    # latents from point cloud encoder
    policy_mi['parent_point_latent'] = parent_local_latent
    policy_mi['child_point_latent'] = child_local_latent
    policy_mi['parent_global_latent'] = parent_global_latent
    policy_mi['child_global_latent'] = child_global_latent

    # get rotation output prediction
    rot_model_output = rot_policy_model(policy_mi)

    # make a rotated version of the original point cloud
    child_start_pcd_original = policy_mi['child_start_pcd'].clone().detach()
    child_pcd_final_pred = torch.bmm(rot_model_output['rot_mat'], policy_mi['child_start_pcd'].transpose(1, 2)).transpose(2, 1)
    child_pcd_rot = child_pcd_final_pred.detach()
    
    # reencode this rotated point cloud to get new descriptor features
    child_rpdiff_mi['point_cloud'] = child_pcd_rot
    child_local_latent = feat_model_c.extract_local_latent(child_rpdiff_mi, new=True).detach()
    child_global_latent = feat_model_c.extract_global_latent(child_rpdiff_mi).detach()
    
    # new policy input based on these features of rotated shape
    policy_mi['child_point_latent'] = child_local_latent
    policy_mi['child_global_latent'] = child_global_latent
    policy_mi['child_start_pcd'] = child_pcd_rot
    
    # get output for translation, and combine outputs with rotation prediction
    model_output = policy_model(policy_mi)
    # print("here after model_output")
    # from IPython import embed; embed()

    ##############################################

    if 'viz_attn' in kwargs.keys():
        if kwargs['viz_attn']:

            mc_vis['scene/anchor_pts'].delete()
            mc_vis['scene/impt_pts'].delete()
            for i in range(10):
                mc_vis[f'scene/anchor_{i}'].delete()

            attn_viz = policy_model.transformer.blocks[-1].attention.attn[0, 0]
            pos_viz = policy_model.transformer.blocks[-1].attention.pos.detach().cpu().numpy().squeeze()

            viz_pts = pos_viz.shape[0]
            viz1_pts = int((viz_pts - 1) / 2)

            util.meshcat_pcd_show(mc_vis, pos_viz[:viz1_pts], (255, 0, 0), 'scene/pos/parent_pts')
            util.meshcat_pcd_show(mc_vis, pos_viz[viz1_pts:-1], (0, 0, 255), 'scene/pos/child_pts')

            util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/real/parent_pts')
            util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 255), 'scene/real/child_pts')

            k = 1000
            topk_values = torch.topk(attn_viz.flatten(), k)[0]
            topk_2d_inds = []
            for val in topk_values:
                ind = (attn_viz == val).nonzero()
                topk_2d_inds.append(ind)
            
            # cls_idx = 1024
            cls_idx = 256
            pt_idxs = []
            pt_pair_idxs = []
            for i, inds in enumerate(topk_2d_inds):
                idxs = inds.detach().cpu().numpy().squeeze().tolist()
                if idxs[0] != cls_idx:
                    # pt_pair_idxs.append((i, idxs))
                    if isinstance(idxs[0], list):
                        for val in idxs:
                            # pt_pair_idxs.append((i, [val]))
                            pt_pair_idxs.append((i, val))
                    else:
                        pt_pair_idxs.append((i, idxs))
                else:
                    pt_idxs.append(idxs[1])
                # print(idxs)
                # pt_idxs.append(idxs[1])

            # for the pairs, form groups with same points
            same_pts_dict = {}
            for i, pt_pair_idx in enumerate(pt_pair_idxs):

                # from IPython import embed; embed()
                try:
                    val_ind = pt_pair_idx[0]
                    val_ind_2d = topk_2d_inds[val_ind].squeeze()
                    attn_score = topk_values[val_ind].detach().cpu().item()
                    # attn_score = attn_viz[val_ind_2d[0], val_ind_2d[1]]
                    ix1, ix2 = pt_pair_idx[1]
                except Exception as e:
                    print(e)
                    from IPython import embed; embed()

                try:
                    if ix1 not in same_pts_dict.keys():
                        same_pts_dict[ix1] = []
                    # same_pts_dict[ix1].append(ix2)
                    # same_pts_dict[ix1].append((ix1, ix2))
                    same_pts_dict[ix1].append((ix1, ix2, attn_score))

                    if ix2 not in same_pts_dict.keys():
                        same_pts_dict[ix2] = []
                    # same_pts_dict[ix2].append(ix1)
                    same_pts_dict[ix2].append((ix1, ix2, attn_score))
                except Exception as e:
                    print(e)
                    from IPython import embed; embed()
            
            # rank these by which have the most points
            num_matches = []
            matching_points = []
            for k, v in same_pts_dict.items():
                num = len(v)
                num_matches.append(num)
                # matching_points.append(v)  # when we save only the index, no score
                matching_points.append(v)  # converts the score that we save

            num_matches = np.asarray(num_matches)
            matching_points = np.asarray(matching_points)

            ranked_by_num = np.argsort(num_matches)[::-1]
            # top_matching_points = [matching_points[ranked_by_num[0]]]
            top_matching_points = [matching_points[ranked_by_num[ind]] for ind in range(5)]
            # top_matching_points = matching_points[np.where(num_matches > 1)[0]].tolist()
            # top_matching_points = matching_points[np.where(num_matches > 3)[0]].tolist()

            anchor_pt_list = []
            key_pt_list = []
            score_list = []
            for i in range(len(top_matching_points)):
                pt_list = top_matching_points[i]
                anchor_pt = pt_list[0][1]  # we just know it's the second coordinate
                anchor_pt_list.append(anchor_pt)

                kp_list = []
                kp_score_list = []
                for pt in pt_list:
                    kp = pt[0]  
                    kp_list.append(kp)
                    score = pt[-1]
                    kp_score_list.append(score)

                key_pt_list.append(kp_list)
                score_list.append(kp_score_list)

            # sph_size = 0.005
            sph_size = 0.05
            cmap1 = plt.get_cmap('plasma')
            cmap2 = plt.get_cmap('viridis')
            color_list1 = cmap1(np.linspace(0, 1, len(anchor_pt_list)))[::-1]
            for i, anchor_idx in enumerate(anchor_pt_list):
                anchor = pos_viz[anchor_idx]
                sph = trimesh.creation.uv_sphere(sph_size)
                sph.apply_translation(anchor)
                color = (color_list1[i][:-1] * 255).astype(int)
                util.meshcat_trimesh_show(mc_vis, f'scene/anchor_pts/pt_{i}', sph, color=tuple(color))

                keypoint_idxs = key_pt_list[i]
                scores = score_list[i]
                keypoints = pos_viz[keypoint_idxs]
                color_list2 = cmap2(np.linspace(0, 1, len(keypoint_idxs)))[::-1]
                for j, kp in enumerate(keypoints):
                    score = scores[j]
                    sph = trimesh.creation.uv_sphere(sph_size)
                    sph.apply_translation(kp)
                    color = (color_list2[j][:-1] * 255).astype(int)
                    util.meshcat_trimesh_show(mc_vis, f'scene/anchor_{i}/pt_{j}', sph, color=tuple(color))

                    print('score: ', score, 'color: ', color, 'j: ', j)
                    
            impt_pts = pos_viz[pt_idxs]
            sph_list = []
            cmap = plt.get_cmap('plasma')
            color_list = cmap(np.linspace(0, 1, len(pt_idxs)))[::-1]
            for i, pt in enumerate(impt_pts):
                sph = trimesh.creation.uv_sphere(sph_size)
                sph.apply_translation(pt)
                color = (color_list[i][:-1] * 255).astype(int)
                util.meshcat_trimesh_show(mc_vis, f'scene/impt_pts/pt_{i}', sph, color=tuple(color))
                sph_list.append(sph)


            # from IPython import embed; embed()


            # mc_vis['scene/impt_pts'].delete()
            # mc_vis['scene/anchor_pts'].delete()
            # for i in range(len(anchor_pt_list)):
            #     mc_vis[f'scene/anchor_{i}'].delete()

            # # which points are the important points attending to?
            # # impt_pt_att_idxs = []
            # impt_pt_att_idxs = {}
            # # for idx in pt_idxs:
            # for idx in [pt_idxs[1]]:
            #     attn_vec = attn_viz[:, idx].detach().cpu().numpy().squeeze()
            #     attn_vec_inds_sorted = np.argsort(attn_vec)[::-1][1:] #[1:100]
            #     
            #     color_list3 = cmap2(np.linspace(0, 1, len(attn_vec_inds_sorted)))[::-1]
            #     for j, ind in enumerate(attn_vec_inds_sorted):
            #         # pt = pos_viz[idx]
            #         pt = pos_viz[ind]
            #         score = attn_vec[ind]
            #         color = (color_list3[j][:-1] * 255).astype(int)
            #         sph = trimesh.creation.uv_sphere(0.005)
            #         sph.apply_translation(pt)
            #         util.meshcat_trimesh_show(mc_vis, f'scene/attn_pts/{ind}', sph, color=tuple(color))
            #         print('score: ', score)


            from IPython import embed; embed()

    ##############################################
    model_output['quat'] = rot_model_output['quat']
    model_output['unnorm_quat'] = rot_model_output['unnorm_quat']
    model_output['rot_mat'] = rot_model_output['rot_mat']
 
    out_rot_mat = model_output['rot_mat'].detach().cpu().numpy().reshape(3, 3)
    out_trans = model_output['trans'].detach().cpu().numpy().reshape(1, 3)

    tf1 = np.eye(4); tf1[:-1, -1] = -1.0 * np.mean(child_pcd, axis=0)
    tf2 = np.eye(4); tf2[:-1, :-1] = out_rot_mat
    tf3 = np.eye(4); tf3[:-1, -1] = np.mean(child_pcd, axis=0) + out_trans
    out_tf = np.matmul(tf3, np.matmul(tf2, tf1))

    print('Out Trans: ', out_trans)
    print('Out Rot: ', out_rot_mat)
    print('Out TF: ', out_tf)

    # util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/real/parent_pts')
    # util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 255), 'scene/real/child_pts')
    # from IPython import embed; embed()

    return out_tf


def iterative_regression_feat_combine(
                         mc_vis, parent_pcd, child_pcd, feat_model, policy_model, 
                         viz=False, n_iters=10, return_all_child_pcds=False, *args, **kwargs):
    """
    Iteratively predict the relative transformation in one shot,
    applying each prediction to the shape and feeding the transformed
    shape back in as input for another step
    """
    out_tf_list = []
    out_tf_full = np.eye(4)
    out_child_pcd_list = []
    current_child_pcd = copy.deepcopy(child_pcd)

    # for it in range(n_iters):
    #     mc_vis[f'scene/real/child_pts_{it}'].delete()

    util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/real/parent_pts')
    util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 0), 'scene/real/child_pts_init')
    cmap = plt.get_cmap('plasma')
    color_list = cmap(np.linspace(0, 1, n_iters, dtype=np.float32))[::-1]

    for it in range(n_iters):
        out_tf = single_shot_regression_feat_combine(mc_vis, parent_pcd, current_child_pcd, feat_model, policy_model, *args, **kwargs)
        current_child_pcd = util.transform_pcd(current_child_pcd, out_tf)

        # color = (color_list[it][:-1] * 255).astype(int).tolist()
        color = (color_list[it][:-1] * 255).astype(np.int8).tolist()
        # util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 255), 'scene/real/child_pts_{it}')
        # util.meshcat_pcd_show(mc_vis, current_child_pcd, color, f'scene/real/child_pts_{it}')

        out_child_pcd_list.append(current_child_pcd)
        out_tf_list.append(out_tf)

        out_tf_full = np.matmul(out_tf, out_tf_full)

    # from IPython import embed; embed()
    if return_all_child_pcds:
        return out_tf_full, current_child_pcd
    else:
        return out_tf_full


def single_shot_regression_scene_combine_sc(mc_vis, parent_pcd, child_pcd, scene_model_dict, feat_model, policy_model, success_model, 
                           grid_pts, rot_grid=None, run_affordance=False, viz=False, *args, **kwargs):
    """
    Directly predict the relative transformation in one shot
    """
    if rot_grid is None:
        rot_grid = util.generate_healpix_grid(size=1e4) 
    
    scene_model = scene_model_dict['scene_model'].eval()
    aff_model = scene_model_dict['aff_model'].eval()

    child_pcd_original = child_pcd.copy()

    rix1 = np.random.permutation(parent_pcd.shape[0])
    rix2 = np.random.permutation(child_pcd.shape[0])
    # parent_pcd = parent_pcd[rix1[:10000]]
    # child_pcd = child_pcd[rix2[:10000]]

    model_dict = None
    if isinstance(policy_model, dict):
        model_dict = policy_model
        policy_model = model_dict['trans']
        rot_policy_model = model_dict['rot']
    else:
        rot_policy_model = policy_model

    policy_model = policy_model.eval()
    rot_policy_model = rot_policy_model.eval()

    if isinstance(feat_model, dict):
        feat_model_p = feat_model['parent']
        feat_model_c = feat_model['child']
    else:
        feat_model_p = feat_model_c = feat_model

    success_model = success_model.eval()

    # prep the region we will use for checking if final translations are valid (i.e., close enough to the parent)
    parent_pcd_obb = trimesh.PointCloud(parent_pcd).bounding_box_oriented.to_mesh()
    
    delta_child_world = np.zeros(3)
    table_mean = np.array([0.35, 0.0, np.min(parent_pcd[:, 2])])
    table_extents = np.array([0.7, 1.2, 0])
    table_scale = 1 / np.max(table_extents)
    if run_affordance:
        pscene_pcd = (copy.deepcopy(parent_pcd) - table_mean ) * table_scale

        parent_scene_mi = {}
        parent_scene_mi['point_cloud'] = torch.from_numpy(pscene_pcd[rix1[:10000]]).float().cuda().reshape(1, -1, 3)
        # parent_scene_mi['point_cloud'] = torch.from_numpy(pscene_pcd).float().cuda().reshape(1, -1, 3)

        k_val = 5
        scene_grid_latent = scene_model.extract_latent(parent_scene_mi)  # dict with keys 'grid', 'xy', 'xz', 'yz'
        # scene_model_output = aff_model(scene_grid_latent['grid'])
        fea_grid = scene_grid_latent['grid'].permute(0, 2, 3, 4, 1)
        scene_model_output = aff_model(fea_grid)
        vals, inds = torch.topk(scene_model_output['voxel_affordance'][0], k=k_val)
        inds_np = inds.detach().cpu().numpy()
        voxel_pts = grid_pts[inds_np]
        world_pts = (voxel_pts / table_scale) + table_mean

        # util.meshcat_pcd_show(mc_vis, pscene_pcd, (255, 0, 255), 'scene/infer/parent_pcd_norm')
        sz_base = 1.1/32
        for i, pt in enumerate(world_pts):
            box = trimesh.creation.box([sz_base]*3).apply_translation(pt)
            # box = trimesh.creation.box([sz_base * 2 * (len(inds) - idx) / len(inds)]*3).apply_translation(pt)
            # print(sm_vals[idx])
            util.meshcat_trimesh_show(mc_vis, f'scene/voxel_grid/{i}', box, opacity=0.3)

        valid_world_pts = parent_pcd_obb.contains(world_pts)
        invalid_world_idx = np.where(np.logical_not(valid_world_pts))[0]
        world_pts[invalid_world_idx] = np.mean(parent_pcd, axis=0)
    else:
        k_val = 1
        world_pts = np.mean(child_pcd, axis=0).reshape(1, 3)

    # build the size of the bounding box we will use
    child_mean = np.mean(child_pcd, axis=0)
    child_pcd_scaled = child_pcd - child_mean
    child_pcd_scaled *= 1.25
    child_pcd_scaled = child_pcd_scaled + child_mean
    child_bb = trimesh.PointCloud(child_pcd_scaled).bounding_box.to_mesh()
    max_length = np.max(child_bb.extents) / 2

    # make separate crops of the parent point cloud
    M = k_val
    N_crop = 2048
    N_parent = parent_pcd.shape[0]
    # pscene_pcd_t = torch.from_numpy(pscene_pcd).float().cuda()
    pscene_pcd_t = torch.from_numpy(parent_pcd).float().cuda()
    parent_cropped_pcds = torch.empty((M, N_crop, 3)).float().cuda()
    world_pts_t = torch.from_numpy(world_pts).float().cuda()
    for i in range(M):
        voxel_3d_pt = world_pts_t[i].reshape(1, 3)  # 1 x 3
        voxel_high = voxel_3d_pt + max_length
        voxel_low = voxel_3d_pt - max_length
        high, low = voxel_high.repeat((N_parent, 1)), voxel_low.repeat((N_parent, 1))
        # high, low = voxel_3d_pt.repeat((1, N_parent, 1)) sample_low_t.repeat((1, N_parent, 1))
        parent_crop_idx = torch.where(torch.logical_and(pscene_pcd_t < high, pscene_pcd_t > low).all(-1))[0]
        parent_crop_pcd = pscene_pcd_t[parent_crop_idx]

        rix = torch.randperm(parent_crop_pcd.shape[0])
        
        if parent_crop_pcd.shape[0] == 0:
            print('!! Parent crop had zero points !!')
            # from IPython import embed; embed()
            rix2 = torch.randperm(pscene_pcd_t.shape[0])
            parent_cropped_pcds[i] = pscene_pcd_t[rix2[:N_crop]]
            
        if parent_crop_pcd.shape[0] < N_crop:
            while True:
                print(f'Parent pcd crop shape: {parent_crop_pcd.shape[0]}')
                parent_crop_pcd = torch.cat((parent_crop_pcd, parent_crop_pcd[rix[:100]]), dim=0)
                if parent_crop_pcd.shape[0] >= N_crop:
                    break
                rix = torch.randperm(parent_crop_pcd.shape[0])
        else:
            parent_crop_pcd = parent_crop_pcd[rix[:N_crop]]

        parent_cropped_pcds[i] = parent_crop_pcd

    # downsample child pcd to right number of points
    child_pcd_t = torch.from_numpy(child_pcd).float().cuda()
    rix = torch.randperm(child_pcd.shape[0])
    child_pcd_ds = child_pcd_t[rix[:N_crop]]

    # make copies of child point cloud and move to the location of parent crops (voxel positions)
    child_pcd_batch = child_pcd_ds.reshape(1, -1, 3).repeat((M, 1, 1))  # M x N_child x 3
    child_mean_batch = child_pcd_t.mean(0).reshape(1, 1, 3).repeat((1, child_pcd_ds.shape[0], 1))
    # delta_trans_batch = world_pts_t.reshape(-1, 1, 3).repeat((1, child_pcd_ds.shape[0], 1))[:M]  # M x N_child x 3
    delta_trans_batch = world_pts_t.reshape(-1, 1, 3).repeat((1, child_pcd_ds.shape[0], 1))[:M] - child_mean_batch.repeat((M, 1, 1)) # M x N_child x 3
    child_pcd_cent_batch = child_pcd_batch - child_mean_batch

    # sample a batch of rotations and apply to batch
    if run_affordance:
        rand_rot_idx = np.random.randint(rot_grid.shape[0], size=M)
        rand_rot_init = matrix_to_axis_angle(torch.from_numpy(rot_grid[rand_rot_idx])).float()
        rand_mat_init = angle_axis_to_rotation_matrix(rand_rot_init)
        rand_mat_init = rand_mat_init.squeeze().float().cuda()
    else:
        rand_mat_init = torch.eye(4).reshape(-1, 4, 4).float().cuda()
    
    # apply voxel location offsets (after un-centering with mean)
    child_pcd_rot_batch = transform_pcd_torch(child_pcd_cent_batch, rand_mat_init)
    child_pcd_trans_batch = child_pcd_rot_batch + delta_trans_batch + child_mean_batch # M x N_child x 3

    # for ii in range(M):
    #     child_pcd_trans_viz = child_pcd_trans_batch[ii].detach().cpu().numpy().squeeze()
    #     parent_crop_pcd_trans_viz = parent_cropped_pcds[ii].detach().cpu().numpy().squeeze()
    #     util.meshcat_pcd_show(mc_vis, child_pcd_trans_viz, (255, 255, 0), f'scene/infer/child_pcd_trans_batch_{ii}')
    #     util.meshcat_pcd_show(mc_vis, parent_crop_pcd_trans_viz, (255, 0, 255), f'scene/infer/parent_pcd_trans_batch_{ii}')

    parent_crop_mean_batch = torch.mean(parent_cropped_pcds, axis=1).reshape(-1, 1, 3).repeat((1, N_crop, 1))
    prpdiff_pcd = parent_cropped_pcds - parent_crop_mean_batch
    crpdiff_pcd = child_pcd_rot_batch

    # obtain descriptor values for the input (parent)
    parent_rpdiff_mi = {}
    parent_rpdiff_mi['point_cloud'] = prpdiff_pcd
    # parent_rpdiff_mi['point_cloud'] = torch.from_numpy(prpdiff_pcd).float().cuda().reshape(1, -1, 3)

    # obtain descriptor values for the input (child)
    child_rpdiff_mi = {}
    child_rpdiff_mi['point_cloud'] = crpdiff_pcd
    # child_rpdiff_mi['point_cloud'] = torch.from_numpy(crpdiff_pcd).float().cuda().reshape(1, -1, 3)

    parent_local_latent = feat_model_p.extract_local_latent(parent_rpdiff_mi, new=True).detach()
    parent_global_latent = feat_model_p.extract_global_latent(parent_rpdiff_mi).detach()

    child_local_latent = feat_model_c.extract_local_latent(child_rpdiff_mi, new=True).detach()
    child_global_latent = feat_model_c.extract_global_latent(child_rpdiff_mi).detach()

    # prepare inputs to the policy
    policy_mi = {}
    policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud']
    policy_mi['child_start_pcd'] = child_rpdiff_mi['point_cloud']
    # policy_mi['parent_start_pcd_mean'] = parent_crop_mean_batch
    # policy_mi['child_start_pcd_mean'] = child_mean_batch
    policy_mi['parent_start_pcd_mean'] = torch.mean(parent_cropped_pcds, axis=1).reshape(-1, 3)
    policy_mi['child_start_pcd_mean'] = torch.mean(child_pcd_trans_batch, axis=1).reshape(-1, 3)


    # latents from point cloud encoder
    policy_mi['parent_point_latent'] = parent_local_latent
    policy_mi['child_point_latent'] = child_local_latent
    policy_mi['parent_global_latent'] = parent_global_latent
    policy_mi['child_global_latent'] = child_global_latent

    # get rotation output prediction
    rot_model_output_raw = rot_policy_model(policy_mi)  # M x N_queries x 3 x 3
    
    # reshape rotations and batch of point clouds to apply rotations
    rot_mat_queries = rot_model_output_raw['rot_mat']
    N_queries = rot_mat_queries.shape[1]
    child_pcd_rot_qb = child_pcd_rot_batch[:, None, :, :].repeat((1, N_queries, 1, 1))
    child_pcd_post_rot_qb = torch.matmul(rot_mat_queries, child_pcd_rot_qb.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # M x N_queries x N_child x 3

    # reshape into batch, pass into success classifier?

#     print('here with rot_model_output_raw (success classifier)')
#     from IPython import embed; embed()

    # have batch of rotated child point clouds, get translation output prediction (maybe in a loop for batch size)
    post_rot_batch = N_queries * M
    child_pcd_post_rot_qb = child_pcd_post_rot_qb.reshape(post_rot_batch, -1, 3)
    child_rpdiff_mi['point_cloud'] = child_pcd_post_rot_qb
    child_local_latent = feat_model_c.extract_local_latent(child_rpdiff_mi, new=True).detach()
    child_global_latent = feat_model_c.extract_global_latent(child_rpdiff_mi).detach()

    voxel_idx = torch.arange(M).reshape(-1, 1).repeat((1, N_queries)).reshape(post_rot_batch, -1)  # M*N_queries x 1
    # rot_idx = torch.arange(N_queries).reshape(1, -1).repeat((M, 1)).reshape(post_rot_batch, -1)  # M*N_queries x 1
    rot_idx = torch.arange(post_rot_batch).reshape(-1, 1)

    # might have to re-encode things, or just expand the parent point cloud to match the child point cloud
    
    policy_mi['child_point_latent'] = child_local_latent
    policy_mi['child_global_latent'] = child_global_latent
    policy_mi['child_start_pcd'] = child_pcd_post_rot_qb

    # policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud'].repeat((N_queries, 1, 1))
    # policy_mi['parent_start_pcd_mean'] = policy_mi['parent_start_pcd_mean'].repeat((N_queries, 1))
    # policy_mi['child_start_pcd_mean'] = policy_mi['child_start_pcd_mean'].repeat((N_queries, 1))
    policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud'][:, None, :, :].repeat((1, N_queries, 1, 1)).reshape(post_rot_batch, -1, 3)  # M x N_child x 3 -> M x N_queries x N_child x 3 -> M*N_queries x N_child x 3 (in right order)
    policy_mi['parent_start_pcd_mean'] = policy_mi['parent_start_pcd_mean'][:, None, :].repeat((1, N_queries, 1)).reshape(post_rot_batch, -1)  # M x 3 -> M x N_queries x 3 -> M*N_queries x 3 (in right order)
    policy_mi['child_start_pcd_mean'] = policy_mi['child_start_pcd_mean'][:, None, :].repeat((1, N_queries, 1)).reshape(post_rot_batch, -1)
    
    # get output for translation, and combine outputs with rotation prediction
    trans_model_output_raw = policy_model(policy_mi)  # M x N_queries x 3

    trans_queries = trans_model_output_raw['trans'].reshape(post_rot_batch, N_queries, 1, 3).repeat((1, 1, N_crop, 1))
    trans_mean = policy_mi['child_start_pcd_mean'][:, None, None, :].repeat((1, N_queries, N_crop, 1))
    child_pcd_trans_qb = child_pcd_post_rot_qb[:, None, :, :].repeat((1, N_queries, 1, 1))
    child_pcd_post_trans_qb = child_pcd_trans_qb + trans_mean + trans_queries

    voxel_idx_final = voxel_idx.repeat((1, N_queries)).reshape(post_rot_batch*N_queries, -1)  # M*N_queries**2 x 1
    rot_idx_final = rot_idx.repeat((1, N_queries)).reshape(post_rot_batch*N_queries, -1)  # M*N_queries**2 x 1
    # trans_idx_final = torch.arange(N_queries).reshape(1, -1).repeat((post_rot_batch, 1)).reshape(post_rot_batch*N_queries, -1)  # M*N_queries**2 x 1
    trans_idx_final = torch.arange(post_rot_batch*N_queries).reshape(-1, 1)  # M*N_queries**2 x 1
    rot_mat_queries_final = rot_mat_queries.reshape(post_rot_batch, 3, 3)
    trans_queries_final = trans_queries[:, :, 0].reshape(post_rot_batch*N_queries, -1)

    # reshape translations and batch of point clouds to apply translations
    child_pcd_to_sc = child_pcd_post_trans_qb.reshape(post_rot_batch*N_queries, -1, 3)
    child_rpdiff_mi['point_cloud'] = child_pcd_to_sc
    child_local_latent = feat_model_c.extract_local_latent(child_rpdiff_mi, new=True).detach()
    child_global_latent = feat_model_c.extract_global_latent(child_rpdiff_mi).detach()
    
    child_pcd_to_sc_mean = torch.mean(child_pcd_to_sc, axis=1)
    policy_mi['child_point_latent'] = child_local_latent
    policy_mi['child_global_latent'] = child_global_latent
    # policy_mi['child_start_pcd'] = child_pcd_to_sc
    policy_mi['child_start_pcd'] = child_pcd_to_sc - child_pcd_to_sc_mean[:, None, :].repeat((1, N_crop, 1))

    # policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud'].repeat((N_queries**2, 1, 1))
    # policy_mi['parent_start_pcd_mean'] = policy_mi['parent_start_pcd_mean'].repeat((N_queries, 1))

    # put in full parent point cloud for final success classification
    pscene_mean = torch.mean(pscene_pcd_t, axis=0).reshape(1, 3)
    rix_p = torch.randperm(pscene_pcd_t.shape[0])
    pscene_pcd_cent_t = pscene_pcd_t[rix[:N_crop]] - pscene_mean.repeat((N_crop, 1))
    policy_mi['parent_start_pcd'] = pscene_pcd_cent_t.reshape(1, -1, 3).repeat((post_rot_batch*N_queries, 1, 1))
    policy_mi['parent_start_pcd_mean'] = pscene_mean.repeat((post_rot_batch*N_queries, 1))
    policy_mi['child_start_pcd_mean'] = child_pcd_to_sc_mean

    # print('here ready to pass to success classifier')
    # from IPython import embed; embed()
    # from pdb import set_trace; set_trace()

    # break into smaller batches
    sm_bs = 8
    idx = 0
    success_out_list = []

    while True:
        if idx >= policy_mi['child_start_pcd'].shape[0]:
            # print('Done with small batch success model predictions')
            break

        small_policy_mi = {k: v[idx:idx+sm_bs] for k, v in policy_mi.items()}
        small_success_out = success_model(small_policy_mi)

        if run_affordance:
            success_out = small_success_out['success'].detach().cpu().numpy().squeeze().tolist()
        else:
            success_out_nominal = small_success_out['success'].detach().cpu().numpy().squeeze()
            voxel_idx = voxel_idx_final[idx:idx+sm_bs].squeeze()
            trans_idx = trans_idx_final[idx:idx+sm_bs].squeeze()
            out_trans = world_pts_t[voxel_idx].reshape(-1, 3) + trans_queries_final[trans_idx].reshape(-1, 3)
            out_trans_np = out_trans.detach().cpu().numpy()
            success_out_trans_valid = parent_pcd_obb.contains(out_trans_np)
            success_out = success_out_nominal * success_out_trans_valid
        success_out_list.extend(success_out)

        idx += sm_bs 
    
    success_out_all = torch.Tensor(success_out_list).float().cuda()

    # pass into success classifier (have to loop to reduce batch size)
    # success_model_output = success_model(policy_mi)

    # success_argmax = np.argmax(success_out_list)
    # voxel_argmax_idx = voxel_idx_final[success_argmax].squeeze()
    # rot_argmax_idx = rot_idx_final[success_argmax].squeeze()
    # trans_argmax_idx = trans_idx_final[success_argmax].squeeze()

    success_topk_vals, success_topk_idx = torch.topk(success_out_all, k=k_val)
    voxel_topk = voxel_idx_final[success_topk_idx].squeeze()
    rot_topk = rot_idx_final[success_topk_idx].squeeze()
    trans_topk = trans_idx_final[success_topk_idx].squeeze()

    out_voxel_trans = world_pts_t[voxel_topk].reshape(-1, 3)
    out_rot_init = rand_mat_init.reshape(-1, 4, 4)[voxel_topk].reshape(-1, 4, 4)
    # out_rot_init = rand_mat_init[voxel_topk].reshape(-1, 4, 4)
    out_rot = rot_mat_queries_final[rot_topk].reshape(-1, 3, 3)
    out_trans = trans_queries_final[trans_topk].reshape(-1, 3)
    out_trans_full = out_voxel_trans + out_trans

    # colormap for the top-k
    child_pcd_cent = child_pcd - np.mean(child_pcd, axis=0)
    cmap = plt.get_cmap('inferno')
    color_list = cmap(np.linspace(0.1, 0.9, 5, dtype=np.float32))[::-1]
    # util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/infer/parent_pcd')
    tmat_full_list = []
    for ii in range(k_val):
        color = (color_list[ii][:-1] * 255).astype(np.int8).tolist()
        tmat = np.eye(4); tmat[:-1, -1] = out_trans_full[ii].detach().cpu().numpy().squeeze()
        tmat[:-1, :-1] = out_rot[ii].detach().cpu().numpy().squeeze()
        tmat_init = out_rot_init[ii].detach().cpu().numpy().squeeze()

        tmat_full = np.matmul(tmat, tmat_init)
        tmat_full_list.append(tmat_full)
        trans_child_pcd = util.transform_pcd(child_pcd_cent, tmat_full)
        util.meshcat_pcd_show(mc_vis, trans_child_pcd, color, f'scene/infer/success_topk/child_pts_{ii}')

    # check to make sure that the translation is valid (close enough to the parent point cloud)
    out_trans_full_np = out_trans_full.detach().cpu().numpy().reshape(-1, 3)
    valid_trans = parent_pcd_obb.contains(out_trans_full_np) 
    valid_trans_idx = np.where(valid_trans.squeeze())[0]
    # try:
    #     best_valid_trans_idx = valid_trans_idx[0]
    # except Exception as e:
    #     for jj, pt in enumerate(out_trans_full_np):
    #         sph = trimesh.creation.uv_sphere(0.004).apply_translation(pt)
    #         util.meshcat_trimesh_show(mc_vis, f'scene/out_trans_sph_{jj}', sph, (0, 0, 255))
    #     print(e)
    #     print('Failed to get valid trans index')
    #     from IPython import embed; embed()

    tf_cent = np.eye(4); tf_cent[:-1, -1] = -1.0 * np.mean(child_pcd_original, axis=0)
    if not len(valid_trans_idx):
        print('Failed to get valid trans index')
        for jj, pt in enumerate(out_trans_full_np):
            sph = trimesh.creation.uv_sphere(0.004).apply_translation(pt)
            util.meshcat_trimesh_show(mc_vis, f'scene/out_trans_sph_{jj}', sph, (0, 0, 255))
        out_tf = np.eye(4); out_tf[:-1, :-1] = tmat_full_list[0][:-1, :-1]
    else:
        best_valid_trans_idx = valid_trans_idx[0]
        out_tf = np.matmul(tmat_full_list[best_valid_trans_idx], tf_cent)

    return out_tf


def iterative_regression_scene_combine_sc(
                         mc_vis, parent_pcd, child_pcd, scene_model_dict, feat_model, policy_model, policy_model_refine, success_model,
                         grid_pts, rot_grid=None, viz=False, n_iters=10, return_all_child_pcds=False, *args, **kwargs):
    """
    Iteratively predict the relative transformation in one shot,
    applying each prediction to the shape and feeding the transformed
    shape back in as input for another step
    """
    out_tf_list = []
    out_tf_full = np.eye(4)
    out_child_pcd_list = []
    current_child_pcd = copy.deepcopy(child_pcd)

    # for it in range(n_iters):
    #     mc_vis[f'scene/real/child_pts_{it}'].delete()

    util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/real/parent_pts')
    util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 0), 'scene/real/child_pts_init')
    cmap = plt.get_cmap('plasma')
    color_list = cmap(np.linspace(0, 1, n_iters, dtype=np.float32))[::-1]

    for it in range(n_iters):
        run_affordance = it == 0
        if it < 1:
        # if it < (n_iters / 2):
            # coarse (use policy_model)
            out_tf = single_shot_regression_scene_combine_sc(
                mc_vis, parent_pcd, current_child_pcd, 
                scene_model_dict, feat_model, policy_model, success_model, 
                grid_pts, rot_grid=rot_grid, run_affordance=run_affordance, 
                *args, **kwargs)
        else:
            # refine (use policy_model_refine)
            out_tf = single_shot_regression_scene_combine_sc(
                mc_vis, parent_pcd, current_child_pcd, 
                scene_model_dict, feat_model, policy_model_refine, success_model, 
                grid_pts, rot_grid=rot_grid, run_affordance=run_affordance, 
                *args, **kwargs)
        current_child_pcd = util.transform_pcd(current_child_pcd, out_tf)

        color = (color_list[it][:-1] * 255).astype(np.int8).tolist()
        util.meshcat_pcd_show(mc_vis, current_child_pcd, color, f'scene/real/child_pts_{it}')

        out_child_pcd_list.append(current_child_pcd)
        out_tf_list.append(out_tf)

        out_tf_full = np.matmul(out_tf, out_tf_full)

    # from IPython import embed; embed()
    if return_all_child_pcds:
        return out_tf_full, current_child_pcd
    else:
        return out_tf_full


def multistep_regression_scene_combine_sc(
    mc_vis, parent_pcd, child_pcd, scene_model_dict, feat_model, policy_model, policy_model_refine, success_model,
    grid_pts, rot_grid=None, viz=False, n_iters=10, return_all_child_pcds=False, no_parent_crop=False, *args, **kwargs):
    """
    Directly predict the relative transformation in one shot
    """
    if rot_grid is None:
        rot_grid = util.generate_healpix_grid(size=1e4) 
    
    scene_model = scene_model_dict['scene_model'].eval()
    aff_model = scene_model_dict['aff_model'].eval()

    child_pcd_original = child_pcd.copy()

    rix1 = np.random.permutation(parent_pcd.shape[0])
    rix2 = np.random.permutation(child_pcd.shape[0])
    # parent_pcd = parent_pcd[rix1[:10000]]
    # child_pcd = child_pcd[rix2[:10000]]

    model_dict = None
    if isinstance(policy_model, dict):
        model_dict = policy_model
        policy_model = model_dict['trans']
        rot_policy_model = model_dict['rot']
    else:
        rot_policy_model = policy_model

    policy_model = policy_model.eval()
    rot_policy_model = rot_policy_model.eval()

    model_dict_refine = None
    if isinstance(policy_model_refine, dict):
        model_dict_refine = policy_model_refine
        policy_model_refine = model_dict_refine['trans']
        rot_policy_model_refine = model_dict_refine['rot']
    else:
        rot_policy_model_refine = policy_model_refine

    policy_model_refine = policy_model_refine.eval()
    rot_policy_model_refine = rot_policy_model_refine.eval()

    if isinstance(feat_model, dict):
        feat_model_p = feat_model['parent']
        feat_model_c = feat_model['child']
    else:
        feat_model_p = feat_model_c = feat_model

    success_model = success_model.eval()
    
    delta_child_world = np.zeros(3)
    table_mean = np.array([0.35, 0.0, np.min(parent_pcd[:, 2])])
    table_extents = np.array([0.7, 1.2, 0])
    table_scale = 1 / np.max(table_extents)

    # prep the region we will use for checking if final translations are valid (i.e., close enough to the parent)
    parent_pcd_obb = trimesh.PointCloud(parent_pcd).bounding_box_oriented.to_mesh()

    # run_affordance = True
    run_affordance = not no_parent_crop
    if run_affordance:
        pscene_pcd = (copy.deepcopy(parent_pcd) - table_mean ) * table_scale

        parent_scene_mi = {}
        parent_scene_mi['point_cloud'] = torch.from_numpy(pscene_pcd[rix1[:10000]]).float().cuda().reshape(1, -1, 3)
        # parent_scene_mi['point_cloud'] = torch.from_numpy(pscene_pcd).float().cuda().reshape(1, -1, 3)

        k_val = 5
        scene_grid_latent = scene_model.extract_latent(parent_scene_mi)  # dict with keys 'grid', 'xy', 'xz', 'yz'
        # scene_model_output = aff_model(scene_grid_latent['grid'])
        fea_grid = scene_grid_latent['grid'].permute(0, 2, 3, 4, 1)
        scene_model_output = aff_model(fea_grid)
        vals, inds = torch.topk(scene_model_output['voxel_affordance'][0], k=k_val)
        inds_np = inds.detach().cpu().numpy()
        voxel_pts = grid_pts[inds_np]
        world_pts = (voxel_pts / table_scale) + table_mean

        # util.meshcat_pcd_show(mc_vis, pscene_pcd, (255, 0, 255), 'scene/infer/parent_pcd_norm')
        sz_base = 1.1/32
        for i, pt in enumerate(world_pts):
            box = trimesh.creation.box([sz_base]*3).apply_translation(pt)
            # box = trimesh.creation.box([sz_base * 2 * (len(inds) - idx) / len(inds)]*3).apply_translation(pt)
            # print(sm_vals[idx])
            util.meshcat_trimesh_show(mc_vis, f'scene/voxel_grid/{i}', box, opacity=0.3)

        valid_world_pts = parent_pcd_obb.contains(world_pts)
        invalid_world_idx = np.where(np.logical_not(valid_world_pts))[0]
        world_pts[invalid_world_idx] = np.mean(parent_pcd, axis=0)
    else:
        k_val = 1
        world_pts = np.mean(child_pcd, axis=0).reshape(1, 3)

    # build the size of the bounding box we will use
    child_mean = np.mean(child_pcd, axis=0)
    child_pcd_scaled = child_pcd - child_mean
    child_pcd_scaled *= 1.25
    child_pcd_scaled = child_pcd_scaled + child_mean
    child_bb = trimesh.PointCloud(child_pcd_scaled).bounding_box.to_mesh()
    max_length = np.max(child_bb.extents) / 2
    
    # make separate crops of the parent point cloud
    M = k_val
    N_crop = 2048
    N_parent = parent_pcd.shape[0]
    # pscene_pcd_t = torch.from_numpy(pscene_pcd).float().cuda()
    pscene_pcd_t = torch.from_numpy(parent_pcd).float().cuda()
    parent_cropped_pcds = torch.empty((M, N_crop, 3)).float().cuda()
    world_pts_t = torch.from_numpy(world_pts).float().cuda()
    fps_ds = FPSDownSample(N_crop)
    full_parent_pcd_ds = fps_ds.forward(pscene_pcd_t.reshape(1, -1, 3))
    full_parent_pcd_mean = torch.mean(full_parent_pcd_ds, axis=1)
    if no_parent_crop:
        parent_cropped_pcds[0] = fps_ds.forward(pscene_pcd_t.reshape(1, -1, 3))
    else:
        for i in range(M):
            voxel_3d_pt = world_pts_t[i].reshape(1, 3)  # 1 x 3
            voxel_high = voxel_3d_pt + max_length
            voxel_low = voxel_3d_pt - max_length
            high, low = voxel_high.repeat((N_parent, 1)), voxel_low.repeat((N_parent, 1))
            # high, low = voxel_3d_pt.repeat((1, N_parent, 1)) sample_low_t.repeat((1, N_parent, 1))
            parent_crop_idx = torch.where(torch.logical_and(pscene_pcd_t < high, pscene_pcd_t > low).all(-1))[0]
            parent_crop_pcd = pscene_pcd_t[parent_crop_idx]

            rix = torch.randperm(parent_crop_pcd.shape[0])
            
            if parent_crop_pcd.shape[0] == 0:
                print('!! Parent crop had zero points !!')
                # from IPython import embed; embed()
                rix2 = torch.randperm(pscene_pcd_t.shape[0])
                parent_cropped_pcds[i] = pscene_pcd_t[rix2[:N_crop]]
                continue
                
            if parent_crop_pcd.shape[0] < N_crop:
                while True:
                    print(f'Parent pcd crop shape: {parent_crop_pcd.shape[0]}')
                    parent_crop_pcd = torch.cat((parent_crop_pcd, parent_crop_pcd[rix[:100]]), dim=0)
                    if parent_crop_pcd.shape[0] >= N_crop:
                        break
                    rix = torch.randperm(parent_crop_pcd.shape[0])
            else:
                parent_crop_pcd = parent_crop_pcd[rix[:N_crop]]

            parent_cropped_pcds[i] = parent_crop_pcd

    # downsample child pcd to right number of points
    child_pcd_t = torch.from_numpy(child_pcd).float().cuda()
    rix = torch.randperm(child_pcd.shape[0])
    child_pcd_ds = child_pcd_t[rix[:N_crop]]

    # make copies of child point cloud and move to the location of parent crops (voxel positions)
    child_pcd_batch = child_pcd_ds.reshape(1, -1, 3).repeat((M, 1, 1))  # M x N_child x 3
    child_mean_batch = child_pcd_t.mean(0).reshape(1, 1, 3).repeat((1, child_pcd_ds.shape[0], 1))
    # delta_trans_batch = world_pts_t.reshape(-1, 1, 3).repeat((1, child_pcd_ds.shape[0], 1))[:M]  # M x N_child x 3
    delta_trans_batch = world_pts_t.reshape(-1, 1, 3).repeat((1, child_pcd_ds.shape[0], 1))[:M] - child_mean_batch.repeat((M, 1, 1)) # M x N_child x 3
    child_pcd_cent_batch = child_pcd_batch - child_mean_batch

    # sample a batch of rotations and apply to batch
    if run_affordance:
        rand_rot_idx = np.random.randint(rot_grid.shape[0], size=M)
        rand_rot_init = matrix_to_axis_angle(torch.from_numpy(rot_grid[rand_rot_idx])).float()
        rand_mat_init = angle_axis_to_rotation_matrix(rand_rot_init)
        rand_mat_init = rand_mat_init.squeeze().float().cuda()
        # print('Using identity random matrix init!')
        # rand_mat_init = torch.eye(4).reshape(-1, 4, 4).repeat((M, 1, 1)).float().cuda()
    else:
        rand_mat_init = torch.eye(4).reshape(-1, 4, 4).float().cuda()
    
    # apply voxel location offsets (after un-centering with mean)
    child_pcd_rot_batch = transform_pcd_torch(child_pcd_cent_batch, rand_mat_init)
    child_pcd_trans_batch = child_pcd_rot_batch + delta_trans_batch + child_mean_batch # M x N_child x 3

    # for ii in range(M):
    #     child_pcd_trans_viz = child_pcd_trans_batch[ii].detach().cpu().numpy().squeeze()
    #     parent_crop_pcd_trans_viz = parent_cropped_pcds[ii].detach().cpu().numpy().squeeze()
    #     util.meshcat_pcd_show(mc_vis, child_pcd_trans_viz, (255, 255, 0), f'scene/infer/child_pcd_trans_batch_{ii}')
    #     util.meshcat_pcd_show(mc_vis, parent_crop_pcd_trans_viz, (255, 0, 255), f'scene/infer/parent_pcd_trans_batch_{ii}')

    parent_crop_mean_batch = torch.mean(parent_cropped_pcds, axis=1).reshape(-1, 1, 3).repeat((1, N_crop, 1))
    prpdiff_pcd = parent_cropped_pcds - parent_crop_mean_batch
    crpdiff_pcd = child_pcd_rot_batch

    # obtain descriptor values for the input (parent)
    parent_rpdiff_mi = {}
    parent_rpdiff_mi['point_cloud'] = prpdiff_pcd
    # parent_rpdiff_mi['point_cloud'] = torch.from_numpy(prpdiff_pcd).float().cuda().reshape(1, -1, 3)

    # obtain descriptor values for the input (child)
    child_rpdiff_mi = {}
    child_rpdiff_mi['point_cloud'] = crpdiff_pcd
    # child_rpdiff_mi['point_cloud'] = torch.from_numpy(crpdiff_pcd).float().cuda().reshape(1, -1, 3)

    parent_local_latent = feat_model_p.extract_local_latent(parent_rpdiff_mi, new=True).detach()
    parent_global_latent = feat_model_p.extract_global_latent(parent_rpdiff_mi).detach()

    child_local_latent = feat_model_c.extract_local_latent(child_rpdiff_mi, new=True).detach()
    child_global_latent = feat_model_c.extract_global_latent(child_rpdiff_mi).detach()

    # prepare inputs to the policy
    policy_mi = {}
    policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud'] #.contiguous()
    policy_mi['child_start_pcd'] = child_rpdiff_mi['point_cloud'] #.contiguous()
    # policy_mi['parent_start_pcd_mean'] = parent_crop_mean_batch
    # policy_mi['child_start_pcd_mean'] = child_mean_batch
    policy_mi['parent_start_pcd_mean'] = torch.mean(parent_cropped_pcds, axis=1).reshape(-1, 3)
    policy_mi['child_start_pcd_mean'] = torch.mean(child_pcd_trans_batch, axis=1).reshape(-1, 3)


    # latents from point cloud encoder
    policy_mi['parent_point_latent'] = parent_local_latent
    policy_mi['child_point_latent'] = child_local_latent
    policy_mi['parent_global_latent'] = parent_global_latent
    policy_mi['child_global_latent'] = child_global_latent

    # visualize policy model input, directly before passing into rotation 

    # print('here before rot model')
    # from IPython import embed; embed()
    rot_debug_idx = 0
    offset = np.array([1.0, 0, 0])
    ppcd_viz = policy_mi['parent_start_pcd'][rot_debug_idx].detach().cpu().numpy()
    cpcd_viz = policy_mi['child_start_pcd'][rot_debug_idx].detach().cpu().numpy() 
    ppcd_mean_viz = policy_mi['parent_start_pcd_mean'][rot_debug_idx].detach().cpu().numpy()
    cpcd_mean_viz = policy_mi['child_start_pcd_mean'][rot_debug_idx].detach().cpu().numpy()
    util.meshcat_pcd_show(mc_vis, ppcd_viz + offset, (255, 0, 0), f'scene/policy_mi_pre_rot/parent_start_pcd')
    util.meshcat_pcd_show(mc_vis, cpcd_viz + offset, (0, 0, 255), f'scene/policy_mi_pre_rot/child_start_pcd')
    util.meshcat_pcd_show(mc_vis, ppcd_viz + ppcd_mean_viz + offset, (255, 0, 0), f'scene/policy_mi_pre_rot/parent_start_pcd_uncent')
    util.meshcat_pcd_show(mc_vis, cpcd_viz + cpcd_mean_viz + offset, (0, 0, 255), f'scene/policy_mi_pre_rot/child_start_pcd_uncent')

    # get rotation output prediction
    with torch.no_grad():
        rot_model_output_raw = rot_policy_model(policy_mi)  # M x N_queries x 3 x 3

    util.meshcat_pcd_show(mc_vis, child_pcd_rot_batch[0].detach().cpu().numpy(), (255, 0, 255), f'scene/viz/child_pcd_pre_rot')
    
    # reshape rotations and batch of point clouds to apply rotations
    rot_mat_queries = rot_model_output_raw['rot_mat']
    N_queries = rot_mat_queries.shape[1]
    N_queries_total = N_queries
    # N_keep_queries = N_queries // 2
    # N_queries = N_keep_queries
    # rot_mat_queries = rot_mat_queries[:, :N_keep_queries]
    # child_pcd_rot_qb = child_pcd_rot_batch[:, None, :, :].repeat((1, N_queries, 1, 1))
    child_pcd_rot_qb = child_pcd_rot_batch[:, None, :, :].repeat((1, N_queries, 1, 1))
    child_pcd_post_rot_qb = torch.matmul(rot_mat_queries, child_pcd_rot_qb.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # M x N_queries x N_child x 3

    util.meshcat_pcd_show(mc_vis, child_pcd_post_rot_qb[0, 0].detach().cpu().numpy(), (0, 255, 255), f'scene/viz/child_pcd_post_rot')

    # reshape into batch, pass into success classifier?

#     print('here with rot_model_output_raw (success classifier)')
#     from IPython import embed; embed()

    # have batch of rotated child point clouds, get translation output prediction (maybe in a loop for batch size)
    post_rot_batch = N_queries * M
    child_pcd_post_rot_qb = child_pcd_post_rot_qb.reshape(post_rot_batch, -1, 3)
    child_rpdiff_mi['point_cloud'] = child_pcd_post_rot_qb
    child_local_latent = feat_model_c.extract_local_latent(child_rpdiff_mi, new=True).detach()
    child_global_latent = feat_model_c.extract_global_latent(child_rpdiff_mi).detach()

    util.meshcat_pcd_show(mc_vis, child_pcd_post_rot_qb[0].detach().cpu().numpy(), (0, 255, 0), f'scene/viz/child_pcd_post_rot2')

    voxel_idx = torch.arange(M).reshape(-1, 1).repeat((1, N_queries)).reshape(post_rot_batch, -1)  # M*N_queries x 1
    # rot_idx = torch.arange(N_queries).reshape(1, -1).repeat((M, 1)).reshape(post_rot_batch, -1)  # M*N_queries x 1
    rot_idx = torch.arange(post_rot_batch).reshape(-1, 1)

    # for memory issues, only keep a few of the queries

    # might have to re-encode things, or just expand the parent point cloud to match the child point cloud
    
    policy_mi['child_point_latent'] = child_local_latent
    policy_mi['child_global_latent'] = child_global_latent
    policy_mi['child_start_pcd'] = child_pcd_post_rot_qb

    # policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud'].repeat((N_queries, 1, 1))
    # policy_mi['parent_start_pcd_mean'] = policy_mi['parent_start_pcd_mean'].repeat((N_queries, 1))
    # policy_mi['child_start_pcd_mean'] = policy_mi['child_start_pcd_mean'].repeat((N_queries, 1))
    policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud'][:, None, :, :].repeat((1, N_queries, 1, 1)).reshape(post_rot_batch, -1, 3) #.contiguous()  # M x N_child x 3 -> M x N_queries x N_child x 3 -> M*N_queries x N_child x 3 (in right order)
    policy_mi['parent_start_pcd_mean'] = policy_mi['parent_start_pcd_mean'][:, None, :].repeat((1, N_queries, 1)).reshape(post_rot_batch, -1)  # M x 3 -> M x N_queries x 3 -> M*N_queries x 3 (in right order)
    policy_mi['child_start_pcd_mean'] = policy_mi['child_start_pcd_mean'][:, None, :].repeat((1, N_queries, 1)).reshape(post_rot_batch, -1)
    
    base_debug_idx = rot_debug_idx*N_queries
    for j in range(4):
        debug_idx = base_debug_idx + j
        offset = np.array([1.0, 0, 0])
        ppcd_viz = policy_mi['parent_start_pcd'][debug_idx].detach().cpu().numpy()
        cpcd_viz = policy_mi['child_start_pcd'][debug_idx].detach().cpu().numpy() 
        ppcd_mean_viz = policy_mi['parent_start_pcd_mean'][debug_idx].detach().cpu().numpy()
        cpcd_mean_viz = policy_mi['child_start_pcd_mean'][debug_idx].detach().cpu().numpy()
        util.meshcat_pcd_show(mc_vis, ppcd_viz + offset, (255, 0, 0), f'scene/policy_mi_pre_trans/parent_start_pcd/{j}')
        util.meshcat_pcd_show(mc_vis, cpcd_viz + offset, (0, 0, 255), f'scene/policy_mi_pre_trans/child_start_pcd/{j}')
        util.meshcat_pcd_show(mc_vis, ppcd_viz + ppcd_mean_viz + offset, (255, 0, 0), f'scene/policy_mi_pre_trans/parent_start_pcd_uncent/{j}')
        util.meshcat_pcd_show(mc_vis, cpcd_viz + cpcd_mean_viz + offset, (0, 0, 255), f'scene/policy_mi_pre_trans/child_start_pcd_uncent/{j}')

    # get output for translation, and combine outputs with rotation prediction
    with torch.no_grad():
        trans_model_output_raw = policy_model(policy_mi)  # M x N_queries x 3

    trans_queries = trans_model_output_raw['trans'].reshape(post_rot_batch, N_queries_total, 1, 3).repeat((1, 1, N_crop, 1))
    trans_queries = trans_queries[:, :N_queries]
    # trans_queries = trans_model_output_raw['trans'].reshape(post_rot_batch, N_queries, 1, 3).repeat((1, 1, N_crop, 1))
    trans_mean = policy_mi['child_start_pcd_mean'][:, None, None, :].repeat((1, N_queries, N_crop, 1))
    child_pcd_trans_qb = child_pcd_post_rot_qb[:, None, :, :].repeat((1, N_queries, 1, 1))
    child_pcd_post_trans_qb = child_pcd_trans_qb + trans_mean + trans_queries

    voxel_idx_final = voxel_idx.repeat((1, N_queries)).reshape(post_rot_batch*N_queries, -1)  # M*N_queries**2 x 1
    # rot_idx_final = rot_idx.repeat((1, N_queries)).reshape(post_rot_batch*N_queries, -1)  # M*N_queries**2 x 1
    # trans_idx_final = torch.arange(N_queries).reshape(1, -1).repeat((post_rot_batch, 1)).reshape(post_rot_batch*N_queries, -1)  # M*N_queries**2 x 1
    rot_idx_final = torch.arange(post_rot_batch*N_queries).reshape(-1, 1)  # M*N_queries**2 x 1
    trans_idx_final = torch.arange(post_rot_batch*N_queries).reshape(-1, 1)  # M*N_queries**2 x 1
    rot_mat_queries_final = rot_mat_queries.reshape(post_rot_batch, 3, 3)[:, None, :, :].repeat((1, N_queries, 1, 1)).reshape(post_rot_batch*N_queries, 3, 3)
    # rot_mat_queries_final = rot_mat_queries.reshape(post_rot_batch, 3, 3)
    trans_queries_final = trans_queries[:, :, 0].reshape(post_rot_batch*N_queries, -1)

    # reshape translations and batch of point clouds to apply translations
    child_pcd_to_sc = child_pcd_post_trans_qb.reshape(post_rot_batch*N_queries, -1, 3) #.contiguous()
    child_rpdiff_mi['point_cloud'] = child_pcd_to_sc
    child_local_latent = feat_model_c.extract_local_latent(child_rpdiff_mi, new=True).detach()
    child_global_latent = feat_model_c.extract_global_latent(child_rpdiff_mi).detach()

    child_pcd_to_sc_mean = torch.mean(child_pcd_to_sc, axis=1)
    policy_mi['child_point_latent'] = child_local_latent
    policy_mi['child_global_latent'] = child_global_latent
    # policy_mi['child_start_pcd'] = child_pcd_to_sc
    policy_mi['child_start_pcd'] = child_pcd_to_sc - child_pcd_to_sc_mean[:, None, :].repeat((1, N_crop, 1))

    # print('here in multistep, before setting point clouds and refinement')
    # from pdb import set_trace; set_trace()
    # policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud'][:, None, :, :].repeat((1, N_queries, 1, 1)).reshape(post_rot_batch*N_queries, -1, 3)  # M x N_child x 3 -> M x N_queries x N_child x 3 -> M*N_queries x N_child x 3 (in right order)
    policy_mi['parent_start_pcd'] = policy_mi['parent_start_pcd'][:, None, :, :].repeat((1, N_queries, 1, 1)).reshape(post_rot_batch*N_queries, -1, 3)  # M x N_child x 3 -> M x N_queries x N_child x 3 -> M*N_queries x N_child x 3 (in right order)
    policy_mi['parent_start_pcd_mean'] = policy_mi['parent_start_pcd_mean'][:, None, :].repeat((1, N_queries, 1)).reshape(post_rot_batch*N_queries, -1)  # M x 3 -> M x N_queries x 3 -> M*N_queries x 3 (in right order)
    policy_mi['child_start_pcd_mean'] = child_pcd_to_sc_mean


    # print('here before refine')
    # from IPython import embed; embed()

    # base_debug_idx = rot_debug_idx*(N_queries**2) 
    # for j in range(4):
    #     for jj in range(4):
    #         jj_val = N_queries*j + jj
    #         debug_idx = base_debug_idx + jj_val
    #         offset = np.array([1.0, 0, 0])
    #         ppcd_viz = policy_mi['parent_start_pcd'][debug_idx].detach().cpu().numpy()
    #         cpcd_viz = policy_mi['child_start_pcd'][debug_idx].detach().cpu().numpy() 
    #         ppcd_mean_viz = policy_mi['parent_start_pcd_mean'][debug_idx].detach().cpu().numpy()
    #         cpcd_mean_viz = policy_mi['child_start_pcd_mean'][debug_idx].detach().cpu().numpy()
    #         util.meshcat_pcd_show(mc_vis, ppcd_viz + offset, (255, 0, 0), f'scene/policy_mi_post_trans/parent_start_pcd/{jj_val}')
    #         util.meshcat_pcd_show(mc_vis, cpcd_viz + offset, (0, 0, 255), f'scene/policy_mi_post_trans/child_start_pcd/{jj_val}')
    #         util.meshcat_pcd_show(mc_vis, ppcd_viz + ppcd_mean_viz + offset, (255, 0, 0), f'scene/policy_mi_post_trans/parent_start_pcd_uncent/{jj_val}')
    #         util.meshcat_pcd_show(mc_vis, cpcd_viz + cpcd_mean_viz + offset, (0, 0, 255), f'scene/policy_mi_post_trans/child_start_pcd_uncent/{jj_val}')

    # print('here ready for refinement')
    # # from pdb import set_trace; set_trace()
    # from IPython import embed; embed()

    # break into smaller batches
    sm_bs = 8
    success_out_list = []

    offset = np.array([1.0, 0, 0])
    ppcd_viz_full = full_parent_pcd_ds[0].detach().cpu().numpy()
    ppcd_mean_viz_full = full_parent_pcd_mean[0].detach().cpu().numpy()
    # util.meshcat_pcd_show(mc_vis, ppcd_viz_full + ppcd_mean_viz_full + offset, (0, 255, 255), f'scene/policy_mi_post_trans/parent_start_pcd_uncent_full', size=0.007)
    util.meshcat_pcd_show(mc_vis, ppcd_viz_full + offset, (0, 255, 255), f'scene/policy_mi_post_trans/parent_start_pcd_uncent_full', size=0.007)

    # # Slow way to do this
    # n_to_crop = policy_mi['child_start_pcd_mean'].shape[0]
    # parent_cropped_pcds_refine = torch.empty((n_to_crop, N_crop, 3)).float().cuda()
    # N_parent = parent_pcd.shape[0]
    # for ii in range(n_to_crop):
    #     voxel_3d_pt = policy_mi['child_start_pcd_mean'][ii].reshape(1, 3)  # 1 x 3
    #     voxel_high = voxel_3d_pt + max_length
    #     voxel_low = voxel_3d_pt - max_length
    #     high, low = voxel_high.repeat((N_parent, 1)), voxel_low.repeat((N_parent, 1))
    #     # high, low = voxel_3d_pt.repeat((1, N_parent, 1)) sample_low_t.repeat((1, N_parent, 1))
    #     parent_crop_idx = torch.where(torch.logical_and(pscene_pcd_t < high, pscene_pcd_t > low).all(-1))[0]
    #     parent_crop_pcd = pscene_pcd_t[parent_crop_idx]

    #     rix = torch.randperm(parent_crop_pcd.shape[0])
    #     
    #     if parent_crop_pcd.shape[0] == 0:
    #         print('!! Parent crop had zero points !!')
    #         # from IPython import embed; embed()
    #         rix2 = torch.randperm(pscene_pcd_t.shape[0])
    #         parent_cropped_pcds[i] = pscene_pcd_t[rix2[:N_crop]]
    #         continue
    #         
    #     if parent_crop_pcd.shape[0] < N_crop:
    #         while True:
    #             print(f'Parent pcd crop shape: {parent_crop_pcd.shape[0]}')
    #             parent_crop_pcd = torch.cat((parent_crop_pcd, parent_crop_pcd[rix[:100]]), dim=0)
    #             if parent_crop_pcd.shape[0] >= N_crop:
    #                 break
    #             rix = torch.randperm(parent_crop_pcd.shape[0])
    #     else:
    #         parent_crop_pcd = parent_crop_pcd[rix[:N_crop]]

    #     parent_cropped_pcds_refine[ii] = parent_crop_pcd

    # # Show some random ones
    # idxs = np.random.randint(0, n_to_crop, (5,))
    # # idxs = torch.randint(n_to_crop, (5,))
    # for idx in idxs:
    #     util.meshcat_pcd_show(mc_vis, offset + parent_cropped_pcds_refine[idx].detach().cpu().numpy(), (0, 255, 0), f'scene/new_cropped_parent/pcd_{idx}') 

    pscene_pcd_t_to_crop = pscene_pcd_t[None, :, :].repeat((8, 1, 1))
    refine_policy_mi = {k: v.clone() for k, v in policy_mi.items()}
    with torch.no_grad():
        for ii in range(n_iters):

            base_debug_idx = rot_debug_idx*(N_queries**2) 
            for j in range(4):
                for jj in range(4):
                    jj_val = N_queries*j + jj
                    debug_idx = base_debug_idx + jj_val
                    ppcd_viz = refine_policy_mi['parent_start_pcd'][debug_idx].detach().cpu().numpy()
                    cpcd_viz = refine_policy_mi['child_start_pcd'][debug_idx].detach().cpu().numpy() 
                    ppcd_mean_viz = refine_policy_mi['parent_start_pcd_mean'][debug_idx].detach().cpu().numpy()
                    cpcd_mean_viz = refine_policy_mi['child_start_pcd_mean'][debug_idx].detach().cpu().numpy()
                    util.meshcat_pcd_show(mc_vis, ppcd_viz + offset, (255, 0, 0), f'scene/policy_mi_post_trans/parent_start_pcd/{jj_val}')
                    util.meshcat_pcd_show(mc_vis, cpcd_viz + offset, (0, 0, 255), f'scene/policy_mi_post_trans/child_start_pcd/{jj_val}')
                    util.meshcat_pcd_show(mc_vis, ppcd_viz + ppcd_mean_viz + offset, (255, 0, 0), f'scene/policy_mi_post_trans/parent_start_pcd_uncent/{jj_val}')
                    util.meshcat_pcd_show(mc_vis, cpcd_viz + cpcd_mean_viz + offset, (0, 0, 255), f'scene/policy_mi_post_trans/child_start_pcd_uncent/{jj_val}')

            idx = 0
            print(f'Round {ii} out of {n_iters} rounds of refinement')
            while True:
                # print(f'Refine index: {idx}')
                if idx >= policy_mi['child_start_pcd'].shape[0]:
                    # print('Done with small batch success model predictions')
                    break

                small_policy_mi = {k: v[idx:idx+sm_bs] for k, v in refine_policy_mi.items()}
                small_child_rpdiff_mi = {k: v[idx:idx+sm_bs] for k, v in child_rpdiff_mi.items()}
                small_parent_rpdiff_mi = {k: v[idx:idx+sm_bs] for k, v in parent_rpdiff_mi.items()}
                small_child_pcd_mean = refine_policy_mi['child_start_pcd_mean'][idx:idx+sm_bs]

                # re-crop for this mini-batch
                # print('here before new crop')
                # from IPython import embed; embed()
                if no_parent_crop:
                    pass
                else:
                    high_b = small_policy_mi['child_start_pcd_mean'] + max_length
                    low_b = small_policy_mi['child_start_pcd_mean'] - max_length
                    high_b_rep = high_b[:, None, :].repeat((1, N_parent, 1))
                    low_b_rep = low_b[:, None, :].repeat((1, N_parent, 1))

                    below = (pscene_pcd_t_to_crop < high_b_rep)
                    above = (pscene_pcd_t_to_crop > low_b_rep)
                    # crop_idx = torch.where(torch.logical_and(above, below).all(-1))[0]
                    crop_idx = torch.logical_and(above, below).all(-1)  # sm_bs x N_parent
                    parent_crop_sb = torch.empty((sm_bs, N_crop, 3)).float().cuda()
                    for jj in range(sm_bs):
                        parent_crop_pcd = pscene_pcd_t[torch.where(crop_idx[jj])[0]]

                        if parent_crop_pcd.shape[0] == 0:
                            print('!! Parent crop had zero points !!')
                            # from IPython import embed; embed()
                            rix2 = torch.randperm(pscene_pcd_t.shape[0])
                            parent_crop_sb[jj] = pscene_pcd_t[rix2[:N_crop]]
                            continue
                            
                        rix = torch.randperm(parent_crop_pcd.shape[0])
                        if parent_crop_pcd.shape[0] < N_crop:
                            while True:
                                print(f'Parent pcd crop shape: {parent_crop_pcd.shape[0]}')
                                parent_crop_pcd = torch.cat((parent_crop_pcd, parent_crop_pcd[rix[:100]]), dim=0)
                                if parent_crop_pcd.shape[0] >= N_crop:
                                    break
                                rix = torch.randperm(parent_crop_pcd.shape[0])
                        else:
                            parent_crop_pcd = parent_crop_pcd[rix[:N_crop]]

                        # parent_crop_mean = torch.mean(parent_crop_pcd, axis=0)
                        # small_policy_mi['parent_start_pcd'][jj] = parent_crop_pcd - parent_crop_mean[:, None].repeat((1, N_crop))
                        # small_policy_mi['parent_start_pcd_mean'][jj] = parent_crop_mean
                        # parent_crop_sb[jj] = parent_crop_pcd - parent_crop_mean[:, None].repeat((1, N_crop))

                        parent_crop_sb[jj] = parent_crop_pcd[:N_crop]
                        # util.meshcat_pcd_show(mc_vis, parent_crop_sb.detach().cpu().numpy().reshape(-1, 3), (0, 255, 0), f'scene/new_cropped_parent_raw/pcd_{jj}') 
                        
                    parent_crop_sb_mean = torch.mean(parent_crop_sb, axis=1)
                    small_parent_rpdiff_mi['point_cloud'] = parent_crop_sb - parent_crop_sb_mean[:, None, :].repeat((1, N_crop, 1))
                    small_policy_mi['parent_start_pcd'] = parent_crop_sb - parent_crop_sb_mean[:, None, :].repeat((1, N_crop, 1))
                    small_policy_mi['parent_start_pcd_mean'] = parent_crop_sb_mean

                # for idx in range(sm_bs):
                #     util.meshcat_pcd_show(mc_vis, offset + small_policy_mi['parent_start_pcd_mean'][idx].detach().cpu().numpy() + small_policy_mi['parent_start_pcd'][idx].detach().cpu().numpy(), (0, 255, 0), f'scene/new_cropped_parent/pcd_{idx}') 
                    # util.meshcat_pcd_show(mc_vis, offset + small_policy_mi['child_start_pcd_mean'][idx].detach().cpu().numpy() + small_policy_mi['child_start_pcd'][idx].detach().cpu().numpy(), (0, 255, 255), f'scene/new_cropped_parent/child_pcd_{idx}') 

                    # util.meshcat_pcd_show(mc_vis, offset + high_b[idx].detach().cpu().numpy().reshape(-1, 3), (255, 255, 128), f'scene/new_cropped_parent_high_sph/{idx}', size=0.15) 
                    # util.meshcat_pcd_show(mc_vis, offset + low_b[idx].detach().cpu().numpy().reshape(-1, 3), (128, 255, 255), f'scene/new_cropped_parent_low_sph/{idx}', size=0.15) 

                # if idx == 0:
                #     pcd_mean = small_policy_mi['child_start_pcd_mean'].detach().cpu().numpy().squeeze()
                #     pcd_cent = small_policy_mi['child_start_pcd'].detach().cpu().numpy().squeeze()
                #     for jj in range(sm_bs): 
                #         pcd_show_cent = pcd_cent[jj]
                #         pcd_show = pcd_cent[jj] + pcd_mean[jj]
                #         util.meshcat_pcd_show(mc_vis, pcd_show, (0, 0, 255), f'scene/refine/sm_bs_{jj}')
                #         util.meshcat_pcd_show(mc_vis, pcd_show_cent, (0, 0, 255), f'scene/refine/sm_bs_cent_{jj}')
                #     
                #     print('showing refine child pcds')
                #     from IPython import embed; embed()

                child_local_latent = feat_model_c.extract_local_latent(small_child_rpdiff_mi, new=True).detach()
                child_global_latent = feat_model_c.extract_global_latent(small_child_rpdiff_mi).detach()

                # new policy input based on these features of rotated shape
                small_policy_mi['child_point_latent'] = child_local_latent
                small_policy_mi['child_global_latent'] = child_global_latent

                # get rotation output prediction
                rot_model_output_raw = rot_policy_model_refine(small_policy_mi)
                # if ii < 3:
                #     rot_model_output_raw = rot_policy_model(small_policy_mi)
                # else:
                #     rot_model_output_raw = rot_policy_model_refine(small_policy_mi)
                nq = rot_model_output_raw['rot_mat'].shape[1]
                rot_idx = 0
                rot_model_output = {}
                rot_model_output['rot_mat'] = rot_model_output_raw['rot_mat'][:, rot_idx]

                # quat_label = matrix_to_quaternion(torch.eye(3).reshape(1, 3, 3))[:, None, :].repeat(1, nq, 1).float().cuda()
                # quat_pred = matrix_to_quaternion(rot_model_output_raw['rot_mat'])
                # quat_scalar_prod = torch.sum(quat_pred * quat_label, axis=-1)
                # quat_dist = 1 - torch.pow(quat_scalar_prod, 2)
                # rot_idx = torch.argmin(quat_dist, 1)
                # rot_model_output = {}
                # rot_model_output['rot_mat'] = torch.gather(rot_model_output_raw['rot_mat'], dim=1, index=rot_idx[:, None, None, None].repeat(1, 1, 3, 3)).reshape(sm_bs, 3, 3)

                # make a rotated version of the original point cloud
                child_pcd_final_pred = torch.bmm(rot_model_output['rot_mat'], small_policy_mi['child_start_pcd'].transpose(1, 2)).transpose(2, 1)
                child_pcd_rot = child_pcd_final_pred.detach()
                
                # reencode this rotated point cloud to get new descriptor features
                small_child_rpdiff_mi['point_cloud'] = child_pcd_rot
                child_local_latent = feat_model_c.extract_local_latent(small_child_rpdiff_mi, new=True).detach()
                child_global_latent = feat_model_c.extract_global_latent(small_child_rpdiff_mi).detach()
                
                # new policy input based on these features of rotated shape
                small_policy_mi['child_point_latent'] = child_local_latent
                small_policy_mi['child_global_latent'] = child_global_latent
                small_policy_mi['child_start_pcd'] = child_pcd_rot
                
                # get output for translation, and combine outputs with rotation prediction
                # from IPython import embed; embed()
                trans_model_output_raw = policy_model_refine(small_policy_mi)
                # if ii < 3:
                #     trans_model_output_raw = policy_model(small_policy_mi)
                # else:
                #     trans_model_output_raw = policy_model_refine(small_policy_mi)
                trans_norm = torch.norm(trans_model_output_raw['trans'], dim=-1)
                trans_idx = torch.argmin(trans_norm, 1)
                # trans_idx = 0

                model_output = {}
                # model_output['trans'] = trans_model_output_raw['trans'][:, trans_idx]
                model_output['trans'] = torch.gather(trans_model_output_raw['trans'], dim=1, index=trans_idx[:, None, None].repeat(1, 1, 3)).reshape(-1, 3)
                # try:
                #     model_output['trans'] = torch.gather(trans_model_output_raw['trans'], dim=1, index=trans_idx[:, None, None].repeat(1, 1, 3)).reshape(-1, 3)
                # except Exception as e:
                #     print(f'Exception: {e}')
                #     # print('here before gathering trans_model_output_raw')
                #     from pdb import set_trace; set_trace()

                trans_queries = model_output['trans']
                trans_mean = small_child_pcd_mean

                refine_policy_mi['child_point_latent'][idx:idx+sm_bs] = child_local_latent
                refine_policy_mi['child_global_latent'][idx:idx+sm_bs] = child_global_latent
                refine_policy_mi['child_start_pcd'][idx:idx+sm_bs] = child_pcd_rot
                refine_policy_mi['child_start_pcd_mean'][idx:idx+sm_bs] = trans_mean + trans_queries

                # adjust the final translation and rotation based on this
                trans_queries_final[idx:idx+sm_bs] = trans_queries_final[idx:idx+sm_bs] + trans_queries
                rot_mat_queries_final[idx:idx+sm_bs] = torch.bmm(rot_model_output['rot_mat'], rot_mat_queries_final[idx:idx+sm_bs])

                idx += sm_bs 

    # print('here done with refinement')
    # from pdb import set_trace; set_trace()
    
    # put in full parent point cloud for final success classification
    # pscene_mean = torch.mean(pscene_pcd_t, axis=0).reshape(1, 3)
    # pscene_pcd_cent_t = pscene_pcd_t - pscene_mean.repeat((N_crop, 1))
    pscene_mean = torch.mean(pscene_pcd_t, axis=0).reshape(1, 3)
    fps_ds = FPSDownSample(N_crop)
    pscene_pcd_cent_t = fps_ds.forward(pscene_pcd_t.reshape(1, -1, 3)) - pscene_mean.repeat((N_crop, 1))
    # rix_p = torch.randperm(pscene_pcd_t.shape[0])
    # pscene_pcd_cent_t = pscene_pcd_t[rix[:N_crop]] - pscene_mean.repeat((N_crop, 1))
    policy_mi['parent_start_pcd'] = pscene_pcd_cent_t.reshape(1, -1, 1).repeat((post_rot_batch*N_queries, 1, 1))
    policy_mi['parent_start_pcd_mean'] = pscene_mean.repeat((post_rot_batch*N_queries, 1))

    # break into smaller batches
    sm_bs = 8
    idx = 0
    success_out_list = []

    with torch.no_grad():
        while True:
            if idx >= policy_mi['child_start_pcd'].shape[0]:
                # print('Done with small batch success model predictions')
                break

            small_policy_mi = {k: v[idx:idx+sm_bs] for k, v in refine_policy_mi.items()}
            small_success_out = success_model(small_policy_mi)
            success_out_list.extend(small_success_out['success'].detach().cpu().numpy().squeeze().tolist())

            idx += sm_bs 
    
    success_out_all = torch.Tensor(success_out_list).float().cuda()

    # print('here in multistep regression + voxel + success')
    # from IPython import embed; embed()

    # pass into success classifier (have to loop to reduce batch size)
    # success_model_output = success_model(policy_mi)

    # success_argmax = np.argmax(success_out_list)
    # voxel_argmax_idx = voxel_idx_final[success_argmax].squeeze()
    # rot_argmax_idx = rot_idx_final[success_argmax].squeeze()
    # trans_argmax_idx = trans_idx_final[success_argmax].squeeze()

    # show_iter = 0
    # stop_show = False
    # while True:
    #     show_iter += 1
    #     if show_iter >= 10 or stop_show:
    #         break

    #     success_kval = 5
    #     success_topk_vals, success_topk_idx = torch.topk(success_out_all, k=success_kval)
    #     # success_topk_idx = torch.randint(success_out_all.shape[0], size=(success_kval,))
    #     child_pcd_cent_topk = refine_policy_mi['child_start_pcd'][success_topk_idx]
    #     mean_topk = refine_policy_mi['child_start_pcd_mean'][success_topk_idx]
    #     child_pcd_final_topk = child_pcd_cent_topk + mean_topk[:, None, :].repeat((1, N_crop, 1))

    #     # colormap for the top-k
    #     # child_pcd_cent = child_pcd - np.mean(child_pcd, axis=0)
    #     cmap = plt.get_cmap('inferno')
    #     color_list = cmap(np.linspace(0.1, 0.9, success_kval, dtype=np.float32))[::-1]
    #     util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/infer/parent_pcd')
    #     tmat_full_list = []
    #     for ii in range(success_kval):
    #         color = (color_list[ii][:-1] * 255).astype(np.uint8).tolist()
    #         # color = color_list[ii] * 255
    #         # print(f'Color: {color}')
    #         box = trimesh.PointCloud(child_pcd_final_topk[ii].detach().cpu().numpy().squeeze()).bounding_box_oriented.to_mesh()
    #         util.meshcat_pcd_show(mc_vis, child_pcd_final_topk[ii].detach().cpu().numpy().squeeze(), color, f'scene/infer/success_topk/child_pts_{ii}')
    #         util.meshcat_trimesh_show(mc_vis, f'scene/infer/success_topk_box/child_pts_box_{ii}', box, color)

    #     print('Here after full refine (showing)')
    #     from IPython import embed; embed()

    #     mc_vis['scene/infer/success_topk'].delete()
    #     mc_vis['scene/infer/success_topk_box'].delete()

    success_kval = 25
    success_topk_vals, success_topk_idx = torch.topk(success_out_all, k=success_kval)
    child_pcd_cent_topk = refine_policy_mi['child_start_pcd'][success_topk_idx]
    mean_topk = refine_policy_mi['child_start_pcd_mean'][success_topk_idx]
    child_pcd_final_topk = child_pcd_cent_topk + mean_topk[:, None, :].repeat((1, N_crop, 1))

    voxel_topk = voxel_idx_final[success_topk_idx].squeeze()
    rot_topk = rot_idx_final[success_topk_idx].squeeze()
    trans_topk = trans_idx_final[success_topk_idx].squeeze()

    out_voxel_trans = world_pts_t[voxel_topk].reshape(-1, 3)
    out_rot_init = rand_mat_init.reshape(-1, 4, 4)[voxel_topk].reshape(-1, 4, 4)
    # out_rot_init = rand_mat_init[voxel_topk].reshape(-1, 4, 4)
    out_rot = rot_mat_queries_final[rot_topk].reshape(-1, 3, 3)
    out_trans = trans_queries_final[trans_topk].reshape(-1, 3)
    out_trans_full = out_voxel_trans + out_trans

    # colormap for the top-k
    child_pcd_cent = child_pcd - np.mean(child_pcd, axis=0)
    cmap = plt.get_cmap('inferno')
    color_list = cmap(np.linspace(0.1, 0.9, success_kval, dtype=np.float32))[::-1]
    util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/infer/parent_pcd')
    tmat_full_list = []
    for ii in range(success_kval):
        color = (color_list[ii][:-1] * 255).astype(np.uint8).tolist()
        box = trimesh.PointCloud(child_pcd_final_topk[ii].detach().cpu().numpy().squeeze()).bounding_box_oriented.to_mesh()
        # util.meshcat_pcd_show(mc_vis, child_pcd_final_topk[ii].detach().cpu().numpy().squeeze(), color, f'scene/infer/success_topk/child_pts_{ii}')
        util.meshcat_trimesh_show(mc_vis, f'scene/infer/success_topk_box/child_pts_box_{ii}', box, color)

        tmat = np.eye(4); tmat[:-1, -1] = out_trans_full[ii].detach().cpu().numpy().squeeze()
        tmat[:-1, :-1] = out_rot[ii].detach().cpu().numpy().squeeze()
        tmat_init = out_rot_init[ii].detach().cpu().numpy().squeeze()

        tmat_full = np.matmul(tmat, tmat_init)
        tmat_full_list.append(tmat_full)
        trans_child_pcd = util.transform_pcd(child_pcd_cent, tmat_full)

        # util.meshcat_pcd_show(mc_vis, trans_child_pcd, (255, 0, 255), f'scene/infer/success_topk/child_pts_tf_{ii}')

    # print('Here after viz refine')
    # from IPython import embed; embed()

    # if viz:
    #     # colormap for the top-k
    #     cmap = plt.get_cmap('plasma')
    #     color_list = cmap(np.linspace(0.1, 0.9, 5, dtype=np.float32))[::-1]
    #     for ii in range(5):
    #         color = (color_list[ii][:-1] * 255).astype(np.int8).tolist()
    #         util.meshcat_pcd_show(mc_vis, current_child_pcd, color, f'scene/infer/success_topk/child_pts_{ii}')

    #     util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/infer/parent_pcd')
    #     util.meshcat_pcd_show(mc_vis, child_pcd_original, (0, 0, 255), 'scene/infer/child_pcd')
    #     if run_affordance:
    #         util.meshcat_pcd_show(mc_vis, pscene_pcd, (255, 0, 255), 'scene/infer/parent_pcd_norm')
    #     util.meshcat_pcd_show(mc_vis, cropped_parent_pcd, (255, 0, 0), 'scene/infer/cropped_parent_pcd')
    #     util.meshcat_pcd_show(mc_vis, child_pcd, (0, 255, 0), 'scene/infer/child_pcd_delta')
    #     viz_rot_wf = child_pcd_rot + policy_mi['child_start_pcd_mean'].reshape(-1, 1, 3).repeat(1, child_pcd_rot.shape[1], 1)
    #     util.meshcat_pcd_show(mc_vis, viz_rot_wf[0].detach().cpu().numpy(), (0, 0, 0), 'scene/infer/post_rot_world_frame')
    #     viz_trans_wf = child_pcd_rot + policy_mi['child_start_pcd_mean'].reshape(-1, 1, 3).repeat(1, child_pcd_rot.shape[1], 1) + model_output['trans'].reshape(-1, 1, 3).repeat(1, child_pcd_rot.shape[1], 1)
    #     util.meshcat_pcd_show(mc_vis, viz_trans_wf[0].detach().cpu().numpy(), (0, 0, 255), 'scene/infer/post_rot_trans_world_frame')

    tf_cent = np.eye(4); tf_cent[:-1, -1] = -1.0 * np.mean(child_pcd_original, axis=0)
    # out_tf = tmat_full_list[0]
    out_tf = np.matmul(tmat_full_list[0], tf_cent)
    # rand_tf_idx = np.random.randint(len(tmat_full_list))
    # out_tf = np.matmul(tmat_full_list[rand_tf_idx], tf_cent)

    # print('Out TF: ', out_tf)

    # util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/real/parent_pts')
    # util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 255), 'scene/real/child_pts')

    # print('here before returning multistep regression')
    # from IPython import embed; embed()

    return out_tf

def single_shot_regression_scene_combine(mc_vis, parent_pcd, child_pcd, scene_model_dict, feat_model, policy_model, 
                           grid_pts, run_affordance=False, viz=False, *args, **kwargs):
    """
    Directly predict the relative transformation in one shot
    """
    scene_model = scene_model_dict['scene_model'].eval()
    aff_model = scene_model_dict['aff_model'].eval()

    child_pcd_original = child_pcd.copy()

    rix1 = np.random.permutation(parent_pcd.shape[0])
    rix2 = np.random.permutation(child_pcd.shape[0])
    # parent_pcd = parent_pcd[rix1[:10000]]
    # child_pcd = child_pcd[rix2[:10000]]

    model_dict = None
    if isinstance(policy_model, dict):
        model_dict = policy_model
        policy_model = model_dict['trans']
        rot_policy_model = model_dict['rot']
    else:
        rot_policy_model = policy_model

    policy_model = policy_model.eval()
    rot_policy_model = rot_policy_model.eval()

    if isinstance(feat_model, dict):
        feat_model_p = feat_model['parent']
        feat_model_c = feat_model['child']
    else:
        feat_model_p = feat_model_c = feat_model
    
    delta_child_world = np.zeros(3)
    table_mean = np.array([0.35, 0.0, np.min(parent_pcd[:, 2])])
    table_extents = np.array([0.7, 1.2, 0])
    table_scale = 1 / np.max(table_extents)
    if run_affordance:
        pscene_pcd = (copy.deepcopy(parent_pcd) - table_mean ) * table_scale

        parent_scene_mi = {}
        parent_scene_mi['point_cloud'] = torch.from_numpy(pscene_pcd[rix1[:10000]]).float().cuda().reshape(1, -1, 3)
        # parent_scene_mi['point_cloud'] = torch.from_numpy(pscene_pcd).float().cuda().reshape(1, -1, 3)

        k_val = 10
        scene_grid_latent = scene_model.extract_latent(parent_scene_mi)  # dict with keys 'grid', 'xy', 'xz', 'yz'
        # scene_model_output = aff_model(scene_grid_latent['grid'])
        fea_grid = scene_grid_latent['grid'].permute(0, 2, 3, 4, 1)
        scene_model_output = aff_model(fea_grid)
        vals, inds = torch.topk(scene_model_output['voxel_affordance'][0], k=k_val)
        inds_np = inds.detach().cpu().numpy()
        voxel_pts = grid_pts[inds_np]
        world_pts = (voxel_pts / table_scale) + table_mean

        # util.meshcat_pcd_show(mc_vis, pscene_pcd, (255, 0, 255), 'scene/infer/parent_pcd_norm')
        sz_base = 1.1/32
        for i, pt in enumerate(world_pts):
            box = trimesh.creation.box([sz_base]*3).apply_translation(pt)
            # box = trimesh.creation.box([sz_base * 2 * (len(inds) - idx) / len(inds)]*3).apply_translation(pt)
            # print(sm_vals[idx])
            util.meshcat_trimesh_show(mc_vis, f'scene/voxel_grid/{i}', box, opacity=0.3)

        world_pt_idx = 0
        world_pt = world_pts[world_pt_idx]
        delta_child_world = world_pt - np.mean(child_pcd, axis=0)

        child_pcd = child_pcd + delta_child_world
        # child_pcd_trans = child_pcd + delta_child_world

    # util.meshcat_pcd_show(mc_vis, child_pcd, (0, 255, 255), 'scene/infer/child_pcd_delta')

    # crop the parent
    child_mean = np.mean(child_pcd, axis=0)
    child_pcd_scaled = child_pcd - child_mean
    child_pcd_scaled *= 1.25
    child_pcd_scaled = child_pcd_scaled + child_mean
    child_bb = trimesh.PointCloud(child_pcd_scaled).bounding_box.to_mesh()
    max_length = np.max(child_bb.extents) / 2
    xmin, xmax = child_mean[0] - max_length, child_mean[0] + max_length
    ymin, ymax = child_mean[1] - max_length, child_mean[1] + max_length
    zmin, zmax = child_mean[2] - max_length, child_mean[2] + max_length
    cropped_parent_pcd = util.crop_pcd(
        parent_pcd, 
        x=[xmin, xmax],
        y=[ymin, ymax],
        z=[zmin, zmax])

    rix1 = np.random.permutation(cropped_parent_pcd.shape[0])
    rix2 = np.random.permutation(child_pcd.shape[0])
    cropped_parent_pcd = cropped_parent_pcd[rix1[:2048]]
    child_pcd = child_pcd[rix2[:2048]]

    prpdiff_pcd = copy.deepcopy(util.center_pcd(cropped_parent_pcd)) 
    crpdiff_pcd = copy.deepcopy(util.center_pcd(child_pcd)) 

    # obtain descriptor values for the input (parent)
    parent_rpdiff_mi = {}
    parent_rpdiff_mi['point_cloud'] = torch.from_numpy(prpdiff_pcd).float().cuda().reshape(1, -1, 3)

    # obtain descriptor values for the input (child)
    child_rpdiff_mi = {}
    child_rpdiff_mi['point_cloud'] = torch.from_numpy(crpdiff_pcd).float().cuda().reshape(1, -1, 3)

    parent_local_latent = feat_model_p.extract_local_latent(parent_rpdiff_mi, new=True).detach()
    parent_global_latent = feat_model_p.extract_global_latent(parent_rpdiff_mi).detach()

    child_local_latent = feat_model_c.extract_local_latent(child_rpdiff_mi, new=True).detach()
    child_global_latent = feat_model_c.extract_global_latent(child_rpdiff_mi).detach()

    # prepare inputs to the policy
    policy_mi = {}
    policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud']
    policy_mi['child_start_pcd'] = child_rpdiff_mi['point_cloud']
    policy_mi['parent_start_pcd_mean'] = torch.from_numpy(np.mean(cropped_parent_pcd, axis=0)).float().cuda().reshape(1, 3)
    policy_mi['child_start_pcd_mean'] = torch.from_numpy(np.mean(child_pcd, axis=0)).float().cuda().reshape(1, 3)

    # latents from point cloud encoder
    policy_mi['parent_point_latent'] = parent_local_latent
    policy_mi['child_point_latent'] = child_local_latent
    policy_mi['parent_global_latent'] = parent_global_latent
    policy_mi['child_global_latent'] = child_global_latent

    # print("here before_policy")
    # from IPython import embed; embed()

    # get rotation output prediction
    rot_model_output_raw = rot_policy_model(policy_mi)
    nq = rot_model_output_raw['rot_mat'].shape[1]
    if run_affordance:
        rot_idx = torch.argmax(rot_model_output_raw['rot_multi_query_affordance'][0])
        # rot_idx = 0
        # quat_label = matrix_to_quaternion(torch.eye(3).reshape(1, 3, 3))[:, None, :].repeat(1, nq, 1).float().cuda()
        # quat_pred = matrix_to_quaternion(rot_model_output_raw['rot_mat'])
        # quat_scalar_prod = torch.sum(quat_pred * quat_label, axis=-1)
        # quat_dist = 1 - torch.pow(quat_scalar_prod, 2)

        # rot_idx = torch.argmin(quat_dist.squeeze())
    else:
        # pick the smallest one
        # quat_label = matrix_to_quaternion(torch.eye(3).reshape(1, 3, 3))[:, None, :].repeat(1, nq, 1).float().cuda()
        # quat_pred = matrix_to_quaternion(rot_model_output_raw['rot_mat'])
        # quat_scalar_prod = torch.sum(quat_pred * quat_label, axis=-1)
        # quat_dist = 1 - torch.pow(quat_scalar_prod, 2)

        # rot_idx = torch.argmin(quat_dist.squeeze())

        rot_idx = 0

    rot_model_output = {}
    # rot_model_output['rot_mat'] = torch.gather(rot_model_output_raw['rot_mat'], dim=1, index=rot_idx[:, None, None, None].repeat(1, 1, 3, 3)).reshape(bs, 3, 3)
    rot_model_output['rot_mat'] = rot_model_output_raw['rot_mat'][:, rot_idx]

    # make a rotated version of the original point cloud
    child_start_pcd_original = policy_mi['child_start_pcd'].clone().detach()
    child_pcd_final_pred = torch.bmm(rot_model_output['rot_mat'], policy_mi['child_start_pcd'].transpose(1, 2)).transpose(2, 1)
    child_pcd_rot = child_pcd_final_pred.detach()

    # print("here after rot")
    # from IPython import embed; embed()
    
    # reencode this rotated point cloud to get new descriptor features
    child_rpdiff_mi['point_cloud'] = child_pcd_rot
    child_local_latent = feat_model_c.extract_local_latent(child_rpdiff_mi, new=True).detach()
    child_global_latent = feat_model_c.extract_global_latent(child_rpdiff_mi).detach()
    
    # new policy input based on these features of rotated shape
    policy_mi['child_point_latent'] = child_local_latent
    policy_mi['child_global_latent'] = child_global_latent
    policy_mi['child_start_pcd'] = child_pcd_rot
    
    # get output for translation, and combine outputs with rotation prediction
    trans_model_output_raw = policy_model(policy_mi)
    if run_affordance:
        #trans_idx = torch.argmax(trans_model_output_raw['trans_multi_query_affordance'][0])
        trans_norm = torch.norm(trans_model_output_raw['trans'], dim=-1).squeeze()
        trans_idx = torch.argmin(trans_norm)
        # trans_idx = 0
    else:
        # pick the smallest one
        # trans_idx = 0
        trans_norm = torch.norm(trans_model_output_raw['trans'], dim=-1).squeeze()
        trans_idx = torch.argmin(trans_norm)

    model_output = {}
    # model_output['trans'] = torch.gather(trans_model_output_raw['trans'], dim=1, index=trans_idx[:, None, None].repeat(1, 1, 3)).reshape(bs, 3)
    trans_scalar = 0.0001 if not policy_model.aff else 1.0
    # model_output['trans'] = trans_model_output_raw['trans'][:, trans_idx]
    model_output['trans'] = trans_model_output_raw['trans'][:, trans_idx] * trans_scalar


    # print("here after model_output")
    # from IPython import embed; embed()
    if viz:
        util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/infer/parent_pcd')
        util.meshcat_pcd_show(mc_vis, child_pcd_original, (0, 0, 255), 'scene/infer/child_pcd')
        if run_affordance:
            util.meshcat_pcd_show(mc_vis, pscene_pcd, (255, 0, 255), 'scene/infer/parent_pcd_norm')
        util.meshcat_pcd_show(mc_vis, cropped_parent_pcd, (255, 0, 0), 'scene/infer/cropped_parent_pcd')
        util.meshcat_pcd_show(mc_vis, child_pcd, (0, 255, 0), 'scene/infer/child_pcd_delta')
        viz_rot_wf = child_pcd_rot + policy_mi['child_start_pcd_mean'].reshape(-1, 1, 3).repeat(1, child_pcd_rot.shape[1], 1)
        util.meshcat_pcd_show(mc_vis, viz_rot_wf[0].detach().cpu().numpy(), (0, 0, 0), 'scene/infer/post_rot_world_frame')
        viz_trans_wf = child_pcd_rot + policy_mi['child_start_pcd_mean'].reshape(-1, 1, 3).repeat(1, child_pcd_rot.shape[1], 1) + model_output['trans'].reshape(-1, 1, 3).repeat(1, child_pcd_rot.shape[1], 1)
        util.meshcat_pcd_show(mc_vis, viz_trans_wf[0].detach().cpu().numpy(), (0, 0, 255), 'scene/infer/post_rot_trans_world_frame')

    ##############################################

    if 'viz_attn' in kwargs.keys():
        if kwargs['viz_attn']:

            mc_vis['scene/anchor_pts'].delete()
            mc_vis['scene/impt_pts'].delete()
            for i in range(10):
                mc_vis[f'scene/anchor_{i}'].delete()

            attn_viz = policy_model.transformer.blocks[-1].attention.attn[0, 0]
            pos_viz = policy_model.transformer.blocks[-1].attention.pos.detach().cpu().numpy().squeeze()

            viz_pts = pos_viz.shape[0]
            viz1_pts = int((viz_pts - 1) / 2)

            util.meshcat_pcd_show(mc_vis, pos_viz[:viz1_pts], (255, 0, 0), 'scene/pos/parent_pts')
            util.meshcat_pcd_show(mc_vis, pos_viz[viz1_pts:-1], (0, 0, 255), 'scene/pos/child_pts')

            util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/real/parent_pts')
            util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 255), 'scene/real/child_pts')

            k = 1000
            topk_values = torch.topk(attn_viz.flatten(), k)[0]
            topk_2d_inds = []
            for val in topk_values:
                ind = (attn_viz == val).nonzero()
                topk_2d_inds.append(ind)
            
            # cls_idx = 1024
            cls_idx = 256
            pt_idxs = []
            pt_pair_idxs = []
            for i, inds in enumerate(topk_2d_inds):
                idxs = inds.detach().cpu().numpy().squeeze().tolist()
                if idxs[0] != cls_idx:
                    # pt_pair_idxs.append((i, idxs))
                    if isinstance(idxs[0], list):
                        for val in idxs:
                            # pt_pair_idxs.append((i, [val]))
                            pt_pair_idxs.append((i, val))
                    else:
                        pt_pair_idxs.append((i, idxs))
                else:
                    pt_idxs.append(idxs[1])
                # print(idxs)
                # pt_idxs.append(idxs[1])

            # for the pairs, form groups with same points
            same_pts_dict = {}
            for i, pt_pair_idx in enumerate(pt_pair_idxs):

                # from IPython import embed; embed()
                try:
                    val_ind = pt_pair_idx[0]
                    val_ind_2d = topk_2d_inds[val_ind].squeeze()
                    attn_score = topk_values[val_ind].detach().cpu().item()
                    # attn_score = attn_viz[val_ind_2d[0], val_ind_2d[1]]
                    ix1, ix2 = pt_pair_idx[1]
                except Exception as e:
                    print(e)
                    from IPython import embed; embed()

                try:
                    if ix1 not in same_pts_dict.keys():
                        same_pts_dict[ix1] = []
                    # same_pts_dict[ix1].append(ix2)
                    # same_pts_dict[ix1].append((ix1, ix2))
                    same_pts_dict[ix1].append((ix1, ix2, attn_score))

                    if ix2 not in same_pts_dict.keys():
                        same_pts_dict[ix2] = []
                    # same_pts_dict[ix2].append(ix1)
                    same_pts_dict[ix2].append((ix1, ix2, attn_score))
                except Exception as e:
                    print(e)
                    from IPython import embed; embed()
            
            # rank these by which have the most points
            num_matches = []
            matching_points = []
            for k, v in same_pts_dict.items():
                num = len(v)
                num_matches.append(num)
                # matching_points.append(v)  # when we save only the index, no score
                matching_points.append(v)  # converts the score that we save

            num_matches = np.asarray(num_matches)
            matching_points = np.asarray(matching_points)

            ranked_by_num = np.argsort(num_matches)[::-1]
            # top_matching_points = [matching_points[ranked_by_num[0]]]
            top_matching_points = [matching_points[ranked_by_num[ind]] for ind in range(5)]
            # top_matching_points = matching_points[np.where(num_matches > 1)[0]].tolist()
            # top_matching_points = matching_points[np.where(num_matches > 3)[0]].tolist()

            anchor_pt_list = []
            key_pt_list = []
            score_list = []
            for i in range(len(top_matching_points)):
                pt_list = top_matching_points[i]
                anchor_pt = pt_list[0][1]  # we just know it's the second coordinate
                anchor_pt_list.append(anchor_pt)

                kp_list = []
                kp_score_list = []
                for pt in pt_list:
                    kp = pt[0]  
                    kp_list.append(kp)
                    score = pt[-1]
                    kp_score_list.append(score)

                key_pt_list.append(kp_list)
                score_list.append(kp_score_list)

            # sph_size = 0.005
            sph_size = 0.05
            cmap1 = plt.get_cmap('plasma')
            cmap2 = plt.get_cmap('viridis')
            color_list1 = cmap1(np.linspace(0, 1, len(anchor_pt_list)))[::-1]
            for i, anchor_idx in enumerate(anchor_pt_list):
                anchor = pos_viz[anchor_idx]
                sph = trimesh.creation.uv_sphere(sph_size)
                sph.apply_translation(anchor)
                color = (color_list1[i][:-1] * 255).astype(int)
                util.meshcat_trimesh_show(mc_vis, f'scene/anchor_pts/pt_{i}', sph, color=tuple(color))

                keypoint_idxs = key_pt_list[i]
                scores = score_list[i]
                keypoints = pos_viz[keypoint_idxs]
                color_list2 = cmap2(np.linspace(0, 1, len(keypoint_idxs)))[::-1]
                for j, kp in enumerate(keypoints):
                    score = scores[j]
                    sph = trimesh.creation.uv_sphere(sph_size)
                    sph.apply_translation(kp)
                    color = (color_list2[j][:-1] * 255).astype(int)
                    util.meshcat_trimesh_show(mc_vis, f'scene/anchor_{i}/pt_{j}', sph, color=tuple(color))

                    print('score: ', score, 'color: ', color, 'j: ', j)
                    
            impt_pts = pos_viz[pt_idxs]
            sph_list = []
            cmap = plt.get_cmap('plasma')
            color_list = cmap(np.linspace(0, 1, len(pt_idxs)))[::-1]
            for i, pt in enumerate(impt_pts):
                sph = trimesh.creation.uv_sphere(sph_size)
                sph.apply_translation(pt)
                color = (color_list[i][:-1] * 255).astype(int)
                util.meshcat_trimesh_show(mc_vis, f'scene/impt_pts/pt_{i}', sph, color=tuple(color))
                sph_list.append(sph)


            # from IPython import embed; embed()


            # mc_vis['scene/impt_pts'].delete()
            # mc_vis['scene/anchor_pts'].delete()
            # for i in range(len(anchor_pt_list)):
            #     mc_vis[f'scene/anchor_{i}'].delete()

            # # which points are the important points attending to?
            # # impt_pt_att_idxs = []
            # impt_pt_att_idxs = {}
            # # for idx in pt_idxs:
            # for idx in [pt_idxs[1]]:
            #     attn_vec = attn_viz[:, idx].detach().cpu().numpy().squeeze()
            #     attn_vec_inds_sorted = np.argsort(attn_vec)[::-1][1:] #[1:100]
            #     
            #     color_list3 = cmap2(np.linspace(0, 1, len(attn_vec_inds_sorted)))[::-1]
            #     for j, ind in enumerate(attn_vec_inds_sorted):
            #         # pt = pos_viz[idx]
            #         pt = pos_viz[ind]
            #         score = attn_vec[ind]
            #         color = (color_list3[j][:-1] * 255).astype(int)
            #         sph = trimesh.creation.uv_sphere(0.005)
            #         sph.apply_translation(pt)
            #         util.meshcat_trimesh_show(mc_vis, f'scene/attn_pts/{ind}', sph, color=tuple(color))
            #         print('score: ', score)


            from IPython import embed; embed()

    ##############################################
    model_output['rot_mat'] = rot_model_output['rot_mat']
 
    out_rot_mat = model_output['rot_mat'].detach().cpu().numpy().reshape(3, 3)
    # out_trans = model_output['trans'].detach().cpu().numpy().reshape(1, 3)
    out_trans = model_output['trans'].detach().cpu().numpy().reshape(1, 3) + delta_child_world.reshape(1, 3)

    tf1 = np.eye(4); tf1[:-1, -1] = -1.0 * np.mean(child_pcd_original, axis=0)
    tf2 = np.eye(4); tf2[:-1, :-1] = out_rot_mat
    tf3 = np.eye(4); tf3[:-1, -1] = np.mean(child_pcd_original, axis=0) + out_trans
    out_tf = np.matmul(tf3, np.matmul(tf2, tf1))

    print('Out Trans: ', out_trans)
    print('Out Rot: ', out_rot_mat)
    print('Out TF: ', out_tf)

    # util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/real/parent_pts')
    # util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 255), 'scene/real/child_pts')
    # from IPython import embed; embed()

    return out_tf


def iterative_regression_scene_combine(
                         mc_vis, parent_pcd, child_pcd, scene_model_dict, feat_model, policy_model, policy_model_refine,
                         grid_pts, viz=False, n_iters=10, return_all_child_pcds=False, *args, **kwargs):
    """
    Iteratively predict the relative transformation in one shot,
    applying each prediction to the shape and feeding the transformed
    shape back in as input for another step
    """
    out_tf_list = []
    out_tf_full = np.eye(4)
    out_child_pcd_list = []
    current_child_pcd = copy.deepcopy(child_pcd)

    # for it in range(n_iters):
    #     mc_vis[f'scene/real/child_pts_{it}'].delete()

    util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/real/parent_pts')
    util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 0), 'scene/real/child_pts_init')
    cmap = plt.get_cmap('plasma')
    color_list = cmap(np.linspace(0, 1, n_iters, dtype=np.float32))[::-1]

    for it in range(n_iters):
        run_affordance = it == 0
        if it < 1:
        # if it < (n_iters / 2):
            out_tf = single_shot_regression_scene_combine(mc_vis, parent_pcd, current_child_pcd, scene_model_dict, feat_model, policy_model, grid_pts, run_affordance=run_affordance, *args, **kwargs)
        else:
            out_tf = single_shot_regression_scene_combine(mc_vis, parent_pcd, current_child_pcd, scene_model_dict, feat_model, policy_model_refine, grid_pts, run_affordance=run_affordance, *args, **kwargs)
        current_child_pcd = util.transform_pcd(current_child_pcd, out_tf)

        # color = (color_list[it][:-1] * 255).astype(int).tolist()
        color = (color_list[it][:-1] * 255).astype(np.int8).tolist()
        # util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 255), 'scene/real/child_pts_{it}')
        util.meshcat_pcd_show(mc_vis, current_child_pcd, color, f'scene/real/child_pts_{it}')

        out_child_pcd_list.append(current_child_pcd)
        out_tf_list.append(out_tf)

        out_tf_full = np.matmul(out_tf, out_tf_full)

    # from IPython import embed; embed()
    if return_all_child_pcds:
        return out_tf_full, current_child_pcd
    else:
        return out_tf_full


def single_shot_regression_transformer_combine(mc_vis, parent_pcd, child_pcd, rpdiff_dict, policy_model_dict, 
                           viz=False, *args, **kwargs):
    """
    Directly predict the relative transformation in one shot
    """
    policy_model = policy_model_dict['trans']
    policy_model_rot = policy_model_dict['rot']

    parent_rpdiff = rpdiff_dict['parent']
    child_rpdiff = rpdiff_dict['child']
    
    rix1 = np.random.permutation(parent_pcd.shape[0])
    rix2 = np.random.permutation(child_pcd.shape[0])
    parent_pcd = parent_pcd[rix1[:1500]]
    child_pcd = child_pcd[rix2[:1500]]

    # prpdiff_pcd, prpdiff_coords = copy.deepcopy(util.center_pcd(parent_pcd)), copy.deepcopy(util.center_pcd(child_pcd, ref_pcd=parent_pcd))
    # crpdiff_pcd, crpdiff_coords = copy.deepcopy(util.center_pcd(child_pcd)), copy.deepcopy(util.center_pcd(parent_pcd, ref_pcd=child_pcd))
    prpdiff_pcd = copy.deepcopy(util.center_pcd(parent_pcd)) 
    prpdiff_coords = np.concatenate([
        copy.deepcopy(util.center_pcd(parent_pcd)),
        copy.deepcopy(util.center_pcd(child_pcd, ref_pcd=parent_pcd))])

    crpdiff_pcd = copy.deepcopy(util.center_pcd(child_pcd)) 
    crpdiff_coords = np.concatenate([
        copy.deepcopy(util.center_pcd(parent_pcd, ref_pcd=child_pcd)),
        copy.deepcopy(util.center_pcd(child_pcd))])

    # obtain descriptor values for the input (parent RPDIFF evaluated at child pcd)
    parent_rpdiff_mi = {}
    parent_rpdiff_mi['point_cloud'] = torch.from_numpy(prpdiff_pcd).float().cuda().reshape(1, -1, 3)
    parent_rpdiff_mi['coords'] = torch.from_numpy(prpdiff_coords).float().cuda().reshape(1, -1, 3)

    # obtain descriptor values for the input (child RPDIFF evaluated at parent pcd)
    child_rpdiff_mi = {}
    child_rpdiff_mi['point_cloud'] = torch.from_numpy(crpdiff_pcd).float().cuda().reshape(1, -1, 3)
    child_rpdiff_mi['coords'] = torch.from_numpy(crpdiff_coords).float().cuda().reshape(1, -1, 3)

    parent_latent = parent_rpdiff.model.extract_latent(parent_rpdiff_mi).detach()  # assumes we have already centered based on the parent
    parent_rpdiff_desc = parent_rpdiff.model.forward_latent(parent_latent, parent_rpdiff_mi['coords']).detach()

    child_latent = child_rpdiff.model.extract_latent(child_rpdiff_mi).detach()  # assumes we have already centered based on the child
    child_rpdiff_desc = child_rpdiff.model.forward_latent(child_latent, child_rpdiff_mi['coords']).detach()

    # prepare inputs to the policy
    policy_mi = {}
    policy_mi['parent_start_pcd'] = parent_rpdiff_mi['point_cloud']
    policy_mi['child_start_pcd'] = child_rpdiff_mi['point_cloud']
    # policy_mi['parent_start_pcd_mean'] = torch.mean(parent_rpdiff_mi['point_cloud'], axis=1)
    # policy_mi['child_start_pcd_mean'] = torch.mean(child_rpdiff_mi['point_cloud'], axis=1)
    # policy_mi['parent_start_pcd_mean'] = torch.from_numpy(np.mean(parent_pcd, axis=0)).float().cuda().reshape(1, -1, 3)
    # policy_mi['child_start_pcd_mean'] = torch.from_numpy(np.mean(child_pcd, axis=0)).float().cuda().reshape(1, -1, 3)
    policy_mi['parent_start_pcd_mean'] = torch.from_numpy(np.mean(parent_pcd, axis=0)).float().cuda().reshape(1, 3)
    policy_mi['child_start_pcd_mean'] = torch.from_numpy(np.mean(child_pcd, axis=0)).float().cuda().reshape(1, 3)
    policy_mi['parent_rpdiff_desc'] = parent_rpdiff_desc
    policy_mi['child_rpdiff_desc'] = child_rpdiff_desc
    policy_mi['parent_latent'] = parent_latent
    policy_mi['child_latent'] = child_latent

    # get rotation output prediction
    rot_model_output = policy_model_rot(policy_mi)

    # make a rotated version of the original point cloud
    child_start_pcd_original = policy_mi['child_start_pcd'].clone().detach()
    child_pcd_final_pred = torch.bmm(rot_model_output['rot_mat'], policy_mi['child_start_pcd'].transpose(1, 2)).transpose(2, 1)
    child_pcd_rot = child_pcd_final_pred.detach()
    
    # reencode this rotated point cloud to get new descriptor features
    n_pts = child_pcd_final_pred.shape[1]
    child_pcd_rot_child_cent = child_pcd_rot
    child_pcd_rot_parent_cent = child_pcd_rot + policy_mi['parent_start_pcd_mean'][:, None, :].repeat(1, n_pts, 1) - policy_mi['child_start_pcd_mean'][:, None, :].repeat(1, n_pts, 1)

    parent_rpdiff_mi['coords'][:, n_pts:] = child_pcd_rot_parent_cent
    child_rpdiff_mi['point_cloud'] = child_pcd_rot_child_cent
    child_rpdiff_mi['coords'][:, n_pts:] = child_pcd_rot_child_cent

    parent_rot_latent = parent_rpdiff.model.extract_latent(parent_rpdiff_mi).detach()  
    parent_rpdiff_rot_desc = parent_rpdiff.model.forward_latent(parent_rot_latent, parent_rpdiff_mi['coords']).detach()
    child_rot_latent = child_rpdiff.model.extract_latent(child_rpdiff_mi).detach()  
    child_rpdiff_rot_desc = child_rpdiff.model.forward_latent(child_rot_latent, child_rpdiff_mi['coords']).detach()
    
    # new policy input based on these RPDIFF features of rotated shape
    policy_mi['parent_rpdiff_desc'] = parent_rpdiff_rot_desc
    policy_mi['child_rpdiff_desc'] = child_rpdiff_rot_desc
    policy_mi['parent_latent'] = parent_rot_latent
    policy_mi['child_latent'] = child_rot_latent
    policy_mi['child_start_pcd'] = child_pcd_rot_child_cent
    
    # get output from transformer for translation, and combine outputs with rotation prediction
    model_output = policy_model(policy_mi)
    # print("here after model_output")
    # from IPython import embed; embed()

    ##############################################

    if 'viz_attn' in kwargs.keys():
        if kwargs['viz_attn']:
            attn_viz = policy_model.transformer.blocks[-1].attention.attn[0, 0]
            pos_viz = policy_model.transformer.blocks[-1].attention.pos.detach().cpu().numpy().squeeze()

            viz_pts = pos_viz.shape[0]
            viz1_pts = int((viz_pts - 1) / 2)

            util.meshcat_pcd_show(mc_vis, pos_viz[:viz1_pts], (255, 0, 0), 'scene/pos/parent_pts')
            util.meshcat_pcd_show(mc_vis, pos_viz[viz1_pts:-1], (0, 0, 255), 'scene/pos/child_pts')

            util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/real/parent_pts')
            util.meshcat_pcd_show(mc_vis, child_pcd, (0, 0, 255), 'scene/real/child_pts')

            k = 1000
            topk_values = torch.topk(attn_viz.flatten(), k)[0]
            topk_2d_inds = []
            for val in topk_values:
                ind = (attn_viz == val).nonzero()
                topk_2d_inds.append(ind)
            
            pt_idxs = []
            pt_pair_idxs = []
            for i, inds in enumerate(topk_2d_inds):
                idxs = inds.detach().cpu().numpy().squeeze().tolist()
                if idxs[0] != 1024:
                    # pt_pair_idxs.append((i, idxs))
                    if isinstance(idxs[0], list):
                        for val in idxs:
                            # pt_pair_idxs.append((i, [val]))
                            pt_pair_idxs.append((i, val))
                    else:
                        pt_pair_idxs.append((i, idxs))
                else:
                    pt_idxs.append(idxs[1])
                # print(idxs)
                # pt_idxs.append(idxs[1])

            # for the pairs, form groups with same points
            same_pts_dict = {}
            for i, pt_pair_idx in enumerate(pt_pair_idxs):

                # from IPython import embed; embed()
                try:
                    val_ind = pt_pair_idx[0]
                    val_ind_2d = topk_2d_inds[val_ind].squeeze()
                    attn_score = topk_values[val_ind].detach().cpu().item()
                    # attn_score = attn_viz[val_ind_2d[0], val_ind_2d[1]]
                    ix1, ix2 = pt_pair_idx[1]
                except Exception as e:
                    print(e)
                    from IPython import embed; embed()

                try:
                    if ix1 not in same_pts_dict.keys():
                        same_pts_dict[ix1] = []
                    # same_pts_dict[ix1].append(ix2)
                    # same_pts_dict[ix1].append((ix1, ix2))
                    same_pts_dict[ix1].append((ix1, ix2, attn_score))

                    if ix2 not in same_pts_dict.keys():
                        same_pts_dict[ix2] = []
                    # same_pts_dict[ix2].append(ix1)
                    same_pts_dict[ix2].append((ix1, ix2, attn_score))
                except Exception as e:
                    print(e)
                    from IPython import embed; embed()
            
            # rank these by which have the most points
            num_matches = []
            matching_points = []
            for k, v in same_pts_dict.items():
                num = len(v)
                num_matches.append(num)
                # matching_points.append(v)  # when we save only the index, no score
                matching_points.append(v)  # converts the score that we save

            num_matches = np.asarray(num_matches)
            matching_points = np.asarray(matching_points)

            ranked_by_num = np.argsort(num_matches)[::-1]
            # top_matching_points = [matching_points[ranked_by_num[0]]]
            top_matching_points = [matching_points[ranked_by_num[ind]] for ind in range(5)]
            # top_matching_points = matching_points[np.where(num_matches > 1)[0]].tolist()
            # top_matching_points = matching_points[np.where(num_matches > 3)[0]].tolist()

            anchor_pt_list = []
            key_pt_list = []
            score_list = []
            for i in range(len(top_matching_points)):
                pt_list = top_matching_points[i]
                anchor_pt = pt_list[0][1]  # we just know it's the second coordinate
                anchor_pt_list.append(anchor_pt)

                kp_list = []
                kp_score_list = []
                for pt in pt_list:
                    kp = pt[0]  
                    kp_list.append(kp)
                    score = pt[-1]
                    kp_score_list.append(score)

                key_pt_list.append(kp_list)
                score_list.append(kp_score_list)

            cmap1 = plt.get_cmap('plasma')
            cmap2 = plt.get_cmap('viridis')
            color_list1 = cmap1(np.linspace(0, 1, len(anchor_pt_list)))[::-1]
            for i, anchor_idx in enumerate(anchor_pt_list):
                anchor = pos_viz[anchor_idx]
                sph = trimesh.creation.uv_sphere(0.005)
                sph.apply_translation(anchor)
                color = (color_list1[i][:-1] * 255).astype(int)
                util.meshcat_trimesh_show(mc_vis, f'scene/anchor_pts/pt_{i}', sph, color=tuple(color))

                keypoint_idxs = key_pt_list[i]
                scores = score_list[i]
                keypoints = pos_viz[keypoint_idxs]
                color_list2 = cmap2(np.linspace(0, 1, len(keypoint_idxs)))[::-1]
                for j, kp in enumerate(keypoints):
                    score = scores[j]
                    sph = trimesh.creation.uv_sphere(0.005)
                    sph.apply_translation(kp)
                    color = (color_list2[j][:-1] * 255).astype(int)
                    util.meshcat_trimesh_show(mc_vis, f'scene/anchor_{i}/pt_{j}', sph, color=tuple(color))

                    print('score: ', score, 'color: ', color, 'j: ', j)
                    
            impt_pts = pos_viz[pt_idxs]
            sph_list = []
            cmap = plt.get_cmap('plasma')
            color_list = cmap(np.linspace(0, 1, len(pt_idxs)))[::-1]
            for i, pt in enumerate(impt_pts):
                sph = trimesh.creation.uv_sphere(0.005)
                sph.apply_translation(pt)
                color = (color_list[i][:-1] * 255).astype(int)
                util.meshcat_trimesh_show(mc_vis, f'scene/impt_pts/pt_{i}', sph, color=tuple(color))
                sph_list.append(sph)


            # from IPython import embed; embed()


            # mc_vis['scene/impt_pts'].delete()
            # mc_vis['scene/anchor_pts'].delete()
            # for i in range(len(anchor_pt_list)):
            #     mc_vis[f'scene/anchor_{i}'].delete()

            # # which points are the important points attending to?
            # # impt_pt_att_idxs = []
            # impt_pt_att_idxs = {}
            # # for idx in pt_idxs:
            # for idx in [pt_idxs[1]]:
            #     attn_vec = attn_viz[:, idx].detach().cpu().numpy().squeeze()
            #     attn_vec_inds_sorted = np.argsort(attn_vec)[::-1][1:] #[1:100]
            #     
            #     color_list3 = cmap2(np.linspace(0, 1, len(attn_vec_inds_sorted)))[::-1]
            #     for j, ind in enumerate(attn_vec_inds_sorted):
            #         # pt = pos_viz[idx]
            #         pt = pos_viz[ind]
            #         score = attn_vec[ind]
            #         color = (color_list3[j][:-1] * 255).astype(int)
            #         sph = trimesh.creation.uv_sphere(0.005)
            #         sph.apply_translation(pt)
            #         util.meshcat_trimesh_show(mc_vis, f'scene/attn_pts/{ind}', sph, color=tuple(color))
            #         print('score: ', score)


            from IPython import embed; embed()

    ##############################################
    model_output['quat'] = rot_model_output['quat']
    model_output['unnorm_quat'] = rot_model_output['unnorm_quat']
    model_output['rot_mat'] = rot_model_output['rot_mat']
 
    out_rot_mat = model_output['rot_mat'].detach().cpu().numpy().reshape(3, 3)
    out_trans = model_output['trans'].detach().cpu().numpy().reshape(1, 3)

    tf1 = np.eye(4); tf1[:-1, -1] = -1.0 * np.mean(child_pcd, axis=0)
    tf2 = np.eye(4); tf2[:-1, :-1] = out_rot_mat
    tf3 = np.eye(4); tf3[:-1, -1] = np.mean(child_pcd, axis=0) + out_trans
    out_tf = np.matmul(tf3, np.matmul(tf2, tf1))

    print('Out Trans: ', out_trans)
    print('Out Rot: ', out_rot_mat)
    print('Out TF: ', out_tf)

    return out_tf


def iterative_regression_transformer_combine(
                         mc_vis, parent_pcd, child_pcd, rpdiff_dict, policy_model_dict, 
                         viz=False, n_iters=10, return_all_child_pcds=False, *args, **kwargs):
    """
    Iteratively predict the relative transformation in one shot,
    applying each prediction to the shape and feeding the transformed
    shape back in as input for another step
    """
    out_tf_list = []
    out_tf_full = np.eye(4)
    out_child_pcd_list = []
    current_child_pcd = copy.deepcopy(child_pcd)
    for it in range(n_iters):
        out_tf = single_shot_regression_transformer_combine(mc_vis, parent_pcd, current_child_pcd, rpdiff_dict, policy_model_dict, *args, **kwargs)
        current_child_pcd = util.transform_pcd(current_child_pcd, out_tf)
        out_child_pcd_list.append(current_child_pcd)
        out_tf_list.append(out_tf)

        out_tf_full = np.matmul(out_tf, out_tf_full)

    if return_all_child_pcds:
        return out_tf_full, current_child_pcd
    else:
        return out_tf_full


policy_inference_methods_dict = {
    'single_shot_regression': single_shot_regression,
    'iterative_regression': iterative_regression,
    'single_shot_regression_transformer_combine': single_shot_regression_transformer_combine,
    'iterative_regression_transformer_combine': iterative_regression_transformer_combine,
    'single_shot_regression_feat_combine': single_shot_regression_feat_combine,
    'iterative_regression_feat_combine': iterative_regression_feat_combine,
    'single_shot_regression_scene_combine': single_shot_regression_scene_combine,
    'iterative_regression_scene_combine': iterative_regression_scene_combine,
    'single_shot_regression_scene_combine_succ_cls': single_shot_regression_scene_combine_sc,
    'iterative_regression_scene_combine_succ_cls': iterative_regression_scene_combine_sc,
    'multistep_regression_scene_combine_succ_cls': multistep_regression_scene_combine_sc,
    'energy_optimization': energy_optimization
}
