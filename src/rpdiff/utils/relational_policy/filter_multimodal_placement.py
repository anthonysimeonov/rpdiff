import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from rpdiff.utils import util
from rpdiff.utils.mesh_util import inside_mesh
from rpdiff.utils.mesh_util.three_util import trimesh_combine


def remove_redundant_tf(policy_mi, voxel_idx_final, rot_idx_final, trans_idx_final, mc_vis):

    filtered_policy_mi = {k: v.clone() for k, v in policy_mi.items()}

    idx = 0

    min_dists = []
    max_dists = []

    # for idx in range(filtered_policy_mi['child_start_pcd'].shape[0]-1):
    while True:

        # print(f'Iteration: {idx}, total {filtered_policy_mi["child_start_pcd"].shape[0]} filtered')
        if idx >= filtered_policy_mi['child_start_pcd'].shape[0] - 1:
            break

        # get a copy of the current ones we are comparing against
        child_pcd_compare = filtered_policy_mi['child_start_pcd'][idx+1:].clone()
        child_mean_compare = filtered_policy_mi['child_start_pcd_mean'][idx+1:].clone()
        compare_pcd_world = child_pcd_compare + child_mean_compare.reshape(-1, 1, 3).repeat((1, child_pcd_compare.shape[1], 1))

        # get the one we are comparing to the others
        child_pcd_ref = filtered_policy_mi['child_start_pcd'][idx].clone().reshape(1, -1, 3)
        child_mean_ref = filtered_policy_mi['child_start_pcd_mean'][idx].clone()
        ref_pcd_world = child_pcd_ref + child_mean_ref.reshape(1, 1, 3).repeat((1, child_pcd_ref.shape[1], 1))

        # compute the pcd distance to the others
        dist = torch.norm(compare_pcd_world - ref_pcd_world.repeat((compare_pcd_world.shape[0], 1, 1)), dim=-1).mean(-1)
        # dist = torch.norm(compare_pcd_world - ref_pcd_world.repeat((compare_pcd_world.shape[0], 1, 1)), dim=-1).sum(-1)
        # dist = torch.norm(child_pcd_compare - child_pcd_ref.reshape(1, -1, 3).repeat((child_pcd_compare.shape[0], 1, 1)), dim=-1).sum(-1)

        min_dists.append(dist.min().item())
        max_dists.append(dist.max().item())

        # # use some threshold to remove the redundant ones that are the ~same
        thresh = 0.005
        # thresh = 0.01
        # thresh = 0.015
        redundant_idx = torch.where(dist < thresh)[0]
        keep_idx = torch.where(dist > thresh)[0]

        box = trimesh.PointCloud(ref_pcd_world.detach().cpu().numpy().squeeze()).bounding_box_oriented.to_mesh()

        # util.meshcat_trimesh_show(mc_vis, f'scene/filter/redundant_box_{idx}/child_pts_ref', box, (0, 0, 128))
        # visualize our reference and those that are detected to be removed
        # for ii in range(redundant_idx.shape[0]):
        #     # if viz:
        #     red_idx = redundant_idx[ii]
        #     box = trimesh.PointCloud(compare_pcd_world[red_idx].detach().cpu().numpy().squeeze()).bounding_box_oriented.to_mesh()
        #     # box = trimesh.PointCloud(child_pcd_compare[red_idx].detach().cpu().numpy().squeeze()).bounding_box_oriented.to_mesh()
        #     util.meshcat_trimesh_show(mc_vis, f'scene/filter/redundant_box_{idx}/remove/child_pts_box_{ii}', box, (128, 0, 0))

        # get the total keep index - everything up to our reference, and the shifted version of everything we keep
        total_keep_idx = torch.cat((torch.arange(idx+1).cuda(), keep_idx + idx + 1))

        out_dict = {}
        for k in filtered_policy_mi.keys():
            try:
                out_dict[k] = filtered_policy_mi[k][total_keep_idx].clone()
                # if idx == 0:
                #     out_dict[k] = torch.cat((
                #         filtered_policy_mi[k][:idx+1].unsqueeze(0),
                #         filtered_policy_mi[k][idx+1:][keep_idx]), dim=0).clone()
                # else:
                #     out_dict[k] = torch.cat((
                #         filtered_policy_mi[k][:idx+1],
                #         filtered_policy_mi[k][idx+1:][keep_idx]), dim=0).clone()
            except Exception as e:
                # print(f'Exception: {e}')
                out_dict[k] = filtered_policy_mi[k].clone()

        filtered_policy_mi = out_dict
        # filtered_policy_mi = {k: v[idx+1:][keep_idx].clone() for k, v in filtered_policy_mi.items()}

        # mc_vis[f'scene/filter/redundant_box_{idx}'].delete()

        voxel_idx_final = voxel_idx_final[total_keep_idx]
        rot_idx_final = rot_idx_final[total_keep_idx]
        trans_idx_final = trans_idx_final[total_keep_idx]
        # rot_mat_queries_final = rot_mat_queries_final[total_keep_idx]
        # trans_queries_final = trans_queries_final[total_keep_idx]

        idx = idx + 1

    n_final = filtered_policy_mi['child_start_pcd'].shape[0]
    # cmap = plt.get_cmap('inferno')
    cmap = plt.get_cmap('cool')
    color_list = cmap(np.linspace(0.1, 0.9, n_final, dtype=np.float32))[::-1]
    for ii in range(n_final):
        # if viz:
        color = (color_list[ii][:-1] * 255).astype(np.uint8).tolist()
        pcd_cent = filtered_policy_mi['child_start_pcd'][ii].detach().cpu().numpy().squeeze()
        pcd_mean = filtered_policy_mi['child_start_pcd_mean'][ii].detach().cpu().numpy().squeeze()
        box = trimesh.PointCloud(pcd_cent + pcd_mean).bounding_box_oriented.to_mesh()
        # box = trimesh.PointCloud(child_pcd_compare[red_idx].detach().cpu().numpy().squeeze()).bounding_box_oriented.to_mesh()
        # util.meshcat_trimesh_show(mc_vis, f'scene/filter/redundant_box_{idx}/remove/child_pts_box_{ii}', box, (128, 0, 0))
        # util.meshcat_trimesh_show(mc_vis, f'scene/infer/filter/box_{ii}', box, color)
        # util.meshcat_pcd_show(mc_vis, pcd_cent + pcd_mean, (255, 0, 128), f'scene/infer/filter_pcd/pcd_{ii}', size=0.002)
        util.meshcat_pcd_show(mc_vis, pcd_cent + pcd_mean, color, f'scene/infer/filter_pcd/pcd_{ii}', size=0.002)

    # print(f'Done filtering!')
    # from IPython import embed; embed()

    # remove everything from the rest of these
    return filtered_policy_mi, voxel_idx_final, rot_idx_final, trans_idx_final #, rot_mat_queries_final, trans_queries_final
    
def manual_place_success(ppcd_world, cpcd_world, mc_vis):

    # get child point cloud bounding box mesh
    cpcd_box = trimesh.PointCloud(cpcd_world).bounding_box_oriented.to_mesh()

    # approximate intersection between the two shapes = query parent points inside of child points
    p_occ_values = inside_mesh.check_mesh_contains(cpcd_box, ppcd_world)
    p_occ_inds = np.where(p_occ_values)[0]
    # print(f'Number of inside parent points: {p_occ_inds.shape[0]}, (total: {ppcd_world.shape[0]})')

    p_occ_score = 1 - p_occ_inds.shape[0] * 1.0 / ppcd_world.shape[0]

    # check if all of object is inside parent bounding box
    ppcd_box = trimesh.PointCloud(ppcd_world).bounding_box_oriented.to_mesh()
    c_occ_values = inside_mesh.check_mesh_contains(ppcd_box, cpcd_world)

    # c_occ_all = np.all(c_occ_values > 0.0)
    # c_occ_score = c_occ_all.astype(np.float32).item()
    c_occ_inds = np.where(c_occ_values)[0]
    c_occ_score = c_occ_inds.shape[0] * 1.0 / cpcd_world.shape[0]

    score = c_occ_score * p_occ_score

    # sz = 0.008
    # util.meshcat_pcd_show(
    #     mc_vis, 
    #     ppcd_world, 
    #     (255, 0, 0), 
    #     f'scene/infer/manual_success_parent',
    #     size=sz)
    # util.meshcat_pcd_show(
    #     mc_vis, 
    #     cpcd_world, 
    #     (0, 0, 255), 
    #     f'scene/infer/manual_success_child',
    #     size=sz)

    # print(f'Parent score: {p_occ_score}, Child score: {c_occ_score}')
    # from IPython import embed; embed()

    return score


def manual_place_success_gt(ppcd_world, cpcd_world, mesh_dict, mc_vis):

    multi = False
    if mesh_dict.get('multi') is not None:
        if mesh_dict['multi']:
            multi = True

    if multi:
        parent_mesh_list = mesh_dict['parent_file']
        parent_scale_list = mesh_dict['parent_scale']
        parent_pose_list = [util.matrix_from_list(val) for val in mesh_dict['parent_pose']]
        child_mesh_list = mesh_dict['child_file']
        child_scale_list = mesh_dict['child_scale']
        child_pose_list = [util.matrix_from_list(val) for val in mesh_dict['child_pose']]

        parent_tmesh = trimesh_combine(parent_mesh_list, parent_pose_list, parent_scale_list)
        child_tmesh = trimesh_combine(child_mesh_list, child_pose_list, child_scale_list)
    else:
        parent_mesh = mesh_dict['parent_file']
        parent_scale = mesh_dict['parent_scale']
        parent_pose = mesh_dict['parent_pose']
        child_mesh = mesh_dict['child_file']
        child_scale = mesh_dict['child_scale']
        child_pose = mesh_dict['child_pose']

        parent_tmesh = trimesh.load(parent_mesh).apply_scale(parent_scale).apply_transform(util.matrix_from_list(parent_pose))
        child_tmesh = trimesh.load(child_mesh).apply_scale(child_scale).apply_transform(util.matrix_from_list(child_pose))

    # get child point cloud bounding box mesh
    cpcd_bb = trimesh.PointCloud(cpcd_world).bounding_box_oriented
    cpcd_box = cpcd_bb.to_mesh()
    cpcd_pts = cpcd_bb.sample_volume(5000)

    # approximate intersection between the two shapes = query parent points inside of child points
    p_occ_values = inside_mesh.check_mesh_contains(cpcd_box, ppcd_world)
    p_occ_inds = np.where(p_occ_values)[0]
    # print(f'Number of inside parent points: {p_occ_inds.shape[0]}, (total: {ppcd_world.shape[0]})')

    p_occ_score = 1 - p_occ_inds.shape[0] * 1.0 / ppcd_world.shape[0]

    # check if all of object is inside parent bounding box
    ppcd_box = trimesh.PointCloud(ppcd_world).bounding_box_oriented.to_mesh()
    c_occ_values = inside_mesh.check_mesh_contains(ppcd_box, cpcd_world)

    # c_occ_all = np.all(c_occ_values > 0.0)
    # c_occ_score = c_occ_all.astype(np.float32).item()
    c_occ_inds = np.where(c_occ_values)[0]
    c_occ_score = c_occ_inds.shape[0] * 1.0 / cpcd_world.shape[0]

    gt_occ_values = inside_mesh.check_mesh_contains(parent_tmesh, cpcd_pts)
    gt_occ_inds = np.where(gt_occ_values)[0]
    # gt_occ_score = np.all(gt_occ_values > 0.0).astype(np.float32).item()
    gt_occ_score = 1 - gt_occ_inds.shape[0] * 1.0 / cpcd_pts.shape[0]

    score = c_occ_score * p_occ_score * gt_occ_score

    # sz = 0.008
    # util.meshcat_pcd_show(
    #     mc_vis, 
    #     ppcd_world, 
    #     (255, 0, 0), 
    #     f'scene/infer/manual_success_parent',
    #     size=sz)
    # util.meshcat_pcd_show(
    #     mc_vis, 
    #     cpcd_world, 
    #     (0, 0, 255), 
    #     f'scene/infer/manual_success_child',
    #     size=sz)

    # print(f'Parent score: {p_occ_score}, Child score: {c_occ_score}, GT score: {gt_occ_score}')
    # from IPython import embed; embed()

    return score


def compute_coverage(out_poses_wf, avail_poses_wf, out_scores, mc_vis, t_thresh=0.035, t_thresh_prec=0.04, r_thresh_deg=5):
    from scipy.spatial.transform import Rotation as R
    
    # print("here to compute coverage")
    # from IPython import embed; embed()

    mc_ex = 'scene/compute_coverage'

    for p_idx, pose in enumerate(avail_poses_wf):
        util.meshcat_frame_show(mc_vis, f'{mc_ex}/avail_poses/{p_idx}', pose)

    for p_idx, pose in enumerate(out_poses_wf):
        util.meshcat_frame_show(mc_vis, f'{mc_ex}/predicted_poses/{p_idx}', pose)
    
    min_trans_dists = []
    min_rot_dists = []
    min_overall_dists = []
    min_trans_rot_dists = []

    # min_trans_idx = []
    # min_rot_idx = []
    min_overall_idx = []
    for i, a_pose in enumerate(avail_poses_wf):
        min_trans = np.inf
        min_rot = np.inf
        min_trans_rot_dist = (None, None)
        min_dist = np.inf

        a_rotmat = a_pose[:-1, :-1] 
        for j, b_pose in enumerate(out_poses_wf):
            trans_ = np.linalg.norm(a_pose[:-1, -1] - b_pose[:-1, -1], axis=-1)

            b_rotmat = b_pose[:-1, :-1]
            qa = R.from_matrix(a_rotmat).as_quat()
            qb = R.from_matrix(b_rotmat).as_quat()
            
            quat_scalar_prod = np.sum(qa * qb)
            rot_ = 1 - quat_scalar_prod**2

            dist_ = trans_ + rot_
            # print(f'Idx: {j} -- t: {trans_}, R: {rot_}, d: {dist_}')

            # if trans_ < min_trans:
            #     min_trans = trans_
            # if rot_ < min_rot:
            #     min_rot = rot_
            # if dist_ < min_dist:
            # if trans_ < min_trans and rot_ < min_rot:
            if trans_ < min_trans and rot_ < min_rot or (trans_ < t_thresh and rot_ < np.deg2rad(r_thresh_deg)):
                min_dist = dist_
                min_trans = trans_
                min_rot = rot_
                min_trans_rot_dist = (trans_, rot_)
                min_dist_idx = j

        # min_trans_dists.append(min_trans)
        # min_rot_dists.append(min_rot)
        min_overall_dists.append(min_dist)
        min_trans_rot_dists.append(min_trans_rot_dist)
        min_overall_idx.append(min_dist_idx)

    
    n_matching = []
    for i, a_pose in enumerate(avail_poses_wf):
        for j, b_pose in enumerate(out_poses_wf):
            if min_overall_idx[i] == j:

                t, R_err = min_trans_rot_dists[i]

                matching = (t < t_thresh) and (R_err < np.deg2rad(r_thresh_deg))
                if not matching:
                    util.meshcat_frame_show(mc_vis, f'{mc_ex}/no_matching_poses/{i}/gt', a_pose)
                    continue

                util.meshcat_frame_show(mc_vis, f'{mc_ex}/matching_poses/{i}/gt', a_pose)
                util.meshcat_frame_show(mc_vis, f'{mc_ex}/matching_poses/{i}/pred', b_pose)

                util.meshcat_frame_show(mc_vis, f'{mc_ex}/matching_poses2/gt/{i}', a_pose)
                util.meshcat_frame_show(mc_vis, f'{mc_ex}/matching_poses2/pred/{i}', b_pose)

                n_matching.append(True)

    min_trans_rot_dists_prec = []
    min_overall_idx_prec = []
    for j, b_pose in enumerate(out_poses_wf):
        min_trans = np.inf
        min_rot = np.inf
        min_trans_rot_dist = (None, None)
        min_dist = np.inf

        b_rotmat = b_pose[:-1, :-1]
        for i, a_pose in enumerate(avail_poses_wf):
            trans_ = np.linalg.norm(a_pose[:-1, -1] - b_pose[:-1, -1], axis=-1)

            a_rotmat = a_pose[:-1, :-1] 
            qa = R.from_matrix(a_rotmat).as_quat()
            qb = R.from_matrix(b_rotmat).as_quat()
            
            quat_scalar_prod = np.sum(qa * qb)
            rot_ = 1 - quat_scalar_prod**2

            dist_ = trans_ + rot_
            # print(f'Idx: {j} -- t: {trans_}, R: {rot_}, d: {dist_}')

            # if trans_ < min_trans:
            #     min_trans = trans_
            # if rot_ < min_rot:
            #     min_rot = rot_
            # if dist_ < min_dist:
            if (trans_ < min_trans and rot_ < min_rot) or (trans_ < t_thresh_prec and rot_ < np.deg2rad(r_thresh_deg)):
                min_dist = dist_
                min_trans = trans_
                min_rot = rot_
                min_trans_rot_dist = (trans_, rot_)
                min_dist_idx = i

        # min_trans_dists.append(min_trans)
        # min_rot_dists.append(min_rot)
        min_trans_rot_dists_prec.append(min_trans_rot_dist)
        min_overall_idx_prec.append(min_dist_idx)

    n_matching_prec = []
    for j, b_pose in enumerate(out_poses_wf):
        for i, a_pose in enumerate(avail_poses_wf):
            if min_overall_idx_prec[j] == i:

                t, R_err = min_trans_rot_dists_prec[j]

                success = (t < t_thresh_prec) and (R_err < np.deg2rad(r_thresh_deg))
                if not success:
                    util.meshcat_frame_show(mc_vis, f'{mc_ex}/prec_not_success/{i}/pred', b_pose)
                    continue

                util.meshcat_frame_show(mc_vis, f'{mc_ex}/prec_success_poses/{i}/gt', a_pose)
                util.meshcat_frame_show(mc_vis, f'{mc_ex}/prec_success_poses/{i}/pred', b_pose)

                util.meshcat_frame_show(mc_vis, f'{mc_ex}/prec_success_poses/gt/{i}', a_pose)
                util.meshcat_frame_show(mc_vis, f'{mc_ex}/prec_success_poses/pred/{i}', b_pose)

                n_matching_prec.append(True)


    match_ratio = 1.0 * len(n_matching) / len(avail_poses_wf)
    recall = 1.0 * len(n_matching) / len(avail_poses_wf)
    precision = 1.0 * len(n_matching_prec) / len(out_poses_wf)
    print(f'''
        ***[Compute Coverage]***

        Match ratio/Recall: {match_ratio} (num matches: {len(n_matching)}, total: {len(avail_poses_wf)}
        Precision: {precision} (num success: {len(n_matching_prec)}, total: {len(out_poses_wf)}

        ************************
    ''')
    # print(f'***[Compute Coverage]***\n\nMatch ratio: {match_ratio} (num matches: {len(n_matching)}, total: {len(avail_poses_wf)}\n\n*********')
            
    out_coverage_dict = {}
    out_coverage_dict['match_ratio'] = match_ratio
    out_coverage_dict['recall'] = recall
    out_coverage_dict['precision'] = precision
    out_coverage_dict['n_avail'] = len(avail_poses_wf)
    out_coverage_dict['n_pred'] = len(out_poses_wf)
    out_coverage_dict['out_scores'] = out_scores
    out_coverage_dict['min_trans_rot_dists'] = min_trans_rot_dists
    out_coverage_dict['min_trans_rot_dists_prec'] = min_trans_rot_dists_prec
    out_coverage_dict['r_thresh_deg'] = r_thresh_deg
    out_coverage_dict['t_thresh'] = t_thresh
    out_coverage_dict['t_thresh_prec'] = t_thresh_prec
    
    # print("Here in out coverage compute")
    # from IPython import embed; embed()

    return out_coverage_dict
