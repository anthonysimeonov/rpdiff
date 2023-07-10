import os, os.path as osp
import json
import copy
import numpy as np
import time
import torch
import torch.nn as nn

import trimesh
import matplotlib.pyplot as plt

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

from rpdiff.utils import util, path_util
from rpdiff.utils.mesh_util import three_util
from rpdiff.utils.torch_util import angle_axis_to_rotation_matrix, transform_pcd_torch
from rpdiff.utils.torch3d_util import matrix_to_quaternion, quaternion_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix
from rpdiff.utils.torch_scatter_utils import fps_downsample

from rpdiff.training.train_util import crop_pcd_batch
from rpdiff.utils import batch_pcd_util

from meshcat import Visualizer

# MC_SIZE = 0.0015
MC_SIZE = 0.004
# MC_SIZE = 0.007


def detach_dict(in_dict: dict) -> dict:
    return {k: v.detach() for k, v in in_dict.items()}


def multistep_regression_scene_combine_sc(
        mc_vis: Visualizer, 
        parent_pcd: np.ndarray, child_pcd: np.ndarray, 
        coarse_aff_model: nn.Module, 
        pose_refine_model: nn.Module, 
        success_model: nn.Module,
        scene_scale: float, scene_mean: np.ndarray,
        grid_pts: np.ndarray, rot_grid: np.ndarray=None, 
        viz: bool=False, n_iters: int=10, 
        return_all_child_pcds: bool=False, no_parent_crop: bool=False, 
        return_top: bool=True, with_coll: bool=False, 
        run_affordance: bool=True, init_k_val: int=10,
        no_sc_score: bool=False, init_parent_mean: bool=False, 
        init_orig_ori: bool=False, refine_anneal: bool=False, 
        add_per_iter_noise: bool=False, per_iter_noise_kwargs: bool=None,
        variable_size_crop: bool=False, timestep_emb_decay_factor: int=20,
        *args, **kwargs):
    """
    Relative pose predictions made in multiple iterative steps. 
    Uses: 
    - a final success classifier
    - potentially two versions of the regression model
    - an initial voxel affordance model to provide an initial translation
      and local scene point cloud crop
    - potentially a separate feature encoder to provide extra per-point
      features

    Args:
        mc_vis (meshcat.Visualizer): meshcat visualization handler
        parent_pcd (np.ndarray): Nx3 point cloud of parent/A object
        child_pcd (np.ndarray): Nx3 point cloud of child/B object
        coarse_aff_model (torch.nn.Module): 3D CNN point cloud encoder 
            (local pointnet + 3D CNN) and the output NN that takes 
            per-voxel scene features and predicts scores. Used to 
            obtain an initialization for translating + cropping the 
            child point cloud. Can also be used to initialize the rotation (WIP).
        pose_refine_model (torch.nn.Module): NN that predicts 
            rotation and translation from point cloud A/B pair input
        success_model (torch.nn.Module): NN that takes parent/A and final
            transformed child/B point cloud as input and predicts an overall
            score of quality. Used to select among multimodal output predictions.
        grid_pts (np.ndarray): Voxelized raster points corresponding to the
            coordinates of the scene model output ("voxel affordance"). Used
            to form initial translation in the world frame
        rot_grid (np.ndarray): Array of uniformly distributed 3D rotation matrices
        viz (bool): If True, visualize the predictions along the way
        n_iters (int): Number of refinement steps to take
        return_all_child_pcd (bool): If True, output all of the child point clouds
            (might be useful for visualizing the refinement, not used in this version)
        no_parent_crop (bool): If True, don't run the scene model and don't perform
            any cropping of the parent/A point cloud.
        return_top (bool): If True, return the TOP scoring among the multimodal
            outputs. Otherwise, randomly sample ANY of the top-k scoring
        with_coll (bool): If True, also include some form of collision checking
            in the success scoring after multimodal prediction is all done
        run_affordance (bool): If True, use trained model to predict initial
            translation positions. Else, randomly sample. 
        init_k_val (int): Number of initial positions to sample
        no_sc_score (bool): If True, don't use the success classifier to score,
            just assign everything an equal score of 1.0

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix representing transformation
            of child/B point cloud in the world frame 
    """

    if rot_grid is None:
        rot_grid = util.generate_healpix_grid(size=1e4) 

    # if compute_coverage_scores:
    if util.exists_and_true(kwargs, 'compute_coverage_scores'):
        assert 'mesh_dict' in kwargs, 'Required to pass in "mesh_dict" when using "compute_coverage_scores"!'

        mesh_dict = kwargs['mesh_dict']
        parent_mesh_fname = mesh_dict['parent_file']
        parent_scale = mesh_dict['parent_scale']
        parent_pose_mat = [util.matrix_from_list(val) for val in mesh_dict['parent_pose']]
        parent_tmesh_origin_list = []
        parent_tmesh_posed_list = []
        for pidx, pfn in enumerate(parent_mesh_fname):
            ptmesh_orig = trimesh.load(pfn, process=False).apply_scale(parent_scale[pidx])
            ptmesh_posed = ptmesh_orig.copy().apply_transform(parent_pose_mat[pidx])
            parent_tmesh_origin_list.append(ptmesh_orig)
            parent_tmesh_posed_list.append(ptmesh_posed)

        if len(parent_tmesh_origin_list) > 1:
            parent_tmesh_origin = trimesh.util.concatenate(parent_tmesh_origin_list)
            parent_tmesh_posed = trimesh.util.concatenate(parent_tmesh_posed_list)
        else:
            parent_tmesh_origin = parent_tmesh_origin_list[0]
            parent_tmesh_posed = parent_tmesh_posed_list[0]

        child_mesh_fname = mesh_dict['child_file'][0]
        child_scale = mesh_dict['child_scale'][0]
        child_pose_mat = util.matrix_from_list(mesh_dict['child_pose'][0])
        child_tmesh_origin = trimesh.load(child_mesh_fname).apply_scale(child_scale)
        child_tmesh_posed = child_tmesh_origin.copy().apply_transform(child_pose_mat)
        
        mc_ex = 'scene/compute_coverage'
        util.meshcat_trimesh_show(mc_vis, f'{mc_ex}/parent_posed', parent_tmesh_posed, opacity=0.3)
        util.meshcat_trimesh_show(mc_vis, f'{mc_ex}/child_posed', child_tmesh_posed, opacity=0.3)

        import os.path as osp
        from scipy.spatial.transform import Rotation as R
            
        # get the list of available poses
        if 'book' in child_mesh_fname and 'bookshelf' in parent_mesh_fname[0]:
            bookshelf_name = parent_mesh_fname[0].split('/')[-1].replace('.obj', '').replace('_dec', '')
            saved_available_poses_fname = osp.join(parent_mesh_fname[0].split(bookshelf_name)[0], 'open_slot_poses', bookshelf_name + '_open_slot_poses.txt')
            loaded_poses = np.loadtxt(saved_available_poses_fname)
            loaded_poses = [util.matrix_from_list(pose) for pose in loaded_poses]

            # get avail poses in the world frame
            avail_poses_wf_base = [np.matmul(parent_pose_mat[0], pose) for pose in loaded_poses]
            avail_poses_wf = []
            for p_idx, pose in enumerate(avail_poses_wf_base):
                # get all four orientations that work
                r1 = R.from_euler('xyz', [0, 0, 0]).as_matrix()
                r2 = R.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
                # r3 = R.from_euler('xyz', [0, 0, np.pi]).as_matrix()
                # r4 = R.from_euler('xyz', [np.pi, 0, np.pi]).as_matrix()
                r3 = R.from_euler('xyz', [0, np.pi, 0]).as_matrix()
                r4 = R.from_euler('xyz', [np.pi, np.pi, 0]).as_matrix()

                tf1 = np.eye(4); tf1[:-1, :-1] = r1
                tf2 = np.eye(4); tf2[:-1, :-1] = r2
                tf3 = np.eye(4); tf3[:-1, :-1] = r3
                tf4 = np.eye(4); tf4[:-1, :-1] = r4

                p1 = np.matmul(pose, tf1)
                p2 = np.matmul(pose, tf2)
                p3 = np.matmul(pose, tf3)
                p4 = np.matmul(pose, tf4)
                
                all_poses_to_save = [p1, p2, p3, p4]

                for p_to_save in all_poses_to_save:

                    a_rotmat = p_to_save[:-1, :-1]
                    close_to_existing = False
                    for p2_idx, pose2 in enumerate(avail_poses_wf):
                        trans_ = np.linalg.norm(p_to_save[:-1, -1] - pose2[:-1, -1], axis=-1)

                        b_rotmat = pose2[:-1, :-1]
                        qa = R.from_matrix(a_rotmat).as_quat()
                        qb = R.from_matrix(b_rotmat).as_quat()
                        
                        quat_scalar_prod = np.sum(qa * qb)
                        rot_ = 1 - quat_scalar_prod**2

                        if trans_ < 0.02 and rot_ < np.deg2rad(5):
                            close_to_existing = True
                            break
                
                    if not close_to_existing:
                        avail_poses_wf.append(p_to_save)

        elif 'can' in child_mesh_fname and 'cabinet' in parent_mesh_fname[0]:
            cabinet_name = parent_mesh_fname[0].split('/')[-1].replace('.obj', '').replace('_dec', '')
            saved_available_poses_fname = osp.join(parent_mesh_fname[0].split(cabinet_name)[0], 'open_slot_poses', cabinet_name + '_open_slot_poses.npz')
            loaded_poses = np.load(saved_available_poses_fname, allow_pickle=True)
            avail_pose_info_all = loaded_poses['avail_top_poses']

            child_h = child_tmesh_origin.extents[-1]
            child_r = child_tmesh_origin.extents[0]/2

            top_poses = [pose_info['pose'] for pose_info in avail_pose_info_all]
            base_poses = []
            for pose in top_poses:
                base_pose = pose.copy()
                base_pose[2, -1] += child_h / 2
                base_poses.append(base_pose)
            avail_poses_wf_base = [np.matmul(parent_pose_mat[0], pose) for pose in base_poses]
            avail_poses_wf = []
            for p_idx, pose in enumerate(avail_poses_wf_base):
                # get all orientations that work

                r1 = R.from_euler('xyz', [0, 0, 0]).as_matrix()
                r2 = R.from_euler('xyz', [np.pi, 0, 0]).as_matrix()

                tf1 = np.eye(4); tf1[:-1, :-1] = r1
                tf2 = np.eye(4); tf2[:-1, :-1] = r2

                p1 = np.matmul(pose, tf1)
                p2 = np.matmul(pose, tf2)

                avail_poses_wf.append(p1)
                avail_poses_wf.append(p2)

                all_poses_to_save = [p1, p2]

                for p_to_save in all_poses_to_save:

                    a_rotmat = p_to_save[:-1, :-1]
                    close_to_existing = False
                    for p2_idx, pose2 in enumerate(avail_poses_wf):
                        trans_ = np.linalg.norm(p_to_save[:-1, -1] - pose2[:-1, -1], axis=-1)

                        b_rotmat = pose2[:-1, :-1]
                        qa = R.from_matrix(a_rotmat).as_quat()
                        qb = R.from_matrix(b_rotmat).as_quat()
                        
                        quat_scalar_prod = np.sum(qa * qb)
                        rot_ = 1 - quat_scalar_prod**2

                        if trans_ < 0.02 and rot_ < np.deg2rad(5):
                            close_to_existing = True
                            break
                
                    if not close_to_existing:
                        avail_poses_wf.append(p_to_save)

        elif 'mug' and 'rack':
            raise NotImplementedError('Dont yet have mug/rack compute coverage')
        else:
            raise ValueError('Unrecognized classes')


    ###############################################################
    # Setup visualizations
    viz_flags = [
        # 'voxel_aff',
        # 'pre_parent_crop',
        # 'pre_rot',
        # 'post_rot',
        # 'pre_trans',
        # 'post_trans',
        # 'post_rot_multi',
        # 'post_rot_multi_refine',
        # 'refine_iters',
        # 'viz_attn',
        'refine_recrop',
        # 'final_box',
        'final_pcd',
    ]
    
    ###############################################################
    # Setup all models (scene/affordance, translation and rotation (coarse and refine), feature encoder, success

    if coarse_aff_model is not None:
        coarse_aff_model = coarse_aff_model.eval() 

    if pose_refine_model is not None:
        pose_refine_model = pose_refine_model.eval()
        if hasattr(pose_refine_model, 'eval_sample'):
            pose_refine_model.set_eval_sample(True)
    
    if success_model is not None:
        success_model = success_model.eval()

    child_pcd_original = child_pcd.copy()

    rix1 = np.random.permutation(parent_pcd.shape[0])
    rix2 = np.random.permutation(child_pcd.shape[0])

    ###############################################################
    # Check if we want to use this run to export data for visualization
    export_viz = False
    if 'export_viz' in kwargs:
        if kwargs['export_viz']:
            assert 'mesh_dict' in kwargs, 'Required to pass in "mesh_dict" when using "export_viz"!'
            assert 'export_viz_relative_trans_guess' in kwargs, 'Required to pass in "export_viz_relative_trans_guess" when using "export_viz"!'
            export_viz = True
            import os.path as osp
            export_dir = osp.join(path_util.get_rpdiff_eval_data(), 'rpdiff_multistep_pose_reg_viz_export', kwargs['export_viz_dirname'])
            util.safe_makedirs(export_dir)

            mesh_dict = kwargs['mesh_dict']
            parent_mesh_fname = mesh_dict['parent_file']
            parent_scale = mesh_dict['parent_scale']
            parent_pose_mat = [util.matrix_from_list(val) for val in mesh_dict['parent_pose']]
            parent_tmesh_origin_list = []
            parent_tmesh_posed_list = []
            for pidx, pfn in enumerate(parent_mesh_fname):
                ptmesh_orig = trimesh.load(pfn, process=False).apply_scale(parent_scale[pidx])
                ptmesh_posed = ptmesh_orig.copy().apply_transform(parent_pose_mat[pidx])
                parent_tmesh_origin_list.append(ptmesh_orig)
                parent_tmesh_posed_list.append(ptmesh_posed)

            if len(parent_tmesh_origin_list) > 1:
                parent_tmesh_origin = trimesh.util.concatenate(parent_tmesh_origin_list)
                parent_tmesh_posed = trimesh.util.concatenate(parent_tmesh_posed_list)
            else:
                parent_tmesh_origin = parent_tmesh_origin_list[0]
                parent_tmesh_posed = parent_tmesh_posed_list[0]

            child_mesh_fname = mesh_dict['child_file'][0]
            child_scale = mesh_dict['child_scale'][0]
            child_pose_mat = util.matrix_from_list(mesh_dict['child_pose'][0])
            child_tmesh_origin = trimesh.load(child_mesh_fname).apply_scale(child_scale)
            child_tmesh_posed = child_tmesh_origin.copy().apply_transform(child_pose_mat)
            
            mc_ex = 'scene/export_viz'
            util.meshcat_trimesh_show(mc_vis, f'{mc_ex}/parent_posed', parent_tmesh_posed, opacity=0.3)
            util.meshcat_trimesh_show(mc_vis, f'{mc_ex}/child_posed', child_tmesh_posed, opacity=0.3)
            
            ex_parent_mesh_orig_fn = osp.join(export_dir, 'parent_mesh_orig.obj')
            ex_parent_mesh_posed_fn = osp.join(export_dir, 'parent_mesh_posed.obj')
            ex_child_mesh_orig_fn = osp.join(export_dir, 'child_mesh_orig.obj')
            ex_child_mesh_posed_fn = osp.join(export_dir, 'child_mesh_posed.obj')
            
            parent_tmesh_posed.export(ex_parent_mesh_posed_fn)
            child_tmesh_posed.export(ex_child_mesh_posed_fn)
            parent_tmesh_origin.export(ex_parent_mesh_orig_fn)
            child_tmesh_origin.export(ex_child_mesh_orig_fn)

            np.savez(osp.join(export_dir, 'initial_poses.npz'), parent_pose_mat=parent_pose_mat, child_pose_mat=child_pose_mat)

            parent_pcd_export = copy.deepcopy(parent_pcd)
            child_pcd_export = copy.deepcopy(child_pcd)
            np.savez(osp.join(export_dir, 'pcd.npz'), parent=parent_pcd_export, child=child_pcd_export)

            ex_data_dict = util.AttrDict(dict())
            ex_data_dict.parent_start_pose = np.eye(4)
            # ex_data_dict.child_start_pose = child_pose_mat
            ex_data_dict.child_start_pose = np.eye(4)
            ex_data_dict.child_relative_transform_guess = kwargs['export_viz_relative_trans_guess']

            ex_data_dict.n_sep_opts = init_k_val
            ex_data_dict.n_iters = n_iters
            ex_data_dict.offset = np.array([1.0, 0.0, 0.0])
            ex_data_dict.offset2 = np.array([0.0, 1.5, 0.0])
            ex_data_dict.small_batch_size = 8
            
            ex_init_pos_data = None
            opt_iter_data = []

    ###############################################################
    # Setup voxel affordance scene prediction
    
    # prep the region we will use for checking if final translations are valid (i.e., close enough to the parent)
    parent_pcd_obb = trimesh.PointCloud(parent_pcd).bounding_box_oriented
    util.meshcat_trimesh_show(mc_vis, f'scene/voxel_grid_world/parent_pcd_obb', parent_pcd_obb.to_mesh(), opacity=0.3)

    if np.max(parent_pcd_obb.extents) > scene_scale and False:
        print("here with broken (large) parent oriented bounding box")
        from IPython import embed; embed()
        parent_pcd_mean = np.mean(parent_pcd, axis=0)
        parent_pcd_close = parent_pcd[np.where(np.linalg.norm(parent_pcd - parent_pcd_mean, axis=-1) < 0.2)[0]]
        parent_pcd_obb = trimesh.PointCloud(parent_pcd_close).bounding_box_oriented
        util.meshcat_trimesh_show(mc_vis, f'scene/voxel_grid_world/parent_pcd_obb_close', parent_pcd_obb.to_mesh(), opacity=0.3)

    # run_affordance = not no_parent_crop  # use no_parent_crop flag to indicate if we should run scene affordance or not
    if run_affordance:
        # normalize parent/A point cloud to unit cube
        scene_mean[2] = np.min(parent_pcd[:, 2])
        pscene_pcd = (copy.deepcopy(parent_pcd) - scene_mean ) * scene_scale

        # downsample and prepare input dict
        coarse_aff_mi = {}
        pscene_pcd_torch = torch.from_numpy(pscene_pcd).float().cuda().reshape(1, -1, 3)
        coarse_aff_mi['parent_start_pcd_sn'] = fps_downsample(pscene_pcd_torch, 8192)

        # extract voxel features and make per-voxel affordance prediction
        # k_val = 5
        k_val = init_k_val
        with torch.no_grad():
            # Encode parent voxel grid (scene-normalized)
            parent_feature_grid = coarse_aff_model.encode_scene_fea_grid(coarse_aff_mi['parent_start_pcd_sn'])
            parent_feature_grid = parent_feature_grid['grid'].permute(0, 2, 3, 4, 1)  # features in last dimension

            # predict best voxel position
            voxel_aff_model_output = coarse_aff_model.voxel_affordance(parent_feature_grid)

        # Apply this prediction to the child objects
        vals, top_voxel_inds = torch.topk(voxel_aff_model_output['voxel_affordance'][0], k=k_val)  # B x 1

        # use top-k indices and map them to the grid points
        inds_np = top_voxel_inds.detach().cpu().numpy()
        voxel_pts = grid_pts[inds_np]

        # unnormalize the grid points back to world frame
        world_pts = (voxel_pts / scene_scale) + scene_mean

        sz_base = 1.1/32
        # if viz:
        if 'voxel_aff' in viz_flags:
            # show the top-k voxels as cubes in meshcat
            print(f'Scene scale: {scene_scale}, scene mean: {scene_mean}')
            util.meshcat_pcd_show(mc_vis, pscene_pcd, (255, 0, 0), f'scene/voxel_aff/pscene_pcd_norm', size=MC_SIZE)
            for i, pt in enumerate(voxel_pts):
                box = trimesh.creation.box([sz_base]*3).apply_translation(pt)
                util.meshcat_trimesh_show(mc_vis, f'scene/voxel_grid_raster/{i}', box, opacity=0.3)
            for i, pt in enumerate(world_pts):
                box = trimesh.creation.box([sz_base]*3).apply_translation(pt)
                util.meshcat_trimesh_show(mc_vis, f'scene/voxel_grid_world/{i}', box, opacity=0.3)
        
        print(f'Not checking "validity" of world points based on inside/outshape bounding box')
        # # check the validity of these voxels (valid if inside bounding box of the parent point cloud)
        # valid_world_pts = parent_pcd_obb.contains(world_pts)
        # invalid_world_idx = np.where(np.logical_not(valid_world_pts))[0]

        # # if prediction is invalid, set the world frame value as the parent point cloud mean
        # world_pts[invalid_world_idx] = np.mean(parent_pcd, axis=0)
    else:
        # if we're not running voxel affordance, just leave the initial translation as the child mean

        k_val = init_k_val
        if init_parent_mean:
            # k_val = 1
            world_pts = np.mean(parent_pcd, axis=0).reshape(1, 3)
            world_pts = np.tile(world_pts, (k_val, 1))
        else:
            sz_base = 1.1/32
            world_pts = parent_pcd_obb.sample_volume(k_val).reshape(-1, 3)

            for i, pt in enumerate(world_pts):
                box = trimesh.creation.box([sz_base]*3).apply_translation(pt)
                util.meshcat_trimesh_show(mc_vis, f'scene/rnd_voxel_grid_world/{i}', box, opacity=0.3)

    if export_viz:
        ex_init_pos_data = world_pts
        ex_data_dict.init_pos_data = ex_init_pos_data

    ###############################################################
    # Setup parent point cloud cropping

    # build the size of the bounding box we will use
    child_mean = np.mean(child_pcd, axis=0)
    child_pcd_scaled = child_pcd - child_mean
    child_pcd_scaled *= 1.25
    child_pcd_scaled = child_pcd_scaled + child_mean
    try:
        child_bb = trimesh.PointCloud(child_pcd_scaled).bounding_box.to_mesh()
    except Exception as e:
        print(f'[Getting bounding box for child pcd] Exception: {e}')
        from IPython import embed; embed()
        print(f'Returning identity to exit out')
        return np.eye(4)
    # max_length = np.max(child_bb.extents) / 2

    if variable_size_crop:
        max_crop_length = np.linalg.norm([parent_pcd.max(0) - parent_pcd.min(0)], axis=1)[0] / np.sqrt(2)
        max_length = max_crop_length
    else:
        max_length = 0.18
        print(f'Using fixed value of crop max length: {max_length} (not using child bounding box)')
    
    # make separate crops of the parent point cloud
    M = k_val
    N_crop = 2048
    N_parent = parent_pcd.shape[0]

    # convert original (unnormalized) parent/A object and initial translations to torch
    pscene_pcd_t = torch.from_numpy(parent_pcd).float().cuda()  # N x 3
    world_pts_t = torch.from_numpy(world_pts).float().cuda()  # N x 3
 
    # prepare parent pcd mean for future use
    full_parent_pcd_mean = torch.mean(pscene_pcd_t, axis=0).reshape(1, 3) # 1 x 3

    ###############################################################
    # Create crops of parent point cloud using top-k voxel coordinates

    # prepare empty array of cropped point clouds (TODO: allow to be non-equal # of points)
    parent_cropped_pcds = torch.empty((M, N_crop, 3)).float().cuda()  # M x N_crop x 3
    if no_parent_crop:
         parent_cropped_pcds = fps_downsample(pscene_pcd_t.reshape(1, -1, 3), N_crop).repeat((M, 1, 1))
         print('Got full parent')
    else:
        for i in range(M):
            # get the 3D coordinate of this voxel prediction
            voxel_3d_pt = world_pts_t[i].reshape(1, 3)  # 1 x 3

            # get high and low bounds, and repeat to prepare for less/greater comparison per-point
            voxel_high = voxel_3d_pt + max_length  # 1 x 3
            voxel_low = voxel_3d_pt - max_length  # 1 x 3
            high, low = voxel_high.repeat((N_parent, 1)), voxel_low.repeat((N_parent, 1))  # N_parent x 3, N_parent x 3

            # per-element bounds check, .all(-1) called on last dimension to ensure full x,y,z are within bounds
            parent_in_crop_bounds = torch.logical_and(pscene_pcd_t < high, pscene_pcd_t > low).all(-1)
            parent_crop_idx = torch.where(parent_in_crop_bounds)[0]

            # grab the indices corresponding to points in these bounds
            parent_crop_pcd = pscene_pcd_t[parent_crop_idx]

            # hacky way to ensure we have enough points in our crop
            rix = torch.randperm(parent_crop_pcd.shape[0])
            if parent_crop_pcd.shape[0] == 0:
                print('!! Parent crop had zero points !!')
                # hacky way to set the cropped point cloud if there are no points in the crop
                # (means this voxel prediction was bad)
                rix2 = torch.randperm(pscene_pcd_t.shape[0])
                parent_cropped_pcds[i] = pscene_pcd_t[rix2[:N_crop]]
                continue
                
            if parent_crop_pcd.shape[0] < N_crop:
                while True:
                    # if not enough points in our crop, make copies of random subsets of points and concatenate
                    # print(f'Parent pcd crop shape: {parent_crop_pcd.shape[0]}')
                    parent_crop_pcd = torch.cat((parent_crop_pcd, parent_crop_pcd[rix[:100]]), dim=0)
                    if parent_crop_pcd.shape[0] >= N_crop:
                        break
                    rix = torch.randperm(parent_crop_pcd.shape[0])
            else:
                parent_crop_pcd = parent_crop_pcd[rix[:N_crop]]

            # parent_cropped_pcds[i] = parent_crop_pcd
            parent_cropped_pcds[i] = parent_crop_pcd[:N_crop]


    ###############################################################
    # Prepare child object point cloud for pose (rotation) prediction

    # prepare child pcd torch tensor and downsample child pcd to right number of points
    child_pcd_t = torch.from_numpy(child_pcd).float().cuda()
    child_pcd_ds = fps_downsample(child_pcd_t.reshape(1, -1, 3), N_crop)  # 1 x N_crop x 3

    # make copies of child point cloud and child pcd mean (points + batch) + initial translation points (points) 
    child_pcd_batch = child_pcd_ds.repeat((M, 1, 1))  # M x N_child x 3  (N_child = N_crop)
    # child_mean_batch = child_pcd_t.mean(0).reshape(1, 1, 3).repeat((1, child_pcd_ds.shape[0], 1))  # 1 x N_child x 3
    child_mean_batch = child_pcd_t.mean(0).reshape(1, 1, 3).repeat((M, child_pcd_ds.shape[0], 1))  # M x N_child x 3
    world_trans_batch = world_pts_t.reshape(-1, 1, 3).repeat((1, child_pcd_ds.shape[0], 1))[:M]  # M x N_child x 3

    # prepare centered point clouds and relative translation to apply for initial translation from voxel positions
    # move to the location of parent crops (voxel positions)
    delta_trans_batch = world_trans_batch - child_mean_batch  # M x N_child x 3
    child_pcd_cent_batch = child_pcd_batch - child_mean_batch  # M x N_child x 3

    # sample a batch of rotations and apply to batch
    if init_orig_ori:
        rand_mat_init = torch.eye(4).reshape(-1, 4, 4).repeat((M, 1, 1)).float().cuda()
    else:
        rand_rot_idx = np.random.randint(rot_grid.shape[0], size=M)
        rand_rot_init = matrix_to_axis_angle(torch.from_numpy(rot_grid[rand_rot_idx])).float()
        rand_mat_init = angle_axis_to_rotation_matrix(rand_rot_init)
        rand_mat_init = rand_mat_init.squeeze().float().cuda()
    
    # apply voxel location offsets (after un-centering with mean)
    child_pcd_rot_batch = child_pcd_init_batch = transform_pcd_torch(child_pcd_cent_batch, rand_mat_init)  # M x N_child x 3
    child_pcd_trans_batch = child_pcd_rot_batch + delta_trans_batch + child_mean_batch # M x N_child x 3

    # if viz:
    if 'pre_parent_crop' in viz_flags:
        for ii in range(M):
            child_pcd_trans_viz = child_pcd_trans_batch[ii].detach().cpu().numpy().squeeze()
            parent_crop_pcd_trans_viz = parent_cropped_pcds[ii].detach().cpu().numpy().squeeze()
            util.meshcat_pcd_show(mc_vis, child_pcd_trans_viz, (255, 255, 0), f'scene/infer/child_pcd_trans_batch_{ii}', size=MC_SIZE)
            util.meshcat_pcd_show(mc_vis, parent_crop_pcd_trans_viz, (255, 0, 255), f'scene/infer/parent_pcd_trans_batch_{ii}', size=MC_SIZE)

    ###############################################################
    # Pass to point cloud encoders in case we want extra per-point features

    # get centered + cropped parent point clouds for feature encoding (copy mean and subtract off from cropped parent points)
    parent_crop_mean_batch = torch.mean(parent_cropped_pcds, axis=1).reshape(-1, 1, 3).repeat((1, N_crop, 1))
    parent_crop_cent_batch = parent_cropped_pcds - parent_crop_mean_batch  # M x N_crop x 3

    ###############################################################
    # Prepare pose prediction policy inputs

    # prepare inputs to the policy
    policy_mi = {}

    # CENTERED point clouds (both M x N_crop x 3)
    policy_mi['parent_start_pcd'] = parent_crop_cent_batch
    policy_mi['child_start_pcd'] = child_pcd_init_batch

    # means for uncentering/providing to policy in a shared world frame
    policy_mi['parent_start_pcd_mean'] = torch.mean(parent_cropped_pcds, axis=1).reshape(-1, 3)
    policy_mi['child_start_pcd_mean'] = torch.mean(child_pcd_trans_batch, axis=1).reshape(-1, 3)

    # if viz:
    rot_debug_idx = 0
    if 'pre_rot' in viz_flags:
        # visualize policy model input, directly before passing into rotation 
        # rot_debug_idx = 0
        offset = np.array([1.0, 0, 0])
        ppcd_viz = policy_mi['parent_start_pcd'][rot_debug_idx].detach().cpu().numpy()
        cpcd_viz = policy_mi['child_start_pcd'][rot_debug_idx].detach().cpu().numpy() 
        ppcd_mean_viz = policy_mi['parent_start_pcd_mean'][rot_debug_idx].detach().cpu().numpy()
        cpcd_mean_viz = policy_mi['child_start_pcd_mean'][rot_debug_idx].detach().cpu().numpy()
        util.meshcat_pcd_show(mc_vis, ppcd_viz + offset, (255, 0, 0), f'scene/policy_mi_pre_rot/parent_start_pcd', size=MC_SIZE)
        util.meshcat_pcd_show(mc_vis, cpcd_viz + offset, (0, 0, 255), f'scene/policy_mi_pre_rot/child_start_pcd', size=MC_SIZE)
        util.meshcat_pcd_show(mc_vis, ppcd_viz + ppcd_mean_viz + offset, (255, 0, 0), f'scene/policy_mi_pre_rot/parent_start_pcd_uncent', size=MC_SIZE)
        util.meshcat_pcd_show(mc_vis, cpcd_viz + cpcd_mean_viz + offset, (0, 0, 255), f'scene/policy_mi_pre_rot/child_start_pcd_uncent', size=MC_SIZE)

    ###############################################################
    # Predict set of rotations to apply about centered child point cloud
    # torch.ones((policy_mi['parent_start_pcd'].shape[0], 1)).float().cuda() * n_iters
    # timestep = torch.ones((policy_mi['parent_start_pcd'].shape[0], 1)).float().cuda() * (n_iters - 44)
    timestep = torch.ones((policy_mi['parent_start_pcd'].shape[0], 1)).float().cuda() * (n_iters)
    if util.hasattr_and_true(pose_refine_model, 'pos_emb'):
        # print(f'Timestep: {timestep}')
        # timestep_emb = pose_refine_model.pos_emb(timestep)
        timestep_emb = pose_refine_model.pos_emb(torch.clip(timestep, 0, 4))
        # timestep_emb = pose_refine_model.pos_emb(torch.clip(timestep, 0, n_diff_steps))
        policy_mi['timestep_emb'] = timestep_emb

    # get rotation output prediction
    with torch.no_grad():
        rot_model_output_raw = pose_refine_model(policy_mi)  # M x N_queries x 3 x 3
            
    # if viz:
    if 'post_rot' in viz_flags:
        # visualize just before applying output rotation
        util.meshcat_pcd_show(mc_vis, child_pcd_rot_batch[0].detach().cpu().numpy(), (255, 0, 255), f'scene/viz/child_pcd_pre_rot', size=MC_SIZE)

    ###############################################################
    # Apply set of predicted rotations to point clouds
    
    # reshape rotations and batch of point clouds to apply rotations
    max_rot_idx = torch.zeros(rot_model_output_raw['rot_mat'].shape[0]).long().cuda()
    rot_mat_queries = torch.gather(rot_model_output_raw['rot_mat'], dim=1, index=max_rot_idx[:, None, None, None].repeat(1, 1, 3, 3)).reshape(M, 1, 3, 3)

    N_queries = rot_mat_queries.shape[1]
    post_rot_batch = N_queries * M  # get the total "batch size" being passed to translation prediction (original batch * number of rotation queries)

    # make copies of the CENTERED child point cloud for each query rotation
    child_pcd_rot_qb = child_pcd_rot_batch[:, None, :, :].repeat((1, N_queries, 1, 1))

    # apply each of the predicted rotations across this batch
    child_pcd_post_rot_qb = torch.matmul(rot_mat_queries, child_pcd_rot_qb.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # M x N_queries x N_child x 3

    # if viz:
    if 'post_rot' in viz_flags:

        # visualize just after applying output rotation
        util.meshcat_pcd_show(mc_vis, child_pcd_post_rot_qb[0, 0].detach().cpu().numpy(), (0, 255, 255), f'scene/viz/child_pcd_post_rot', size=MC_SIZE)

    sm = torch.nn.Softmax(dim=0)

    ###############################################################
    # **IMPORTANT** keep track of indices corresponding to each transformed output child point cloud 
    # (so we know what translation/rotation to apply at the end, after scoring)

    # voxel indices -- start with just the original batch (k_val), make copies for each query ([0, 0, 0..., 1, 1, 1..., 2, 2, 2..., ], etc.)
    voxel_idx = torch.arange(M).reshape(-1, 1).repeat((1, N_queries)).reshape(post_rot_batch, -1).cuda()  # M*N_queries x 1

    # rotation indices -- go in order from 0 to total number of rotations ([0, 1, 2, 3..., M*N_queries-1])
    # TODO: This is unused?
    rot_idx = torch.arange(post_rot_batch).reshape(-1, 1)

    ###############################################################
    # Prepare rotated child point clouds for input to translation prediction

    # have batch of rotated child point clouds, get translation output prediction
    child_pcd_post_rot_qb = child_pcd_post_rot_qb.reshape(post_rot_batch, -1, 3)  # M*N_queries x N_crop x 3

    # if viz:
    if 'post_rot' in viz_flags:
        # visualize after applying output rotation and reshaping into new batch
        util.meshcat_pcd_show(mc_vis, child_pcd_post_rot_qb[0].detach().cpu().numpy(), (0, 255, 0), f'scene/viz/child_pcd_post_rot2', size=MC_SIZE)

    # re-encode the rotated + CENTERED child points cloud
    policy_mi['child_start_pcd'] = child_pcd_post_rot_qb

    # make copies of the parent point clouds and parent/child means for extra elements along new batch dimension
    # (careful here)
    trans_p_feat_pcd = parent_crop_cent_batch[:, None, :, :]  # add extra dimension, M x 1 x N_crop x 3
    trans_p_feat_pcd = trans_p_feat_pcd.repeat((1, N_queries, 1, 1))  # repeat N_queries, M x N_queries x N_crop x 3
    trans_p_feat_pcd = trans_p_feat_pcd.reshape(post_rot_batch, -1, 3)  # reshape, M*N_queries x N_crop x 3
    policy_mi['parent_start_pcd'] = trans_p_feat_pcd

    trans_p_pcd_mean = policy_mi['parent_start_pcd_mean'][:, None, :]  # add extra dim, M x 1 x 3
    trans_p_pcd_mean = trans_p_pcd_mean.repeat((1, N_queries, 1))  # repeat N_queries, M x N_queries x 3
    trans_p_pcd_mean = trans_p_pcd_mean.reshape(post_rot_batch, -1)  # reshape, M*N_queries x 3
    policy_mi['parent_start_pcd_mean'] = trans_p_pcd_mean

    trans_c_pcd_mean = policy_mi['child_start_pcd_mean'][:, None, :]
    trans_c_pcd_mean = trans_c_pcd_mean.repeat((1, N_queries, 1))
    trans_c_pcd_mean = trans_c_pcd_mean.reshape(post_rot_batch, -1)
    policy_mi['child_start_pcd_mean'] = trans_c_pcd_mean

    # if viz: 
    if 'pre_trans' in viz_flags: 
        base_debug_idx = rot_debug_idx*N_queries
        for j in range(4):
            debug_idx = base_debug_idx + j
            offset = np.array([1.0, 0, 0])
            ppcd_viz = policy_mi['parent_start_pcd'][debug_idx].detach().cpu().numpy()
            cpcd_viz = policy_mi['child_start_pcd'][debug_idx].detach().cpu().numpy() 
            ppcd_mean_viz = policy_mi['parent_start_pcd_mean'][debug_idx].detach().cpu().numpy()
            cpcd_mean_viz = policy_mi['child_start_pcd_mean'][debug_idx].detach().cpu().numpy()
            util.meshcat_pcd_show(mc_vis, ppcd_viz + offset, (255, 0, 0), f'scene/policy_mi_pre_trans/parent_start_pcd/{j}', size=MC_SIZE)
            util.meshcat_pcd_show(mc_vis, cpcd_viz + offset, (0, 0, 255), f'scene/policy_mi_pre_trans/child_start_pcd/{j}', size=MC_SIZE)
            util.meshcat_pcd_show(mc_vis, ppcd_viz + ppcd_mean_viz + offset, (255, 0, 0), f'scene/policy_mi_pre_trans/parent_start_pcd_uncent/{j}', size=MC_SIZE)
            util.meshcat_pcd_show(mc_vis, cpcd_viz + cpcd_mean_viz + offset, (0, 0, 255), f'scene/policy_mi_pre_trans/child_start_pcd_uncent/{j}', size=MC_SIZE)


    ###############################################################
    # Predict set of translations

    # get output for translation, and combine outputs with rotation prediction
    with torch.no_grad():
        trans_model_output_raw = pose_refine_model(policy_mi)  # M x N_queries x 3

    # prepare output translations for adding to child pcd (for viz) + child pcd mean (for output) (careful -- reshaping)
    # # M*N_queries x N_queris x 3 -> M*N_queries x N_queries x 1 x 3 -> M*N_queries x N_queries x N_crop x 3
    # trans_queries = trans_model_output_raw['trans'].reshape(post_rot_batch, N_queries, 1, 3).repeat((1, 1, N_crop, 1))
    max_trans_idx = torch.zeros(trans_model_output_raw['trans'].shape[0]).long().cuda()
    trans_queries = torch.gather(trans_model_output_raw['trans'], dim=1, index=max_trans_idx[:, None, None].repeat(1, 1, 3)).reshape(post_rot_batch, N_queries, 1, 3).repeat((1, 1, N_crop, 1))
 
    # M*N_queries x 3 -> M*N_queries x 1 x 1 x 3 -> M*N_queries x N_queries x N_crop x 3
    post_trans_child_pcd_mean = policy_mi['child_start_pcd_mean'][:, None, None, :].repeat((1, N_queries, N_crop, 1))

    # M*N_queries x N_crop x 3 -> M*N_queries x N_queries x N_crop x 3
    child_pcd_trans_qb = child_pcd_post_rot_qb[:, None, :, :].repeat((1, N_queries, 1, 1))

    # obtain final UNCENTERED child point cloud via add it all together (M*N_queries x N_queries x N_crop x 3)
    child_pcd_post_trans_qb = child_pcd_trans_qb + post_trans_child_pcd_mean + trans_queries

    ###############################################################
    # **IMPORTANT** keep track of indices corresponding to each transformed output child point cloud 
    # (so we know what translation/rotation to apply at the end, after scoring)

    # voxel indices -- take previous voxel inds, repeat them along new query dimension, and reshape to new new batch dimension
    voxel_idx_final = voxel_idx.repeat((1, N_queries)).reshape(post_rot_batch*N_queries, -1)  # M*N_queries**2 x 1

    # rotation indices -- indices are full range ([0, 1, 2, ..., 59, 60, 61, ..., M*N_queries**2 - 2, M*N_queries**2 - 1]) 
    # and we make copies of the original rotations so that [0, 1, 2, ..., N_queries] corresponds to the SAME rotation
    rot_idx_final = torch.arange(post_rot_batch*N_queries).reshape(-1, 1).cuda()  # M*N_queries**2 x 1
    rot_q_final = rot_mat_queries.reshape(post_rot_batch, 3, 3)
    rot_q_final = rot_q_final[:, None, :, :]
    rot_q_final = rot_q_final.repeat((1, N_queries, 1, 1))
    rot_q_final = rot_q_final.reshape(post_rot_batch*N_queries, 3, 3)
    rot_mat_queries_final = rot_q_final

    # translation indices -- indices are full range ([0, 1, 2, ..., M*N_queries**2 - 2, M*N_queries**2 - 1]), just reshape output trans
    trans_idx_final = torch.arange(post_rot_batch*N_queries).reshape(-1, 1).cuda()  # M*N_queries**2 x 1
    trans_queries_final = trans_queries[:, :, 0].reshape(post_rot_batch*N_queries, -1)

    if export_viz:
        # collect the initial set of predicted poses
        export_all_idx = np.arange(trans_queries_final.shape[0]).astype(int)
        voxel_topk = voxel_idx_final[export_all_idx].squeeze()
        rot_topk = rot_idx_final[export_all_idx].squeeze()
        trans_topk = trans_idx_final[export_all_idx].squeeze()

        # voxel - index into the world points
        out_export_voxel_trans = world_pts_t[voxel_topk].reshape(-1, 3)

        # initial random rotation - use the VOXEL inds (these correspond to our original batch size)
        out_export_rot_init = rand_mat_init.reshape(-1, 4, 4)[voxel_topk].reshape(-1, 4, 4)

        # rotation and translation - index into the final queries
        out_export_rot = rot_mat_queries_final[rot_topk].reshape(-1, 3, 3)
        out_export_trans = trans_queries_final[trans_topk].reshape(-1, 3)

        # combine trans all together with voxel trans and trans queries
        out_export_trans_full = out_export_voxel_trans + out_export_trans
        
        out_export_tmat_list = []
        for ii in export_all_idx:
            # save the translation component
            tmat_export = np.eye(4); tmat_export[:-1, -1] = out_export_trans_full[ii].detach().cpu().numpy().squeeze()

            # get the rotation and the initial rotation
            tmat_export[:-1, :-1] = out_export_rot[ii].detach().cpu().numpy().squeeze()
            tmat_export_init = out_export_rot_init[ii].detach().cpu().numpy().squeeze()

            # combine the initial rotation with the final rotation
            tmat_export_full = np.matmul(tmat_export, tmat_export_init)

            tf_cent = np.eye(4); tf_cent[:-1, -1] = -1.0 * np.mean(child_pcd_original, axis=0)
            out_export_tf = np.matmul(tmat_export_full, tf_cent)
            # out_export_tf = np.matmul(tmat_export_full_list[0], tf_cent)  # get the TOP scoring one

            export_iter_data = {}
            export_iter_data['parent_pose_mat'] = np.eye(4)
            export_iter_data['child_pose_mat'] = out_export_tf

            ppcd_viz = policy_mi['parent_start_pcd'][ii].detach().cpu().numpy()
            ppcd_mean_viz = policy_mi['parent_start_pcd_mean'][ii].detach().cpu().numpy()
            export_iter_data['parent_pcd_cropped'] = ppcd_viz + ppcd_mean_viz

            out_export_tmat_list.append(export_iter_data)

        opt_iter_data.append(out_export_tmat_list)


    ###############################################################
    # Prepare inputs for success classifier/refinement loop

    # reshape translations and batch of point clouds to apply translations
    child_pcd_to_sc = child_pcd_post_trans_qb.reshape(post_rot_batch*N_queries, -1, 3)  # M*N_queries**2 x N_crop x 3
    child_pcd_to_sc_mean = torch.mean(child_pcd_to_sc, axis=1)  # M*N_queries**2 x 3

    # get features for translated + rotated + CENTERED point cloud 
    policy_mi['child_start_pcd'] = child_pcd_to_sc - child_pcd_to_sc_mean[:, None, :].repeat((1, N_crop, 1))
    policy_mi['child_start_pcd_mean'] = child_pcd_to_sc_mean  # new mean after applying translation

    # copy + reshape parent point cloud for success classifier/refinement
    p_start_pcd = policy_mi['parent_start_pcd'][:, None, :, :]  # extra dim
    p_start_pcd = p_start_pcd.repeat((1, N_queries, 1, 1))  # repeat, M*N_queries x N_queries x N_crop x 3
    p_start_pcd = p_start_pcd.reshape(post_rot_batch*N_queries, -1, 3)  # reshape, M*N_queries**2 x 3
    policy_mi['parent_start_pcd'] = p_start_pcd

    p_start_pcd_mean = policy_mi['parent_start_pcd_mean'][:, None, :]  # extra dim
    p_start_pcd_mean = p_start_pcd_mean.repeat((1, N_queries, 1))  # repeat, M*N_queries x N_queries x 3
    p_start_pcd_mean = p_start_pcd_mean.reshape(post_rot_batch*N_queries, -1)  # reshape, M*N_queries**2 x 3
    policy_mi['parent_start_pcd_mean'] = p_start_pcd_mean
    
    # if viz:
    if 'post_trans' in viz_flags:
        base_debug_idx = rot_debug_idx*(N_queries**2) 
        for j in range(4):
            for jj in range(4):
                jj_val = N_queries*j + jj
                debug_idx = base_debug_idx + jj_val
                offset = np.array([1.0, 0, 0])
                ppcd_viz = policy_mi['parent_start_pcd'][debug_idx].detach().cpu().numpy()
                cpcd_viz = policy_mi['child_start_pcd'][debug_idx].detach().cpu().numpy() 
                ppcd_mean_viz = policy_mi['parent_start_pcd_mean'][debug_idx].detach().cpu().numpy()
                cpcd_mean_viz = policy_mi['child_start_pcd_mean'][debug_idx].detach().cpu().numpy()
                util.meshcat_pcd_show(mc_vis, ppcd_viz + offset, (255, 0, 0), f'scene/policy_mi_post_trans/parent_start_pcd/{jj_val}', size=MC_SIZE)
                util.meshcat_pcd_show(mc_vis, cpcd_viz + offset, (0, 0, 255), f'scene/policy_mi_post_trans/child_start_pcd/{jj_val}', size=MC_SIZE)
                util.meshcat_pcd_show(mc_vis, ppcd_viz + ppcd_mean_viz + offset, (255, 0, 0), f'scene/policy_mi_post_trans/parent_start_pcd_uncent/{jj_val}', size=MC_SIZE)
                util.meshcat_pcd_show(mc_vis, cpcd_viz + cpcd_mean_viz + offset, (0, 0, 255), f'scene/policy_mi_post_trans/child_start_pcd_uncent/{jj_val}', size=MC_SIZE)


    # break into smaller batches
    sm_bs = 8
    success_out_list = []

    offset = np.array([1.0, 0, 0])
    offset2 = np.array([0.0, 1.5, 0])

    # downsample full parent point cloud and prepare parent pcd mean for future use
    full_parent_pcd_ds = fps_downsample(pscene_pcd_t.reshape(1, -1, 3), N_crop)  # 1 x N_crop x 3
    ppcd_viz_full = full_parent_pcd_ds[0].detach().cpu().numpy()
    ppcd_mean_viz_full = full_parent_pcd_mean[0].detach().cpu().numpy()

    if 'refine_recrop' in viz_flags:
        ridx = 0
        ridx2 = 0
        while True:
            if ridx >= policy_mi['child_start_pcd'].shape[0]:
                break
            ridx2 += 1
            for ri in range(sm_bs):
                ridx += 1
                util.meshcat_pcd_show(
                    mc_vis, 
                    ppcd_viz_full + offset*ridx2  + offset2*ri - (offset2 * sm_bs / 2.0), 
                    (0, 255, 0), 
                    f'scene/policy_mi_post_trans/parent_start_pcd_uncent_full_{ri}_{ridx2}', 
                    size=MC_SIZE*1.5)

    # make a copy of the original parent point cloud tensor for re-cropping (same size as small batch)
    pscene_pcd_t_to_crop = pscene_pcd_t[None, :, :].repeat((sm_bs, 1, 1))

    # clone policy model input tensors
    refine_policy_mi = {k: v.clone() for k, v in policy_mi.items()}

    step_size = 1.0
    lin_step_size = 1.0

    refine_recrop_ii_viz = 0

    n_diff_steps = n_iters
    if util.hasattr_and_not_none(pose_refine_model, 'pos_emb_max_timestep'):
        n_diff_steps = pose_refine_model.pos_emb_max_timestep
        print(f'Adjustment of timesteps based on max timestep')
        print(f'n_diff_steps (pose_refine_model.pos_emb_max_timestep): {n_diff_steps}')
    else:
        log_warn(f'Did NOT see the model with a "pos_emb_max_timestep" parameter! Will not do any timestep clipping...')
        log_warn(f'MANUALLY SETTING TO 5 BECAUSE WE KNOW THIS IS WHAT THE MODEL USED!!!')
        log_warn(f'NEXT TIME, MAKE SURE TO SET "MAX_TIMESTEP" KEYWORD ARG IN TRAINING CONFIG FILE!!!')
        print(f'pose_refine_model.pos_emb_max_timestep: {pose_refine_model.pos_emb_max_timestep}')
        pose_refine_model.pos_emb_max_timestep = 5
        n_diff_steps = pose_refine_model.pos_emb_max_timestep
        print(f'n_diff_steps (pose_refine_model.pos_emb_max_timestep): {n_diff_steps}')
    delta_steps = n_iters - n_diff_steps
    print(f'delta_steps: {delta_steps}')

    if add_per_iter_noise:
        aa_noise_angle = per_iter_noise_kwargs['rot']['angle_deg']
        aa_noise_rate = per_iter_noise_kwargs['rot']['rate']

        trans_noise_dist = per_iter_noise_kwargs['trans']['trans_dist']
        trans_noise_rate = per_iter_noise_kwargs['trans']['rate']
    
    timestep_emb_decay_factor = np.clip(timestep_emb_decay_factor, a_min=1, a_max=None)
    factor = timestep_emb_decay_factor**np.arange(1, 1 + n_diff_steps)

    nom_per_temb_steps = np.ceil((factor / factor.sum()) * n_iters).astype(int)

    scale_down_factor = n_iters / nom_per_temb_steps.sum()
    nom_per_temb_steps = np.ceil(scale_down_factor * nom_per_temb_steps).astype(int)[::-1]
    delta_zero_steps = nom_per_temb_steps.sum() - n_iters
    nom_per_temb_steps[0] -= delta_zero_steps
    per_temb_steps = nom_per_temb_steps[::-1]

    iter_to_temb = []
    for ii in range(n_diff_steps):
        for jj in range(per_temb_steps[ii]):
            iter_to_temb.append(np.clip(n_diff_steps - ii - 1, a_min=0, a_max=None))

    print(f'\n\nPer timestep iters: {per_temb_steps.tolist()}\n\n')
    time.sleep(2.0)

    with torch.no_grad():
        for ii in range(n_iters):

            if 'refine_iters' in viz_flags:
                base_debug_idx = rot_debug_idx*(N_queries**2) 
                for j in range(12):
                    debug_idx = j*N_queries
                    jj_val = j

                    ppcd_viz = refine_policy_mi['parent_start_pcd'][debug_idx].detach().cpu().numpy()
                    cpcd_viz = refine_policy_mi['child_start_pcd'][debug_idx].detach().cpu().numpy() 
                    ppcd_mean_viz = refine_policy_mi['parent_start_pcd_mean'][debug_idx].detach().cpu().numpy()

                    cpcd_mean_viz = refine_policy_mi['child_start_pcd_mean'][debug_idx].detach().cpu().numpy()
                    util.meshcat_pcd_show(
                        mc_vis, 
                        ppcd_viz + offset, 
                        (255, 0, 0),
                        f'scene/policy_mi_post_trans/parent_start_pcd/{jj_val}', size=MC_SIZE)
                    util.meshcat_pcd_show(
                        mc_vis, 
                        cpcd_viz + offset, 
                        (0, 0, 255), 
                        f'scene/policy_mi_post_trans/child_start_pcd/{jj_val}', size=MC_SIZE)
                    util.meshcat_pcd_show(
                        mc_vis, 
                        ppcd_viz + ppcd_mean_viz + offset, 
                        (255, 0, 0), 
                        f'scene/policy_mi_post_trans/parent_start_pcd_uncent/{jj_val}', size=MC_SIZE)
                    util.meshcat_pcd_show(
                        mc_vis, 
                        cpcd_viz + cpcd_mean_viz + offset, 
                        (0, 0, 255), 
                        f'scene/policy_mi_post_trans/child_start_pcd_uncent/{jj_val}', size=MC_SIZE)

            idx = 0  # go through idx:idx+sm_bs on each round, and increment idx
            print(f'Round {ii} out of {n_iters} rounds of refinement')
            if refine_anneal:
                step_size = step_size * np.exp(-1.0 * ii / n_iters)
                lin_step_size = lin_step_size * 1.0 * (n_iters - ii) / n_iters
            else:
                step_size = 1.0
                lin_step_size = 1.0
            # print(f'Step: {step_size}, Lin: {lin_step_size}')
            ref_idx2 = 0

            # print(f'Step size: {step_size}')
            while True:
                if idx >= policy_mi['child_start_pcd'].shape[0]:
                    break

                ###############################################################
                # Grab our "small batch" for re-cropping + encoding parent, re-encoding child, and getting policy preds
                
                small_policy_mi = {k: v[idx:idx+sm_bs] for k, v in refine_policy_mi.items()}
                small_child_pcd_mean = refine_policy_mi['child_start_pcd_mean'][idx:idx+sm_bs]

                sm_bs_iter = small_policy_mi['child_start_pcd'].shape[0]

                ###############################################################
                # Re-crop for this "small batch"

                no_crop_flag = False
                if idx == 0 and no_crop_flag:
                    print(f'USING NO CROP FLAG FOR RECROPPING PARENT')
                if no_parent_crop or no_crop_flag:
                    pass
                else:
                    if variable_size_crop:
                        min_crop_length = 0.18  # todo - we should make this adjustable in config
                        max_crop_length = np.linalg.norm([parent_pcd.max(0) - parent_pcd.min(0)], axis=1)[0] / np.sqrt(2)
                        max_length = max_crop_length
                        
                        temb_this_step = iter_to_temb[ii]
                        delta_crop_length = temb_this_step * (max_crop_length - min_crop_length) / n_diff_steps
                        max_length = min_crop_length + delta_crop_length
                        # print(f'Crop max length: {max_length} (temb this step: {temb_this_step})')

                    # get new high/low bounds based on current mean of chid pcd
                    high_b = small_policy_mi['child_start_pcd_mean'] + max_length
                    low_b = small_policy_mi['child_start_pcd_mean'] - max_length

                    # reshape to get values for each point in parent point cloud
                    high_b_rep = high_b[:, None, :].repeat((1, N_parent, 1))
                    low_b_rep = low_b[:, None, :].repeat((1, N_parent, 1))
                    
                    # check the batched parent point cloud for being inside these crop bounds
                    below = (pscene_pcd_t_to_crop[:sm_bs_iter] < high_b_rep)
                    above = (pscene_pcd_t_to_crop[:sm_bs_iter] > low_b_rep)

                    crop_idx = torch.logical_and(above, below).all(-1)  # sm_bs x N_parent
                    parent_crop_sb = torch.empty((sm_bs_iter, N_crop, 3)).float().cuda()

                    # loop through each element of the batch and grab the indices corresponding to the crop
                    # TODO: see if we can do this without loops? not sure though, need different # of inds
                    # per dimension... 
                    for jj in range(sm_bs_iter):
                        parent_crop_pcd = pscene_pcd_t[torch.where(crop_idx[jj])[0]]

                        if parent_crop_pcd.shape[0] == 0:
                            print('!! Parent crop had zero points !!')
                            rix2 = torch.randperm(pscene_pcd_t.shape[0])
                            parent_crop_sb[jj] = pscene_pcd_t[rix2[:N_crop]]
                            continue
                            
                        rix = torch.randperm(parent_crop_pcd.shape[0])
                        if parent_crop_pcd.shape[0] < N_crop:
                            while True:
                                # print(f'Parent pcd crop shape: {parent_crop_pcd.shape[0]}')
                                parent_crop_pcd = torch.cat((parent_crop_pcd, parent_crop_pcd[rix[:100]]), dim=0)
                                if parent_crop_pcd.shape[0] >= N_crop:
                                    break
                                rix = torch.randperm(parent_crop_pcd.shape[0])
                        else:
                            parent_crop_pcd = parent_crop_pcd[rix[:N_crop]]

                        # save the cropped parent points
                        parent_crop_sb[jj] = parent_crop_pcd[:N_crop]
                        
                    # get the mean of these new cropped parent points
                    parent_crop_sb_mean = torch.mean(parent_crop_sb, axis=1)

                    # use the CENTERED cropped parent points in new input for feat encoding and policy prediction
                    small_policy_mi['parent_start_pcd'] = parent_crop_sb - parent_crop_sb_mean[:, None, :].repeat((1, N_crop, 1))
                    small_policy_mi['parent_start_pcd_mean'] = parent_crop_sb_mean  # sm_bs x 3
                
                if 'refine_recrop' in viz_flags:
                    ref_idx2 += 1
                    for c_idx in range(sm_bs_iter):

                        new_p_crop = small_policy_mi['parent_start_pcd_mean'][c_idx].detach().cpu().numpy() + small_policy_mi['parent_start_pcd'][c_idx].detach().cpu().numpy()
                        
                        if False:
                        # if True:
                            util.meshcat_pcd_show(
                                mc_vis, 
                                offset + new_p_crop,
                                (0, 255, 0), 
                                f'scene/new_cropped_parent/pcd_{c_idx}', size=MC_SIZE) 

                            new_c_crop = small_policy_mi['child_start_pcd_mean'][c_idx].detach().cpu().numpy() + small_policy_mi['child_start_pcd'][c_idx].detach().cpu().numpy() 
                            util.meshcat_pcd_show(
                                mc_vis, 
                                offset + new_c_crop,
                                (0, 255, 255), 
                                f'scene/new_cropped_parent/child_pcd_{c_idx}', size=MC_SIZE) 
                        
                        if True:
                        # if False:
                            util.meshcat_pcd_show(
                                mc_vis, 
                                offset*ref_idx2 + new_p_crop + offset2*c_idx - (offset2 * sm_bs / 2.0),
                                (0, 0, 255), 
                                f'scene/new_cropped_parent/pcd_{c_idx}_{ref_idx2}', 
                                size=MC_SIZE*2.5) 

                            new_c_crop = small_policy_mi['child_start_pcd_mean'][c_idx].detach().cpu().numpy() + small_policy_mi['child_start_pcd'][c_idx].detach().cpu().numpy() 
                            util.meshcat_pcd_show(
                                mc_vis, 
                                offset*ref_idx2 + new_c_crop + offset2*c_idx - (offset2 * sm_bs / 2.0),
                                (255, 0, 0),
                                f'scene/new_cropped_parent/child_pcd_{c_idx}_{ref_idx2}', 
                                size=MC_SIZE*2.5) 

                        # util.meshcat_pcd_show(
                        #     mc_vis, 
                        #     offset + high_b[c_idx].detach().cpu().numpy().reshape(-1, 3), 
                        #     (255, 255, 128), 
                        #     f'scene/new_cropped_parent_high_sph/{c_idx}', size=0.15) 
                        # util.meshcat_pcd_show(
                        #     mc_vis, 
                        #     offset + low_b[c_idx].detach().cpu().numpy().reshape(-1, 3), 
                        #     (128, 255, 255), 
                        #     f'scene/new_cropped_parent_low_sph/{c_idx}', size=0.15) 

                ###############################################################
                # new policy input based on these features of re-cropped parent + rotated child
                child_pcd_pre_rot = small_policy_mi['child_start_pcd'].clone()
                child_pcd_mean_pre_rot = small_policy_mi['child_start_pcd_mean'].clone()

                if util.hasattr_and_true(pose_refine_model, 'pos_emb'):
                    temb_this_step = iter_to_temb[ii]
                    timestep = torch.ones((small_policy_mi['parent_start_pcd'].shape[0], 1)).float().cuda() * temb_this_step

                    # print(f'Timestep: {timestep}, clipping to n_diff_steps: {n_diff_steps}')
                    timestep_emb = pose_refine_model.pos_emb(torch.clip(timestep, 0, n_diff_steps))
                    small_policy_mi['timestep_emb'] = timestep_emb

                rot_model_output_raw = pose_refine_model(small_policy_mi)

                nq = rot_model_output_raw['rot_mat'].shape[1]
                rot_idx = torch.zeros(rot_model_output_raw['rot_mat'].shape[0]).long().cuda()

                rot_model_output = {}
                rot_model_output['rot_mat'] = torch.gather(rot_model_output_raw['rot_mat'], dim=1, index=rot_idx[:, None, None, None].repeat(1, 1, 3, 3)).reshape(sm_bs_iter, 3, 3)
                
                axis_angle = matrix_to_axis_angle(rot_model_output['rot_mat'])
                angle = axis_angle.norm(2, -1)
                axis = torch.nn.functional.normalize(axis_angle, dim=-1)
                angle_scaled = angle * step_size
                # angle_scaled = angle * lin_step_size
                axis_angle_scaled = angle_scaled.reshape(-1, 1).repeat((1, 3)) * axis
                rot_model_output['rot_mat'] = axis_angle_to_matrix(axis_angle_scaled)

                if add_per_iter_noise:
                    # sample 3-D "axis-angle noise"
                    if ii > (n_iters * 0.8):
                        aa_std = 0.0
                    else:
                        aa_std = np.deg2rad(aa_noise_angle) * np.exp(-1.0 * aa_noise_rate * (ii / n_iters))
                    aa_noise = torch.randn((rot_model_output['rot_mat'].shape[0], 3)).float().cuda() * aa_std
                    rot_mat_noise = axis_angle_to_matrix(aa_noise)
                    log_debug(f'Average rot noise: {np.rad2deg(torch.linalg.norm(aa_noise, dim=-1).mean(0).item())}')
                    rot_model_output['rot_mat'] = torch.bmm(rot_mat_noise, rot_model_output['rot_mat'])

                # make a rotated version of the original point cloud
                child_pcd_final_pred = torch.bmm(rot_model_output['rot_mat'], small_policy_mi['child_start_pcd'].transpose(1, 2)).transpose(2, 1)
                child_pcd_rot = child_pcd_final_pred.detach()
                
                # new policy input based on these features of rotated shape
                small_policy_mi['child_start_pcd'] = child_pcd_rot
                
                # get output for translation, and combine outputs with rotation prediction
                trans_model_output_raw = pose_refine_model(small_policy_mi)
                trans_idx = torch.zeros(trans_model_output_raw['trans'].shape[0]).long().cuda()

                model_output = {}
                model_output['trans'] = torch.gather(trans_model_output_raw['trans'], dim=1, index=trans_idx[:, None, None].repeat(1, 1, 3)).reshape(-1, 3)

                if add_per_iter_noise:
                    # sample 3-D translation noise
                    if ii > (n_iters * 0.8):
                        trans_std = 0.0
                    else:
                        trans_std = trans_noise_dist * np.exp(-1.0 * trans_noise_rate * (ii / n_iters))
                    trans_noise = torch.randn((model_output['trans'].shape[0], 3)).float().cuda() * trans_std
                    log_debug(f'Average trans noise: {torch.linalg.norm(trans_noise, dim=-1).mean(0).item()}')
                    model_output['trans'] = model_output['trans'] + trans_noise

                trans_queries = model_output['trans']
                trans_queries = trans_queries * step_size
                # trans_queries = trans_queries * lin_step_size
                trans_mean = small_child_pcd_mean  # initial means of child point clouds before translation

                ###############################################################
                # Update the policy model input for the next step

                # rotated child point cloud and updated world-frame mean
                refine_policy_mi['child_start_pcd'][idx:idx+sm_bs] = child_pcd_rot
                refine_policy_mi['child_start_pcd_mean'][idx:idx+sm_bs] = trans_mean + trans_queries

                # adjust the final translation and rotation based on this
                trans_queries_final[idx:idx+sm_bs] = trans_queries_final[idx:idx+sm_bs] + trans_queries
                rot_mat_queries_final[idx:idx+sm_bs] = torch.bmm(rot_model_output['rot_mat'], rot_mat_queries_final[idx:idx+sm_bs])

                idx += sm_bs 

            if export_viz:
                # collect the initial set of predicted poses
                export_all_idx = np.arange(trans_queries_final.shape[0]).astype(int)
                voxel_topk = voxel_idx_final[export_all_idx].squeeze()
                rot_topk = rot_idx_final[export_all_idx].squeeze()
                trans_topk = trans_idx_final[export_all_idx].squeeze()

                # get the final point clouds (final rot, final mean, which includes trans)
                child_pcd_cent_topk = refine_policy_mi['child_start_pcd'][export_all_idx]
                mean_topk = refine_policy_mi['child_start_pcd_mean'][export_all_idx]
                child_pcd_final_topk = child_pcd_cent_topk + mean_topk[:, None, :].repeat((1, N_crop, 1))

                # voxel - index into the world points
                out_export_voxel_trans = world_pts_t[voxel_topk].reshape(-1, 3)

                # initial random rotation - use the VOXEL inds (these correspond to our original batch size)
                out_export_rot_init = rand_mat_init.reshape(-1, 4, 4)[voxel_topk].reshape(-1, 4, 4)

                # rotation and translation - index into the final queries
                out_export_rot = rot_mat_queries_final[rot_topk].reshape(-1, 3, 3)
                out_export_trans = trans_queries_final[trans_topk].reshape(-1, 3)

                # combine trans all together with voxel trans and trans queries
                out_export_trans_full = out_export_voxel_trans + out_export_trans

                child_pcd_ds_np = child_pcd_ds.reshape(-1, 3).detach().cpu().numpy()
                child_pcd_cent = child_pcd_ds_np - np.mean(child_pcd_ds_np, axis=0)
                
                out_export_tmat_list = []
                for ex_ii in export_all_idx:
                    # save the translation component
                    tmat_export = np.eye(4); tmat_export[:-1, -1] = out_export_trans_full[ex_ii].detach().cpu().numpy().squeeze()

                    # get the rotation and the initial rotation
                    tmat_export[:-1, :-1] = out_export_rot[ex_ii].detach().cpu().numpy().squeeze()
                    tmat_export_init = out_export_rot_init[ex_ii].detach().cpu().numpy().squeeze()

                    # combine the initial rotation with the final rotation
                    tmat_export_full = np.matmul(tmat_export, tmat_export_init)

                    # apply correction for mean offsets that accumulated
                    final_pcd_ii = child_pcd_final_topk[ex_ii].detach().cpu().numpy().squeeze()
                    trans_child_pcd_ii = util.transform_pcd(child_pcd_cent, tmat_export_full)
                    delta_mean = np.mean(final_pcd_ii, axis=0) - np.mean(trans_child_pcd_ii, axis=0)
                    tmat_export_full[:-1, -1] += delta_mean

                    tf_cent = np.eye(4); tf_cent[:-1, -1] = -1.0 * np.mean(child_pcd_original, axis=0)
                    out_export_tf = np.matmul(tmat_export_full, tf_cent)
                    # out_export_tf = np.matmul(tmat_export_full_list[0], tf_cent)  # get the TOP scoring one

                    export_iter_data = {}
                    export_iter_data['parent_pose_mat'] = np.eye(4)
                    export_iter_data['child_pose_mat'] = out_export_tf

                    ppcd_viz = refine_policy_mi['parent_start_pcd'][ex_ii].detach().cpu().numpy()
                    ppcd_mean_viz = refine_policy_mi['parent_start_pcd_mean'][ex_ii].detach().cpu().numpy()
                    export_iter_data['parent_pcd_cropped'] = ppcd_viz + ppcd_mean_viz

                    out_export_tmat_list.append(export_iter_data)

                opt_iter_data.append(out_export_tmat_list)

    if export_viz:
        ex_data_dict.opt_iter_data = opt_iter_data
        np.savez(osp.join(export_dir, 'export_data.npz'), **dict(ex_data_dict))

        from IPython import embed; embed()

    ###############################################################
    # Filter out redundant final predictions
    if util.exists_and_true(kwargs, 'compute_coverage_scores'):
        pass
    else:
        from rpdiff.utils.relational_policy.filter_multimodal_placement import remove_redundant_tf

        out = remove_redundant_tf(refine_policy_mi, voxel_idx_final, rot_idx_final, trans_idx_final, mc_vis)
        refine_policy_mi, voxel_idx_final, rot_idx_final, trans_idx_final = out

    ###############################################################
    # Make final success output predictions
    
    # put in FULL parent point cloud for final success classification
    pscene_mean = torch.mean(pscene_pcd_t, axis=0).reshape(1, 3)
    pscene_pcd_down_t = fps_downsample(pscene_pcd_t.reshape(1, -1, 3), 8192)
    pscene_pcd_cent_t = pscene_pcd_down_t - pscene_mean.repeat((8192, 1))
    policy_mi['parent_start_pcd'] = pscene_pcd_cent_t.reshape(1, -1, 3).repeat((refine_policy_mi['child_start_pcd'].shape[0], 1, 1))
    policy_mi['parent_start_pcd_mean'] = pscene_mean.repeat((refine_policy_mi['child_start_pcd'].shape[0], 1))

    # break into smaller batches
    sm_bs = 8
    idx = 0
    success_out_list = []

    if no_sc_score:
        success_out_all = torch.ones((refine_policy_mi['child_start_pcd'].shape[0])).float().cuda()
    else:
        with torch.no_grad():
            while True:
                if idx >= refine_policy_mi['child_start_pcd'].shape[0]:
                    break

                # get the final resulting child point cloud + child mean
                small_policy_mi = {k: v[idx:idx+sm_bs] for k, v in refine_policy_mi.items()}
               
                # overwrite the parent point cloud to ensure we use FULL parent point cloud and mean
                small_policy_mi['parent_start_pcd'] = policy_mi['parent_start_pcd'][idx:idx+sm_bs]
                small_policy_mi['parent_start_pcd_mean'] = policy_mi['parent_start_pcd_mean'][idx:idx+sm_bs]

                # make success prediction and save success probability in list
                small_success_out = success_model(small_policy_mi)
                success_out_list.extend(small_success_out['success'].detach().cpu().numpy().reshape(-1).tolist())

                idx += sm_bs 

        success_out_all = torch.Tensor(success_out_list).float().cuda()

        man_idx = 0
        man_success_out_list = []

        ###############################################################
        # Incorporate additional collision checking into success scoring

        if with_coll:
            # ppcd_world_all = (refine_policy_mi['parent_start_pcd'] + refine_policy_mi['parent_start_pcd_mean'].reshape(-1, 1, 3).repeat((1, refine_policy_mi['parent_start_pcd'].shape[1], 1))).detach().cpu().numpy()
            ppcd_world = pscene_pcd_t.detach().cpu().numpy()
            cpcd_world_all = (refine_policy_mi['child_start_pcd'] + refine_policy_mi['child_start_pcd_mean'].reshape(-1, 1, 3).repeat((1, refine_policy_mi['child_start_pcd'].shape[1], 1))).detach().cpu().numpy()

            if 'mesh_dict' in kwargs:
                mesh_dict = kwargs['mesh_dict']

            from rpdiff.utils.relational_policy.filter_multimodal_placement import manual_place_success, manual_place_success_gt
            for j in range(cpcd_world_all.shape[0]):
                # ppcd_world = ppcd_world_all[j]
                cpcd_world = cpcd_world_all[j]

                man_success = manual_place_success_gt(ppcd_world, cpcd_world, mesh_dict, mc_vis)
                # man_success = manual_place_success(ppcd_world, cpcd_world, mc_vis)
                man_success_out_list.append(man_success)
            
            man_success_out_all = torch.Tensor(man_success_out_list).float().cuda()

            success_out_all = man_success_out_all * success_out_all

    ###############################################################
    # Gather top-k success values and corresponding rotations/translations/voxel indices

    success_top_val, success_top_idx = torch.topk(success_out_all, k=1)

    if util.exists_and_true(kwargs, 'compute_coverage_scores'):
        success_topk_idx = torch.where(success_out_all > 0.0*success_top_val)[0]
    else:
        success_topk_idx = torch.where(success_out_all > 0.9*success_top_val)[0]
    # success_topk_idx = torch.where(success_out_all > 0.9*success_top_val)[0]
    success_topk_vals = success_out_all[success_topk_idx]
    success_kval = success_topk_idx.shape[0]
    print(f'Top value: {success_top_val.item():.4f}, Number of top (to sample): {success_kval}')

    # get the final point clouds (final rot, final mean, which includes trans)
    child_pcd_cent_topk = refine_policy_mi['child_start_pcd'][success_topk_idx]
    mean_topk = refine_policy_mi['child_start_pcd_mean'][success_topk_idx]
    child_pcd_final_topk = child_pcd_cent_topk + mean_topk[:, None, :].repeat((1, N_crop, 1))

    # get the final values for the translation/rotation

    # start with the idx
    voxel_topk = voxel_idx_final[success_topk_idx].squeeze()
    rot_topk = rot_idx_final[success_topk_idx].squeeze()
    trans_topk = trans_idx_final[success_topk_idx].squeeze()

    # voxel - index into the world points
    out_voxel_trans = world_pts_t[voxel_topk].reshape(-1, 3)

    # initial random rotation - use the VOXEL inds (these correspond to our original batch size)
    out_rot_init = rand_mat_init.reshape(-1, 4, 4)[voxel_topk].reshape(-1, 4, 4)

    # rotation and translation - index into the final queries
    out_rot = rot_mat_queries_final[rot_topk].reshape(-1, 3, 3)
    out_trans = trans_queries_final[trans_topk].reshape(-1, 3)

    # combine trans all together with voxel trans and trans queries
    out_trans_full = out_voxel_trans + out_trans

    ###############################################################
    # Combine and compute final transforms, and visualize final output predictions 

    child_pcd_ds_np = child_pcd_ds.reshape(-1, 3).detach().cpu().numpy()
    child_pcd_cent = child_pcd_ds_np - np.mean(child_pcd_ds_np, axis=0)

    # colormap for the top-k
    cmap = plt.get_cmap('inferno')
    color_list = cmap(np.linspace(0.1, 0.9, success_kval, dtype=np.float32))[::-1]
    # util.meshcat_pcd_show(mc_vis, parent_pcd, (255, 0, 0), 'scene/infer/parent_pcd', size=MC_SIZE)

    # create list to store full transforms
    tmat_full_list = []
    ii_bucket = 0
    for ii in range(success_kval):

        if ii % 10 == 0:
            ii_bucket += 1
        # if viz:
        if 'final_box' in viz_flags:
            color = (color_list[ii][:-1] * 255).astype(np.uint8).tolist()
            box = trimesh.PointCloud(child_pcd_final_topk[ii].detach().cpu().numpy().squeeze()).bounding_box_oriented.to_mesh()
            # util.meshcat_trimesh_show(mc_vis, f'scene/infer/success_topk_box/child_pts_box_{ii}', box, color)
            util.meshcat_trimesh_show(mc_vis, f'scene/infer/success_topk_box/{ii_bucket}/child_pts_box_{ii}', box, color)
        
        # save the translation component
        tmat = np.eye(4); tmat[:-1, -1] = out_trans_full[ii].detach().cpu().numpy().squeeze()

        # get the rotation and the initial rotation
        tmat[:-1, :-1] = out_rot[ii].detach().cpu().numpy().squeeze()
        tmat_init = out_rot_init[ii].detach().cpu().numpy().squeeze()

        # combine the initial rotation with the final rotation
        tmat_full = np.matmul(tmat, tmat_init)
        
        # apply correction for mean offsets that accumulated
        final_pcd_ii = child_pcd_final_topk[ii].detach().cpu().numpy().squeeze()
        trans_child_pcd_ii = util.transform_pcd(child_pcd_cent, tmat_full)
        delta_mean = np.mean(final_pcd_ii, axis=0) - np.mean(trans_child_pcd_ii, axis=0)
        tmat_full[:-1, -1] += delta_mean

        # save whole thing
        tmat_full_list.append(tmat_full)
        
        # double check that applying transfom to the CENTERED child point cloud results in correct final pose
        if 'final_pcd' in viz_flags:
            color = (color_list[ii][:-1] * 255).astype(np.uint8).tolist()
            final_pcd_ii = child_pcd_final_topk[ii].detach().cpu().numpy().squeeze()
            util.meshcat_pcd_show(mc_vis, final_pcd_ii, color, f'scene/infer/success_topk_pcd/child_pts_tf_{ii}', size=MC_SIZE)

            trans_child_pcd = util.transform_pcd(child_pcd_cent, tmat_full)
            util.meshcat_pcd_show(mc_vis, trans_child_pcd, (0, 255, 255), f'scene/infer/success_topk_apply_pcd/{ii_bucket}/child_pts_apply_tf_{ii}', size=MC_SIZE)

    tf_cent = np.eye(4); tf_cent[:-1, -1] = -1.0 * np.mean(child_pcd_ds_np, axis=0)

    if return_top:
        out_tf = np.matmul(tmat_full_list[0], tf_cent)  # get the TOP scoring one
    else:
        if 'return_highest' in kwargs:
            if kwargs['return_highest']:
                z_vals = [np.max(child_pcd_final_topk[ii][:, 2].detach().cpu().numpy()) for ii in range(child_pcd_final_topk.shape[0])]
                topz_idx = np.argmax(z_vals)
                out_tf = np.matmul(tmat_full_list[topz_idx], tf_cent)
            else:
                rand_tf_idx = np.random.randint(len(tmat_full_list))  # randomly sample ANY of the top-k scoring ones
                out_tf = np.matmul(tmat_full_list[rand_tf_idx], tf_cent)
        else:
            rand_tf_idx = np.random.randint(len(tmat_full_list))  # randomly sample ANY of the top-k scoring ones
            out_tf = np.matmul(tmat_full_list[rand_tf_idx], tf_cent)

    # print('here done with inf (multistep)')
    # from IPython import embed; embed()

    # if compute_coverage_scores:
    if util.exists_and_true(kwargs, 'compute_coverage_scores'):

        tf_cent = np.eye(4); tf_cent[:-1, -1] = -1.0 * np.mean(child_pcd_original, axis=0)

        tmat_full_w_guess = []
        final_pose_pred_list = []
        for tmat in tmat_full_list:
            tmat_w_guess = np.matmul(tmat, np.matmul(tf_cent, kwargs['export_viz_relative_trans_guess']))
            tmat_full_w_guess.append(tmat_w_guess)
            
            pred_pose_wf = np.matmul(tmat_w_guess, child_pose_mat)
            final_pose_pred_list.append(pred_pose_wf)

        from rpdiff.utils.relational_policy.filter_multimodal_placement import compute_coverage
        out_coverage_dict = compute_coverage(
            final_pose_pred_list, 
            avail_poses_wf, 
            success_out_all.detach().cpu().numpy().tolist(), 
            mc_vis) 
        out_coverage_dir1 = osp.join(path_util.get_rpdiff_eval_data(), 'rpdiff_coverage_data', kwargs['out_coverage_dirname1'])
        out_coverage_dir2 = osp.join(path_util.get_rpdiff_eval_data(), 'rpdiff_coverage_data', kwargs['out_coverage_dirname2'])
        util.safe_makedirs(out_coverage_dir1)
        util.safe_makedirs(out_coverage_dir2)
        out_cov_fname1 = osp.join(out_coverage_dir1, f'out_coverage_data_iter_{kwargs["iteration"]}.json')
        out_cov_fname2 = osp.join(out_coverage_dir2, f'out_coverage_data_iter_{kwargs["iteration"]}.json')

        json.dump(out_coverage_dict, open(out_cov_fname1, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
        json.dump(out_coverage_dict, open(out_cov_fname2, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    return out_tf


def multistep_qattn_scene_combine_sc(
        mc_vis, parent_pcd, child_pcd, 
        coarse_aff_model, 
        pose_refine_model, pose_refine_model_refine,
        success_model,
        feat_model,
        scene_scale, scene_mean,
        grid_pts, rot_grid=None, 
        viz=False, n_iters=10, 
        return_all_child_pcds=False, no_parent_crop=False, 
        return_top=True, with_coll=False, run_affordance=True, init_k_val=10,
        no_sc_score=False, init_parent_mean=False, init_orig_ori=False,
        refine_anneal=False,
        *args, **kwargs):
    """
    Relative pose predictions made in multiple iterative steps. 
    Uses: 
    - a final success classifier
    - potentially two versions of the regression model
    - an initial voxel affordance model to provide an initial translation
      and local scene point cloud crop
    - potentially a separate feature encoder to provide extra per-point
      features

    Args:
        mc_vis (meshcat.Visualizer): meshcat visualization handler
        parent_pcd (np.ndarray): Nx3 point cloud of parent/A object
        child_pcd (np.ndarray): Nx3 point cloud of child/B object
        coarse_aff_model (torch.nn.Module): 3D CNN point cloud encoder 
            (local pointnet + 3D CNN) and the output NN that takes 
            per-voxel scene features and predicts scores. Used to 
            obtain an initialization for translating + cropping the 
            child point cloud. Can also be used to initialize the rotation (WIP).
        success_model (torch.nn.Module): NN that takes parent/A and final
            transformed child/B point cloud as input and predicts an overall
            score of quality. Used to select among multimodal output predictions.
        feat_model (dict or torch.nn.Module): If dict, Keys are "parent" and "child",
            Values are separate NNs (torch.nn.Module), corresponding
            to an encoder for the parent and the child. Else, just a 
            shared NN encoder.
        grid_pts (np.ndarray): Voxelized raster points corresponding to the
            coordinates of the scene model output ("voxel affordance"). Used
            to form initial translation in the world frame
        rot_grid (np.ndarray): Array of uniformly distributed 3D rotation matrices
        viz (bool): If True, visualize the predictions along the way
        n_iters (int): Number of refinement steps to take
        return_all_child_pcd (bool): If True, output all of the child point clouds
            (might be useful for visualizing the refinement, not used in this version)
        no_parent_crop (bool): If True, don't run the scene model and don't perform
            any cropping of the parent/A point cloud.
        return_top (bool): If True, return the TOP scoring among the multimodal
            outputs. Otherwise, randomly sample ANY of the top-k scoring
        with_coll (bool): If True, also include some form of collision checking
            in the success scoring after multimodal prediction is all done
        run_affordance (bool): If True, use trained model to predict initial
            translation positions. Else, randomly sample. 
        init_k_val (int): Number of initial positions to sample
        no_sc_score (bool): If True, don't use the success classifier to score,
            just assign everything an equal score of 1.0

    Returns:
        np.ndarray: 4x4 homogeneous transformation matrix representing transformation
            of child/B point cloud in the world frame 
    """

    raise NotImplementedError('C2F Q-attention implementation TODO')


policy_inference_methods_dict = {
    'multistep_regression_scene_combine_succ_cls': multistep_regression_scene_combine_sc,
    'multistep_qattn_succ_cls': multistep_qattn_scene_combine_sc
}
