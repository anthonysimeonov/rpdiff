import os, os.path as osp
import sys
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch
from torch.utils.data import Dataset
import time

# from conv_onet.common import make_3d_grid

from rpdiff.utils import util, path_util, pcd_aug_util
from rpdiff.utils.mesh_util import three_util
from rpdiff.utils.mesh_util.inside_mesh import check_mesh_contains

from rpdiff.training import train_util
import trimesh

from rpdiff.utils.config_util import AttrDict
from meshcat import Visualizer
from typing import List, Tuple, Union
# from rpdiff.torch_utils import center_pcd

# import open3d


def o3d_fps(np_pcd, num_samples=1024):
    o3d_pcd = open3d.geometry.PointCloud()
    o3d_pcd.points = open3d.utility.Vector3dVector(np_pcd)

    o3d_pcd_ds = o3d_pcd.farthest_point_down_sample(num_samples=num_samples)
    return np.asarray(o3d_pcd_ds.points)
    

class FullRelationPointcloudPolicyDataset(Dataset):
    def __init__(self, dataset_path: str, data_args: AttrDict, phase: str='train', 
                 train_coarse_aff: bool=True, train_refine_pose: bool=True, train_success: bool=True,
                 mc_vis: Visualizer=None, debug_viz: bool=False):
        self.data_path = dataset_path
        self.data_args = data_args

        self.train_coarse_aff = train_coarse_aff
        self.train_refine_pose = train_refine_pose
        self.train_success = train_success

        self.diffusion_steps = self.data_args.refine_pose.diffusion_steps
        
        split = self.data_args.split

        split_path = osp.join(self.data_path, 'split_info')
        test_split = np.loadtxt(osp.join(split_path, 'test_split.txt'), dtype=str).tolist()
        val_split = np.loadtxt(osp.join(split_path, 'train_val_split.txt'), dtype=str).tolist()
        train_split = np.loadtxt(osp.join(split_path, 'train_split.txt'), dtype=str).tolist()

        self.phase = phase
        self.not_success_idx = []

        self.mc_vis = mc_vis
        self.debug_viz = debug_viz

        self.load_chunked_data = self.data_args.chunked
        self.load_data_into_memory = self.data_args.load_into_memory
        self.files = []
        self.img_files = []

        if self.load_chunked_data:
            for (root, dirs, files) in os.walk(self.data_path):
                if not len(dirs):
                    self.files.extend([osp.join(root, fn) for fn in files if fn.endswith('.npz')])
                    self.img_files.extend([osp.join(root, fn) for fn in files if fn.endswith('.png')])
        else:
            self.files = [osp.join(self.data_path, fn) for fn in os.listdir(self.data_path) if fn.endswith('.npz')]
            self.img_files = [osp.join(self.data_path, fn) for fn in os.listdir(self.data_path) if fn.endswith('.png')]
        if split == 'train' and self.phase != 'val':
            files = [fn for fn in self.files if fn.split('/')[-1] in train_split]
        elif split == 'val' or self.phase == 'val':
            files = [fn for fn in self.files if fn.split('/')[-1] in val_split]
        elif split == 'test':
            files = [fn for fn in self.files if fn.split('/')[-1] in test_split]
                
        self.files = files 
        print(f'Split: {split}, files length: {len(self.files)} ')

        if not len(self.files):
            print("here with no files")
            from IPython import embed; embed()
       
        self.data_chunks = None
        self.data = None
        self.index2chunked_inds = None
        if self.load_data_into_memory:
            print('Loading data into memory...')
            tic = time.time()
            if self.load_chunked_data:
                print(f'Loading data chunks into memory')
                self.data_chunks = [np.load(fn, allow_pickle=True) for fn in self.files]
                self.data = []
                for i, data_chunk in enumerate(self.data_chunks):
                    print(f'Processed {i+1} data chunks...')
                    for f in data_chunk.files:
                        data = data_chunk[f].item()
                        self.data.append(data)
            else:
                print(f'\n\n\n***Configuration settings "load_data_into_memory" True and "load_chunked_data" False***\n***This can lead to OSError being raised, due to too many open files... You have been warned...***\n\n\n')
                time.sleep(4.0)
                self.data = []
                for i, fname in enumerate(self.files):
                    print(f'Processed {i+1} files...')
                    data = np.load(fname, allow_pickle=True)
                    self.data.append(data)
            print(f'Total number of data samples: {len(self.data)}')
        else:
            if self.load_chunked_data:
                self.index2chunked_inds = []  # maps index for full dataset size into pair of indices for chunked data
                for i, fn in enumerate(self.files):
                    data_chunk = np.load(fn, allow_pickle=True)
                    num_data = len(data_chunk.files)
                    for j in range(num_data):
                        self.index2chunked_inds.append((i, j))
                print(f'Total number of data samples: {len(self.index2chunked_inds)}')
            else:
                print(f'Total number of data samples: {len(self.files)}')

            toc = time.time()
            dif = toc - tic; print(f'Load data time: {dif}')
        
        print('Loading full point clouds...')

        # let's load in "dense point clouds" for all of the objects
        # parent_mesh_dir = osp.join(path_util.get_rpdiff_obj_descriptions(), 'syn_bookshelf_obj_unnormalized')
        # child_mesh_dir = osp.join(path_util.get_rpdiff_obj_descriptions(), 'syn_books_obj')

        self.parent_full_pcd_arr = None
        self.child_full_pcd_arr = None
        if self.data_args.load_full_pcd:
            parent_mesh_dir = osp.join(path_util.get_rpdiff_obj_descriptions(), self.data_args.parent_mesh_dir)
            child_mesh_dir = osp.join(path_util.get_rpdiff_obj_descriptions(), self.data_args.child_mesh_dir)
            assert osp.exists(parent_mesh_dir), f'Mesh directory: {parent_mesh_dir} does not exist'
            assert osp.exists(child_mesh_dir), f'Mesh directory: {child_mesh_dir} does not exist'

            parent_mesh_fnames = [osp.join(parent_mesh_dir, fn) for fn in os.listdir(parent_mesh_dir) if fn.endswith('.obj') and 'dec' not in fn]
            child_mesh_fnames = [osp.join(child_mesh_dir, fn) for fn in os.listdir(child_mesh_dir) if fn.endswith('.obj') and 'dec' not in fn]

            self.parent_full_pcd_dict = {}
            self.child_full_pcd_dict = {}
            parent_full_pcd_arr = []
            child_full_pcd_arr = []
            self.parent_name2idx = {}
            self.child_name2idx = {}
            
            n_full = 5192
            for i, mesh_fname in enumerate(parent_mesh_fnames):
                mesh = trimesh.load(mesh_fname, process=False)
                pts = mesh.sample(n_full)
                parent_full_pcd_arr.append(pts)
                obj_key = mesh_fname.split('/')[-1].replace('.obj', '')
                self.parent_full_pcd_dict[obj_key] = pts
                self.parent_name2idx[obj_key] = i
            for i, mesh_fname in enumerate(child_mesh_fnames):
                mesh = trimesh.load(mesh_fname, process=False)
                pts = mesh.sample(n_full)
                child_full_pcd_arr.append(pts)
                obj_key = mesh_fname.split('/')[-1].replace('.obj', '')
                self.child_full_pcd_dict[obj_key] = pts
                self.child_name2idx[obj_key] = i

            self.parent_full_pcd_arr = np.asarray(parent_full_pcd_arr).reshape(-1, n_full, 3)
            self.child_full_pcd_arr = np.asarray(child_full_pcd_arr).reshape(-1, n_full, 3)
            # print(f'Loaded set of "complete point clouds" from object meshes\nParent array dim: {parent_full_pcd_arr.shape}\nChild array dim: {child_full_pcd_arr.shape}')
            print(f'Loaded set of "complete point clouds" from object meshes\nParent dict: {len(self.parent_full_pcd_dict)}\nChild array dim: {len(self.child_full_pcd_dict)}')

        # setup 
        self._setup_general()
        # self._setup_debug()
        self._setup_voxel_grid()
        self._setup_rotation_grid()
        self._setup_pose_prediction()
        self._setup_local_pose_prediction()
        self._setup_success()

        # self.MC_SIZE = 0.004
        self.MC_SIZE = 0.0055

    def _setup_general(self):
        self.shape_pcd_n = self.data_args.shape_pcd_n
    
        if self.data_args.parent_shape_pcd_n is not None:
            self.parent_shape_pcd_n = self.data_args.parent_shape_pcd_n
        else:
            self.parent_shape_pcd_n = self.shape_pcd_n

        if self.data_args.child_shape_pcd_n is not None:
            self.child_shape_pcd_n = self.data_args.child_shape_pcd_n
        else:
            self.child_shape_pcd_n = self.shape_pcd_n

        rot_aug = self.data_args.rot_aug
        # self.rot_aug = rot_aug if rot_aug not in [None, 'None'] else None
        self.rot_aug = rot_aug
        # self.interaction_pt_aug = self.data_args.interaction_pt_aug
        self.apply_pcd_aug = self.data_args.apply_pcd_aug
        self.pcd_aug_prob = self.data_args.pcd_aug_prob
        self.full_pcd_aug = self.data_args.full_pcd_aug
        self.pcd_aug_pp_std = self.data_args.pcd_aug_pp_std

        self.load_full_pcd = self.data_args.load_full_pcd

    def _setup_debug(self):

        self.use_small_rot_scale = False
        self.use_small_trans_scale = False

        single_small_rpy = self.rot_scale * (np.random.random(3) - 0.5)
        # single_small_rotmat = R.from_euler('xyz', single_small_rpy).as_matrix()
        single_small_rotmat = R.from_euler('xyz', [0, -np.pi/2, -np.pi/6]).as_matrix()
        single_small_trans = self.trans_scale * (np.random.random(3) - 0.5)
        self.single_small_rotmat = single_small_rotmat
        self.single_small_trans = single_small_trans
        
        self.set_small_trans = [
            self.trans_scale * (np.random.random(3) - 0.5),
            self.trans_scale * (np.random.random(3) - 0.5),
            self.trans_scale * (np.random.random(3) - 0.5),
            self.trans_scale * (np.random.random(3) - 0.5),
            self.trans_scale * (np.random.random(3) - 0.5),
        ]

        self.set_small_rpy = [
            self.rot_scale * (np.random.random(3) - 0.5),
            self.rot_scale * (np.random.random(3) - 0.5),
            self.rot_scale * (np.random.random(3) - 0.5),
            self.rot_scale * (np.random.random(3) - 0.5),
            self.rot_scale * (np.random.random(3) - 0.5),
        ]
        self.set_small_rotmat = [R.from_euler('xyz', euler).as_matrix() for euler in self.set_small_rpy]

        debug_mode = False
        self.single_offset_debug = True and debug_mode
        self.single_rot_offset_debug = False and debug_mode
        self.single_trans = False and debug_mode
        self.single_rot = False and debug_mode
        if self.single_offset_debug:
            print('!! Using single trans/rot for debugging !!')
        self.fixed_perm = debug_mode
        if self.fixed_perm:
            print('!!! Using fixed permutation of the point clouds! !!')

        self.aff_out_viz = None
        self.pose_out_viz = None
        self.success_out_viz = None

    def _setup_voxel_grid(self):
        self.reso_grid = self.data_args.voxel_grid.reso_grid
        self.padding = self.data_args.voxel_grid.padding

        # get the B x N x 3 raster points
        # raster_pts = make_3d_grid((-0.55, -0.55, -0.55), (0.55, 0.55, 0.55), (self.reso_grid, self.reso_grid, self.reso_grid))[None, :, :].repeat(batch_size, 1, 1).cuda() #.numpy()
        raster_pts = three_util.get_raster_points(self.reso_grid, padding=self.padding)
        self.raster_pts_z = raster_pts.copy()

        # reshape to grid, and swap axes (permute x and z), B x reso x reso x reso x 3
        raster_pts = raster_pts.reshape(self.reso_grid, self.reso_grid, self.reso_grid, 3)
        raster_pts = raster_pts.transpose(2, 1, 0, 3)

        # reshape back to B x N x 3
        # self.raster_pts_z = raster_pts.reshape(-1, 3)
        self.raster_pts = raster_pts.reshape(-1, 3)

        # raster_pts2 = make_3d_grid((-0.55, -0.55, -0.55), (0.55, 0.55, 0.55), (self.reso_grid, self.reso_grid, self.reso_grid)).numpy()

    def _setup_rotation_grid(self):
        self.rot_grid = util.generate_healpix_grid(size=self.data_args.rot_grid_samples) 
        bins_per_axis = self.data_args.euler_bins_per_axis
        self.euler_rot_disc = np.linspace(-np.pi, np.pi, bins_per_axis)

    def _setup_pose_prediction(self):
        # self.use_small_rot_scale = False
        # self.use_small_trans_scale = False
        self.use_small_rot_scale = self.data_args.pose_perturb.use_small_rot_scale
        self.use_small_trans_scale = self.data_args.pose_perturb.use_small_trans_scale
        if self.use_small_rot_scale:
            print('!! Using small rotation scale !!')
        if self.use_small_trans_scale:
            print('!! Using small translation scale !!')

        self.rot_scale = np.deg2rad(self.data_args.pose_perturb.rot_scale_deg)
        self.trans_scale = self.data_args.pose_perturb.trans_scale

        self.small_rot_scale = np.deg2rad(self.data_args.pose_perturb.small_rot_scale_deg)
        self.small_trans_scale = self.data_args.pose_perturb.small_trans_scale

        # self.small_rot_scale = np.deg2rad(1)
        # self.small_trans_scale = 0.001

        # self.rot_scale = np.deg2rad(360) if not self.refine else np.deg2rad(30)
        # self.trans_scale = 0.12 if not self.refine else 0.04

        self.parent_cent_offset_scale = self.data_args.pose_perturb.parent_cent_offset_scale
        # self.parent_cent_offset_scale = 0.01
        # self.parent_cent_offset_prob = 0.5 if not self.refine else 0.0

        self.parent_cent_offset_prob = self.data_args.pose_perturb.parent_cent_offset_prob
        self.rnd_parent_pt_offset_prob = self.data_args.pose_perturb.rnd_parent_pt_offset_prob
        self.rnd_parent_pcd_pt_offset_prob = self.data_args.pose_perturb.rnd_parent_pcd_pt_offset_prob
        self.parent_crop = self.data_args.parent_crop

        # self.parent_cent_offset_prob = 1.0 if self.no_parent_crop else self.parent_cent_offset_prob
        # self.parent_cent_offset_prob = self.parent_cent_offset_prob if self.parent_crop else 1.0
        if self.parent_cent_offset_prob > 0.0:
            print('!! Using LARGE prob for applying offset w.r.t. the parent center / random translation (initial guess!) !!')
        if self.rnd_parent_pt_offset_prob > 0.0:
            print('!! Using LARGE prob for applying offset w.r.t. the random point inside parent bounding box / random translation (initial guess!) !!')
        if self.rnd_parent_pcd_pt_offset_prob > 0.0:
            print('!! Using LARGE prob for applying offset w.r.t. the random point from the parent point cloud / random translation (initial guess!) !!')

    def _setup_local_pose_prediction(self):
        self.valid_rots_euler = [
            np.array([0, 0, np.pi]),
            np.array([0, 0, -np.pi]),
            np.array([np.pi, 0, 0]),
            np.array([-np.pi, 0, 0]),
            np.array([0, np.pi, 0]),
            np.array([0, -np.pi, 0]),
        ]
    
        self.parent_crop_refine = self.data_args.refine_pose.parent_crop
        self.max_length = self.data_args.refine_pose.crop_box_length

    def _setup_success(self):
        self.success_rot_scale = np.deg2rad(self.data_args.success.success_rot_scale_deg)
        self.success_trans_scale = self.data_args.success.success_trans_scale
        self.fail_rot_scale = np.deg2rad(self.data_args.success.fail_rot_scale_deg)
        self.fail_trans_scale = self.data_args.success.fail_trans_scale
        self.fail_rot_min = np.deg2rad(self.data_args.success.fail_rot_min_deg)
        self.fail_trans_min = self.data_args.success.fail_trans_min

        self.success_fail_prob = self.data_args.success.success_fail_prob

    def __len__(self):
        # if self.load_chunked_data:
        if self.load_data_into_memory:
            return len(self.data)
        else:
            if self.load_chunked_data:
                return len(self.index2chunked_inds)
            else:
                return len(self.files)

    def get_sample_success(self, data):
        success = data['success'].item()

        if success is None:
            success = False
        
        queried_success = pred_success = False
        if 'queried_success' in data.keys():
            queried_success = data['queried_success']
        if 'pred_success' in data.keys():
            pred_success = data['pred_success']
        
        success = success or queried_success or pred_success
        return float(success)

    def get_item(self, index: int, embed_invalid: bool=False) -> Tuple[Tuple[dict]]:
        """
        Args:
            index (int): Dataset index to sample
            embed_invalid (bool): If True, embed into interactive termainl if we hit a runtime error

        Returns:
            3-element tuple containing:
              2-element tuple containing inputs and ground truth for voxel affordance model:
                dict: model inputs 
                dict: ground truth 
              2-element tuple containing inputs and ground truth for pose regression model:
                dict: model inputs 
                dict: ground truth 
              2-element tuple containing inputs and ground truth for success classifier
                dict: model inputs 
                dict: ground truth 
        """
        if index in self.not_success_idx:
            # return self.get_item(np.random.randint(len(self.files)))
            return self.get_item(np.random.randint(len(self.data)))
        
        if self.load_data_into_memory:
            data = self.data[index]
        else:
            if self.load_chunked_data:
                chunk_i, chunk_j = self.index2chunked_inds[index]
                data_chunk = np.load(self.files[chunk_i], allow_pickle=True)
                data = data_chunk[str(chunk_j)].item()
            else:
                data = np.load(self.files[index], allow_pickle=True)

        if False:
            data = self.data[index]
        sample_success = self.get_sample_success(data)

        if not sample_success:
            self.not_success_idx.append(index)
            # return self.get_item(np.random.randint(len(self.files)))
            return self.get_item(np.random.randint(len(self.data)))

        # get the start and final point clouds (used by all)
        parent_final_pcd_raw = data['multi_obj_final_pcd'].item()['parent']
        child_final_pcd_raw = data['multi_obj_final_pcd'].item()['child']

        # Load "full" point clouds instead of camera point clouds
        if self.load_full_pcd:
            child_name = data['multi_obj_mesh_file'].item()['child'].split('/')[-1].replace('.obj', '')
            child_full_pcd_canon = self.child_full_pcd_arr[self.child_name2idx[child_name]]
            child_rot = R.from_quat(data['multi_obj_final_obj_pose'].item()['child'][3:]).as_matrix()
            child_trans = data['multi_obj_final_obj_pose'].item()['child'][:3]
            child_pose_mat = np.eye(4); child_pose_mat[:-1, :-1] = child_rot; child_pose_mat[:-1, -1] = child_trans
            child_full_pcd = util.transform_pcd(child_full_pcd_canon, child_pose_mat)

            child_final_pcd_raw = child_full_pcd

        if self.debug_viz:
            util.meshcat_pcd_show(self.mc_vis, parent_final_pcd_raw, (255, 0, 0), 'scene/data_loader/get_raw/parent_final_pcd')
            util.meshcat_pcd_show(self.mc_vis, child_final_pcd_raw, (0, 0, 255), 'scene/data_loader/get_raw/child_final_pcd')

        try:
            parent_final_pcd, child_final_pcd = self.process_general_start_final_pcd(data, parent_final_pcd_raw, child_final_pcd_raw)
        except ValueError as e:
            print(f'[Data Loader] Exception: {e}')
            self.not_success_idx.append(index)
            # return self.get_item(np.random.randint(len(self.files)))
            return self.get_item(np.random.randint(len(self.data)))

        # get the specific inputs and targets for the affordance, pose, and success
        voxel_mi, voxel_gt = {}, {}
        pose_mi, pose_gt = {}, {}
        success_mi, success_gt = {}, {}
        if self.train_coarse_aff:
            voxel_mi, voxel_gt = self.get_aff_input_gt(
                data, 
                parent_final_pcd, child_final_pcd)
        if self.train_refine_pose:
            # print(f'Getting index: {index}')
            if self.diffusion_steps:
                pose_mi, pose_gt = self.get_diff_pose_input_gt(
                    data, 
                    parent_final_pcd, child_final_pcd)
            else:
                pose_mi, pose_gt = self.get_pose_input_gt(
                    data, 
                    parent_final_pcd, child_final_pcd)
            # pose_mi, pose_gt = self.get_pose_input_gt(
            #     data, 
            #     parent_final_pcd, child_final_pcd)
        if self.train_success:
            success_mi, success_gt = self.get_success_input_gt(
                data, 
                parent_final_pcd, child_final_pcd)
        
        valid_aff_sample = (not self.train_coarse_aff) or (len(voxel_mi.keys()) > 0)
        valid_pose_sample = (not self.train_refine_pose) or (len(pose_mi.keys()) > 0)
        valid_success_sample = (not self.train_success) or (len(success_mi.keys()) > 0)
        valid_sample = valid_aff_sample and valid_pose_sample and valid_success_sample

        if not valid_sample:
            print(f'[Data Loader] Invalid sample (index: {index})')
            if embed_invalid:
                print('Here with valid_sample False')
                from IPython import embed; embed()
            # self.not_success_idx.append(index)
            return self.get_item(np.random.randint(len(self.data)))

        return (voxel_mi, voxel_gt), (pose_mi, pose_gt), (success_mi, success_gt)

    def process_general_start_final_pcd(self, data: dict, parent_final_pcd: np.ndarray, child_final_pcd: np.ndarray) -> Tuple[np.ndarray]:

        if child_final_pcd.shape[0] == 0:
            # self.not_success_idx.append(index)
            # return self.get_item(np.random.randint(len(self.files)))
            raise ValueError(f'Child final point cloud shape was zero')


        if self.debug_viz:
            util.meshcat_pcd_show(
                self.mc_vis, 
                parent_final_pcd, 
                (255, 0, 0), 
                'scene/data_loader/process_general/parent_final_pcd_pre')
            util.meshcat_pcd_show(
                self.mc_vis, 
                child_final_pcd, 
                (0, 0, 255), 
                'scene/data_loader/process_general/child_final_pcd_pre')
        
        # if self.parent_shape_pcd_n is not None:
        #     parent_final_pcd = train_util.check_enough_points(parent_final_pcd, self.parent_shape_pcd_n)
        # else:
        #     parent_final_pcd = train_util.check_enough_points(parent_final_pcd, self.shape_pcd_n)

        # if self.child_shape_pcd_n is not None:
        #     child_final_pcd = train_util.check_enough_points(child_final_pcd, self.child_shape_pcd_n)
        # else:
        #     child_final_pcd = train_util.check_enough_points(child_final_pcd, self.shape_pcd_n)

        parent_final_pcd = train_util.check_enough_points(parent_final_pcd, self.parent_shape_pcd_n)
        child_final_pcd = train_util.check_enough_points(child_final_pcd, self.child_shape_pcd_n)

        if self.debug_viz:
            util.meshcat_pcd_show(
                self.mc_vis, 
                parent_final_pcd, 
                (255, 0, 0), 
                'scene/data_loader/process_general/parent_final_pcd_post')
            util.meshcat_pcd_show(
                self.mc_vis, 
                child_final_pcd, 
                (0, 0, 255), 
                'scene/data_loader/process_general/child_final_pcd_post')

        if self.apply_pcd_aug and (np.random.random() > (1 - self.pcd_aug_prob)):

            parent_final_pcd_aug, child_final_pcd_aug = self.apply_general_pcd_aug(
                data, parent_final_pcd, child_final_pcd)

            parent_final_pcd = parent_final_pcd_aug
            child_final_pcd = child_final_pcd_aug

        # rpy_scale = self.small_rot_scale if self.use_small_rot_scale else self.rot_scale
        # trans_scale = self.small_trans_scale if self.use_small_trans_scale else self.trans_scale

        # small_rpy = train_util.sample_rot_perturb(scale=rpy_scale)
        # small_trans = train_util.sample_trans_perturb(scale=trans_scale)
        # small_rotmat = R.from_euler('xyz', small_rpy).as_matrix()
        # 
        # if self.single_offset_debug:
        #     if self.single_trans:
        #         small_trans = self.single_small_trans
        #     else:
        #         small_trans = random.sample(self.set_small_trans, 1)[0]
        # if self.single_rot_offset_debug:
        #     if self.single_rot:
        #         small_rotmat = self.single_small_rotmat
        #     else:
        #         small_rotmat = random.sample(self.set_small_rotmat, 1)[0]

        # # apply rotation
        # child_pcd_cent_rot = util.rotate_pcd_center(child_final_pcd, leave_centered=True)

        # # with some probability, initialize the child object near the center of the parent object
        # if np.random.random() > (1 - self.parent_cent_offset_prob):
        #     small_trans = np.mean(parent_final_pcd, axis=0) - child_pcd_mean + (np.random.random(3) - 0.5) * self.parent_cent_offset_scale

        # # create the start point clouds
        # child_start_pcd = child_pcd + child_pcd_mean + small_trans
        # parent_start_pcd = copy.deepcopy(parent_final_pcd)
        
        # parent_start_pcd, rix_p = util.downsample_pcd_perm(parent_start_pcd, return_perm=True)
        # child_start_pcd, rix_c = util.downsample_pcd_perm(child_start_pcd, return_perm=True)
        # parent_final_pcd = util.downsample_pcd_perm(parent_final_pcd, perm=rix_p)
        # child_final_pcd = util.downsample_pcd_perm(child_final_pcd, perm=rix_c)

        # return the trans/rot we used to perturb from good final pose
        # perturb_dict = dict(trans=small_trans, rot_mat=small_rotmat)

        if 'cabinet' in self.data_args.dataset_path and 'can' in self.data_args.dataset_path:
            if np.random.random() > 0.5:
                # augment to make sure we have examples where can is upright and upside down (relative to visible surface)
                child_final_pcd = util.rotate_pcd_center(
                    child_final_pcd, 
                    R.from_euler('xyz', [np.pi, 0, 0]).as_matrix(), 
                    leave_centered=False,
                    pcd_mean=data['multi_obj_final_obj_pose'].item()['child'][:3])

        return parent_final_pcd, child_final_pcd

    def apply_general_pcd_aug(self, data: dict, parent_pcd: np.ndarray, child_pcd: np.ndarray, 
                              min_p_pts: int=None, min_c_pts: int=None) -> Tuple[np.ndarray]:

        frame = data['multi_obj_part_pose_dict'].item()['parent_part_world']
        frame_pos = frame[:-1, -1]

        # pn_cams = np.random.randint(2, self.pcd_aug_n_cams+1)
        # cn_cams = np.random.randint(2, self.pcd_aug_n_cams+1)

        if self.full_pcd_aug:
            # sometimes use random occlusions using dropped out cameras from random poses
            p_rnd_occ = np.random.random() > 0.5
            c_rnd_occ = np.random.random() > 0.5
            pn_cams = np.random.randint(2, 5)
            cn_cams = np.random.randint(2, 5)
        else:
            p_rnd_occ = c_rnd_occ = False
            pn_cams = cn_cams = 4

        parent_pcd_aug = pcd_aug_util.pcd_aug_full(
            parent_pcd, 
            rot_grid=self.rot_grid, 
            deform_about_point=frame_pos,
            rnd_occlusion=p_rnd_occ,
            cut_plane=False,
            per_point_noise=True,
            apply_deformation=False,
            uniform_scaling=False,
            per_point_noise_std=self.pcd_aug_pp_std,
            n_cams=pn_cams,
            min_pts=min_p_pts)
        child_pcd_aug = pcd_aug_util.pcd_aug_full(
            child_pcd, 
            rot_grid=self.rot_grid, 
            deform_about_point=frame_pos,
            rnd_occlusion=c_rnd_occ,
            cut_plane=False,
            per_point_noise=True,
            apply_deformation=False,
            uniform_scaling=False,
            per_point_noise_std=self.pcd_aug_pp_std,
            n_cams=cn_cams,
            min_pts=min_c_pts)
        return parent_pcd_aug, child_pcd_aug

    def sample_pose_perturbation(self, parent_final_pcd: np.ndarray, child_final_pcd: np.ndarray, 
                                 normal: bool=False, start_scene_bb: bool=False) -> Tuple[Tuple[np.ndarray]]:
        rpy_scale = self.small_rot_scale if self.use_small_rot_scale else self.rot_scale
        trans_scale = self.small_trans_scale if self.use_small_trans_scale else self.trans_scale

        small_rpy = train_util.sample_rot_perturb(scale=rpy_scale, normal=normal)
        small_trans = train_util.sample_trans_perturb(scale=trans_scale, normal=normal)
        small_rotmat = R.from_euler('xyz', small_rpy).as_matrix()
        
        # if self.single_offset_debug:
        #     if self.single_trans:
        #         small_trans = self.single_small_trans
        #     else:
        #         small_trans = random.sample(self.set_small_trans, 1)[0]
        # if self.single_rot_offset_debug:
        #     if self.single_rot:
        #         small_rotmat = self.single_small_rotmat
        #     else:
        #         small_rotmat = random.sample(self.set_small_rotmat, 1)[0]

        # apply rotation
        child_pcd_mean = np.mean(child_final_pcd, axis=0)
        child_pcd_cent_rot = util.rotate_pcd_center(child_final_pcd, small_rotmat, leave_centered=True)

        # with some probability, initialize the child object near the center of the parent object
        if np.random.random() > (1 - self.parent_cent_offset_prob):
            small_trans = np.mean(parent_final_pcd, axis=0) - child_pcd_mean 
            small_trans = small_trans + (np.random.random(3) - 0.5) * self.parent_cent_offset_scale

        # with some probability, initialize the child object at a random point inside the parent bounding box
        if np.random.random() > (1 - self.rnd_parent_pt_offset_prob):
            bb = trimesh.PointCloud(parent_final_pcd).bounding_box_oriented
            pt = bb.sample_volume(1)[0]
            small_trans = pt - np.mean(child_final_pcd, axis=0) 

        if np.random.random() > (1 - self.rnd_parent_pcd_pt_offset_prob):
            idx = np.random.randint(parent_final_pcd.shape[0])
            pt = parent_final_pcd[idx]
            small_trans = pt - np.mean(child_final_pcd, axis=0) 
        
        # if we specify to start in the parent bounding box, overwrite with sample from bounding box
        if start_scene_bb:
            max_pt = parent_final_pcd.max(0)
            min_pt = parent_final_pcd.min(0)
            pt = np.random.random(3) * (max_pt - min_pt) + min_pt
            # bb = trimesh.PointCloud(parent_final_pcd).bounding_box_oriented
            # pt = bb.sample_volume(1)[0]
            small_trans = pt - np.mean(child_final_pcd, axis=0) 
            small_rotmat = self.rot_grid[np.random.randint(self.rot_grid.shape[0])]
            child_pcd_cent_rot = util.rotate_pcd_center(child_final_pcd, small_rotmat, leave_centered=True)

        # create the start point clouds
        child_start_pcd = child_pcd_cent_rot + child_pcd_mean + small_trans
        parent_start_pcd = copy.deepcopy(parent_final_pcd)

        return (parent_start_pcd, child_start_pcd), (small_rotmat, small_trans)

    def get_rotation_grid_index(self, rot_mat: np.ndarray, return_closest_mat: bool=False) -> Union[int, Tuple[Union[int, np.ndarray]]]:
        # multiply with each in rot_grid

        rot_mat_mult = np.matmul(np.linalg.inv(rot_mat), self.rot_grid)
        norms = np.linalg.norm((np.eye(3) - rot_mat_mult), ord='fro', axis=(1, 2))

        min_idx = np.argmin(norms)
        closest_rotmat = self.rot_grid[min_idx]

        if self.debug_viz:
            # print('here to get rotation_grid_index')
            # from IPython import embed; embed()
            # from pdb import set_trace; set_trace()
            # from rpdiff.utils.fork_pdb import ForkablePdb; ForkablePdb().set_trace()

            original_pt = np.array([1, 0, 0]).reshape(3, 1)
            rot_pts = np.matmul(self.rot_grid, original_pt).squeeze()

            input_rot_pt = np.matmul(rot_mat, original_pt).reshape(1, 3)

            offset = np.array([1.5, 0.0, 0.0])

            util.meshcat_pcd_show(self.mc_vis, rot_pts + offset, (0, 0, 0), name='scene/dataloader/rot_grid/rot_grid_pts', size=0.015)
            util.meshcat_pcd_show(self.mc_vis, input_rot_pt + offset, (0, 0, 255), 'scene/dataloader/rot_grid/input_rot_pt', size=0.025)
        
        if return_closest_mat:
            return min_idx, closest_rotmat
        return min_idx

    def get_euler_onehot(self, euler_angles: np.ndarray, 
                         bins_per_axis: int=None, return_closest_euler: bool=False) -> Union[dict, Tuple[Union[dict, np.ndarray]]]:
        if bins_per_axis is None:
            bins_per_axis = self.data_args.euler_bins_per_axis

        onehot_dict = {}
        onehot_dict['x'] = np.zeros(bins_per_axis)
        onehot_dict['y'] = np.zeros(bins_per_axis)
        onehot_dict['z'] = np.zeros(bins_per_axis)
        
        if self.euler_rot_disc is not None:
            euler_rot_disc = np.linspace(-np.pi, np.pi, bins_per_axis)
            self.euler_rot_disc = copy.deepcopy(euler_rot_disc)
        else:
            euler_rot_disc = self.euler_rot_disc

        for i, k in enumerate(onehot_dict.keys()):
            min_idx = np.argmin(np.sqrt((euler_angles[i] - euler_rot_disc)**2))
            onehot_dict[k][min_idx] = 1.0

        if return_closest_euler:
            return onehot_dict, closest_euler
        return onehot_dict

    def get_aff_input_gt(self, data: dict, 
                         parent_final_pcd: np.ndarray, child_final_pcd: np.ndarray) -> Tuple[dict]:

        mc_load_name = 'scene/dataio/aff'
        
        # configs for coarse affordance
        aff_args = self.data_args.coarse_aff

        # get pcd n for pcd aug and downsampling
        shape_pcd_n = self.shape_pcd_n
        parent_shape_pcd_n = self.parent_shape_pcd_n
        child_shape_pcd_n = self.child_shape_pcd_n
        if aff_args.shape_pcd_n is not None:
            shape_pcd_n = aff_args.shape_pcd_n
        if aff_args.parent_shape_pcd_n is not None:
            parent_shape_pcd_n = aff_args.parent_shape_pcd_n
        if aff_args.child_shape_pcd_n is not None:
            child_shape_pcd_n = aff_args.child_shape_pcd_n

        # pcd aug
        apply_pcd_aug = self.apply_pcd_aug
        if aff_args.aug.apply_pcd_aug is not None:
            apply_pcd_aug = aff_args.aug.apply_pcd_aug

        pcd_aug_prob = self.pcd_aug_prob
        if aff_args.aug.pcd_aug_prob is not None:
            pcd_aug_prob = aff_args.aug.pcd_aug_prob

        if apply_pcd_aug and (np.random.random() > (1 - pcd_aug_prob)):

            parent_final_pcd_aug, child_final_pcd_aug = self.apply_general_pcd_aug(
                data, parent_final_pcd, child_final_pcd, min_p_pts=parent_shape_pcd_n, min_c_pts=child_shape_pcd_n)

            parent_final_pcd = parent_final_pcd_aug
            child_final_pcd = child_final_pcd_aug

        # rot aug
        rot_aug = aff_args.aug.rot_aug

        if rot_aug is not None:
            if rot_aug == 'rot':
                random_large_rot = self.rot_grid[np.random.randint(self.rot_grid.shape[0])]
                pcd_ref2 = child_final_pcd
            elif rot_aug == 'yaw':
                random_yaw = np.array([0, 0, np.random.random() * 2 * np.pi])
                random_large_rot = R.from_euler('xyz', random_yaw).as_matrix()
                pcd_ref2 = parent_final_pcd
            elif rot_aug == 'yaw_rot':
                # yaw_rot mean only yaw with a very small amount of additional roll/pitch
                random_yaw_pr = np.array([
                    np.deg2rad(30)*(np.random.random() - 0.5), 
                    np.deg2rad(30)*(np.random.random() - 0.5), 
                    np.random.random() * 2 * np.pi])
                random_large_rot = R.from_euler('xyz', random_yaw_pr).as_matrix()
                pcd_ref2 = parent_final_pcd
            parent_final_pcd = util.rotate_pcd_center(parent_final_pcd, random_large_rot, pcd_ref=pcd_ref2)
            child_final_pcd = util.rotate_pcd_center(child_final_pcd, random_large_rot, pcd_ref=pcd_ref2)

        start_pcds, perturb_pose = self.sample_pose_perturbation(parent_final_pcd, child_final_pcd)
        parent_start_pcd, child_start_pcd = start_pcds
        small_rotmat, small_trans = perturb_pose
        
        # downsampling
        parent_start_pcd_dense = util.downsample_pcd_perm(parent_start_pcd, 4096)
        parent_start_pcd, rix_p = util.downsample_pcd_perm(parent_start_pcd, parent_shape_pcd_n, return_perm=True)
        child_start_pcd, rix_c = util.downsample_pcd_perm(child_start_pcd, child_shape_pcd_n, return_perm=True)
        parent_final_pcd = util.downsample_pcd_perm(parent_final_pcd, parent_shape_pcd_n, rix=rix_p)
        child_final_pcd = util.downsample_pcd_perm(child_final_pcd, child_shape_pcd_n, rix=rix_c)
        
        # get ground truth voxel position
        scene_mean = np.array(aff_args.scene_mean)
        scene_mean[2] = np.min(parent_start_pcd[:, 2])
        scene_extents = aff_args.scene_extents
        scene_scale = 1 / np.max(scene_extents)

        # parent_bb = trimesh.PointCloud(parent_start_pcd).bounding_box
        # scene_mean = copy.deepcopy(np.asarray(parent_bb.centroid))
        # scene_extents = copy.deepcopy(np.asarray(parent_bb.extents))
        # scene_scale = 1 / np.max(scene_extents)

        # parent point cloud goes through conv net to get feat grid, gets evaluated at points on child pcd
        parent_pcd_scene_norm = (parent_start_pcd - scene_mean) * scene_scale
        child_final_pcd_scene_norm = (child_final_pcd - scene_mean) * scene_scale
        child_start_pcd_scene_norm = (child_start_pcd - scene_mean) * scene_scale

        # get the ground truth for which voxels contain any of the points, and those that contain the centroid
        parent_pcd_mean_norm = np.mean(parent_pcd_scene_norm, axis=0)
        # parent_pcd_mean_norm = task_relevant_frame_pos_norm
        child_start_pcd_mean_norm = np.mean(child_start_pcd_scene_norm, axis=0)
        child_final_pcd_mean_norm = np.mean(child_final_pcd_scene_norm, axis=0)
        
        try:
            ppcd_mean_raster_index = three_util.coordinate2index_np(
                three_util.normalize_3d_coordinate_np(parent_pcd_mean_norm.reshape(1, -1)),
                self.reso_grid,
                '3d').squeeze()
            ppcd_mean_raster_coord = self.raster_pts[ppcd_mean_raster_index]
            cpcd_final_mean_raster_index = three_util.coordinate2index_np(
                three_util.normalize_3d_coordinate_np(child_final_pcd_mean_norm.reshape(1, -1)),
                self.reso_grid,
                '3d').squeeze()
            cpcd_final_mean_raster_coord = self.raster_pts[cpcd_final_mean_raster_index]

            ppcd_raster_index = three_util.coordinate2index_np(
                three_util.normalize_3d_coordinate_np(parent_pcd_scene_norm),
                self.reso_grid,
                '3d').squeeze()
            ppcd_raster_coords = self.raster_pts[ppcd_raster_index]
            cpcd_final_raster_index = three_util.coordinate2index_np(
                three_util.normalize_3d_coordinate_np(child_final_pcd_scene_norm),
                self.reso_grid,
                '3d').squeeze()
            cpcd_final_raster_coords = self.raster_pts[cpcd_final_raster_index]
        except Exception as e:
            print(f'[Data loader exception, trying again] {e}')
            # return self.get_item(np.random.randint(len(self.files)))
            return self.get_item(np.random.randint(len(self.data)))
        
        if self.debug_viz:
            util.meshcat_pcd_show(self.mc_vis, parent_start_pcd, (255, 0, 0), f'{mc_load_name}/parent_start_pcd')
            util.meshcat_pcd_show(self.mc_vis, child_start_pcd, (0, 0, 255), f'{mc_load_name}/child_start_pcd')
            util.meshcat_pcd_show(self.mc_vis, parent_pcd_scene_norm, (255, 0, 0), f'{mc_load_name}/parent_pcd_scene_norm')
            util.meshcat_pcd_show(self.mc_vis, child_start_pcd_scene_norm, (0, 0, 255), f'{mc_load_name}/child_start_pcd_scene_norm')
            util.meshcat_pcd_show(self.mc_vis, child_final_pcd_scene_norm, (0, 0, 255), f'{mc_load_name}/child_final_pcd_scene_norm')
            util.meshcat_pcd_show(self.mc_vis, ppcd_raster_coords, (0, 255, 255), f'{mc_load_name}/parent_raster_coords')
            util.meshcat_pcd_show(self.mc_vis, cpcd_final_raster_coords, (255, 0, 255), f'{mc_load_name}/child_final_raster_coords')

        # get the index on the discrete grid, corresponding to the rotation
        rot_mat_grid_index = self.get_rotation_grid_index(np.linalg.inv(small_rotmat))
        euler_rot = R.from_matrix(np.linalg.inv(small_rotmat)).as_euler('ZYX')
        euler_onehot = self.get_euler_onehot(euler_rot)

        self.aff_out_viz = dict(
            parent_start_pcd=parent_start_pcd,
            child_start_pcd=child_start_pcd,
            parent_pcd_scene_norm=parent_pcd_scene_norm,
            child_final_pcd_scene_norm=child_final_pcd_scene_norm,
            ppcd_raster_coords=ppcd_raster_coords,
            cpcd_final_raster_coords=cpcd_final_raster_coords,
            scene_mean=scene_mean,
            scene_scale=scene_scale
        )
        
        # pass in the scene-normalized point clouds, and save the scene normalization parameters
        aff_mi = dict(
            parent_start_pcd_sn=parent_pcd_scene_norm,
            child_start_pcd_sn=child_start_pcd_scene_norm,
            parent_start_pcd=util.center_pcd(parent_start_pcd), 
            parent_start_pcd_dense=util.center_pcd(parent_start_pcd_dense), 
            child_start_pcd=util.center_pcd(child_start_pcd),
            parent_start_pcd_mean_sn=parent_pcd_mean_norm,
            child_start_pcd_mean_sn=child_start_pcd_mean_norm,
            parent_start_pcd_mean=np.mean(parent_start_pcd, axis=0),
            child_start_pcd_mean=np.mean(child_start_pcd, axis=0),
            scene_mean=scene_mean,
            scene_scale=scene_scale)

        aff_gt = dict(
            child_centroid_voxel_index=cpcd_final_mean_raster_index,
            child_centroid_voxel_coord=cpcd_final_mean_raster_coord,
            trans=(-1.0 * small_trans),
            rot_mat=np.linalg.inv(small_rotmat),
            rot_mat_grid_index=rot_mat_grid_index,
            euler_onehot=euler_onehot,
            child_final_pcd_sn=child_final_pcd_scene_norm,
            child_final_pcd=child_final_pcd)

        return aff_mi, aff_gt

    def get_pose_input_gt(self, data: dict, 
                          parent_final_pcd: np.ndarray, child_final_pcd: np.ndarray) -> Tuple[dict]:
        
        mc_load_name = 'scene/dataio/pose'

        # get pose args
        pose_args = self.data_args.refine_pose

        if self.debug_viz:
            util.meshcat_pcd_show(self.mc_vis, parent_final_pcd, (255, 0, 0), f'{mc_load_name}/parent_final_pcd_pre_rot')
            util.meshcat_pcd_show(self.mc_vis, child_final_pcd, (0, 0, 255), f'{mc_load_name}/child_final_pcd_pre_rot')
            # util.meshcat_pcd_show(self.mc_vis, parent_full_pcd, (255, 0, 0), 'scene/parent_final_pcd_pre_rot_full')
            # util.meshcat_pcd_show(self.mc_vis, child_full_pcd, (0, 0, 255), 'scene/child_final_pcd_pre_rot_full')

        # get n points for pcd aug and downsampling
        shape_pcd_n = self.shape_pcd_n
        parent_shape_pcd_n = self.parent_shape_pcd_n
        child_shape_pcd_n = self.child_shape_pcd_n
        if pose_args.shape_pcd_n is not None:
            shape_pcd_n = pose_args.shape_pcd_n
        if pose_args.parent_shape_pcd_n is not None:
            parent_shape_pcd_n = pose_args.parent_shape_pcd_n
        if pose_args.child_shape_pcd_n is not None:
            child_shape_pcd_n = pose_args.child_shape_pcd_n
        
        # pcd aug
        apply_pcd_aug = self.apply_pcd_aug
        if pose_args.aug.apply_pcd_aug is not None:
            apply_pcd_aug = pose_args.aug.apply_pcd_aug

        pcd_aug_prob = self.pcd_aug_prob
        if pose_args.aug.pcd_aug_prob is not None:
            pcd_aug_prob = pose_args.aug.pcd_aug_prob

        if apply_pcd_aug and (np.random.random() > (1 - pcd_aug_prob)):

            parent_final_pcd_aug, child_final_pcd_aug = self.apply_general_pcd_aug(
                data, parent_final_pcd, child_final_pcd, min_p_pts=parent_shape_pcd_n, min_c_pts=child_shape_pcd_n)

            parent_final_pcd = parent_final_pcd_aug
            child_final_pcd = child_final_pcd_aug

        # rot aug
        rot_aug = pose_args.aug.rot_aug

        if rot_aug is not None:
            if rot_aug == 'rot':
                random_large_rot = self.rot_grid[np.random.randint(self.rot_grid.shape[0])]
                pcd_ref2 = child_final_pcd
            elif rot_aug == 'yaw':
                random_yaw = np.array([0, 0, np.random.random() * 2 * np.pi])
                random_large_rot = R.from_euler('xyz', random_yaw).as_matrix()
                pcd_ref2 = parent_final_pcd
            elif rot_aug == 'yaw_rot':
                # yaw_rot mean only yaw with a very small amount of additional roll/pitch
                random_yaw_pr = np.array([
                    np.deg2rad(30)*(np.random.random() - 0.5), 
                    np.deg2rad(30)*(np.random.random() - 0.5), 
                    np.random.random() * 2 * np.pi])
                random_large_rot = R.from_euler('xyz', random_yaw_pr).as_matrix()
                pcd_ref2 = parent_final_pcd
            parent_final_pcd = util.rotate_pcd_center(parent_final_pcd, random_large_rot, pcd_ref=pcd_ref2)
            child_final_pcd = util.rotate_pcd_center(child_final_pcd, random_large_rot, pcd_ref=pcd_ref2)
        
        if pose_args.child_start_ori_init:
            # sample trans perturbation, use demonstration to get initial orientation
            child_start_pose = np.asarray(data['multi_obj_start_obj_pose'].item()['child'])
            child_final_pose = np.asarray(data['multi_obj_final_obj_pose'].item()['child'])
            if child_start_pose.ndim > 1:
                child_start_pose = child_start_pose[0]
            if child_final_pose.ndim > 1:
                child_final_pose = child_final_pose[0]
            final_to_start = np.matmul(
                util.matrix_from_list(child_start_pose),
                np.linalg.inv(util.matrix_from_list(child_final_pose)))
            small_rotmat = final_to_start[:-1, :-1]

            start_pcds, perturb_pose = self.sample_pose_perturbation(parent_final_pcd, child_final_pcd)
            parent_start_pcd, _ = start_pcds
            _, small_trans = perturb_pose

            child_start_pcd = util.rotate_pcd_center(child_final_pcd, small_rotmat, pcd_ref=child_final_pcd)
            child_start_pcd = child_start_pcd + small_trans
        else:
            # sample perturbation
            start_pcds, perturb_pose = self.sample_pose_perturbation(parent_final_pcd, child_final_pcd)
            parent_start_pcd, child_start_pcd = start_pcds
            small_rotmat, small_trans = perturb_pose

        # form the cropped parent point cloud point clouds here
        # get the mean of the perturbed child point cloud
        start_child_mean = np.mean(child_start_pcd, axis=0)

        # cropping
        max_length = self.max_length
        xmin, xmax = start_child_mean[0] - max_length, start_child_mean[0] + max_length
        ymin, ymax = start_child_mean[1] - max_length, start_child_mean[1] + max_length
        zmin, zmax = start_child_mean[2] - max_length, start_child_mean[2] + max_length
        if self.parent_crop: # self.parent_crop_refine
            cropped_parent_start_pcd = util.crop_pcd(
                parent_start_pcd, 
                x=[xmin, xmax],
                y=[ymin, ymax],
                z=[zmin, zmax])
        else:
            cropped_parent_start_pcd = parent_start_pcd

        if cropped_parent_start_pcd.shape[0] == 0:
            print(f'[Refine pose data loader] Parent point cloud crop led to zero points')
            return {}, {}

        if self.debug_viz:
            util.meshcat_pcd_show(self.mc_vis, parent_start_pcd, (255, 0, 0), f'{mc_load_name}/parent_start_pcd_pre_ds')
            util.meshcat_pcd_show(self.mc_vis, child_start_pcd, (0, 0, 255), f'{mc_load_name}/child_start_pcd_pre_ds')
            util.meshcat_pcd_show(self.mc_vis, child_final_pcd, (0, 255, 255), f'{mc_load_name}/child_final_pcd_pre_ds')
            # util.meshcat_trimesh_show(self.mc_vis, f'{mc_load_name}/child_bb', child_bb, opacity=0.3)
            util.meshcat_pcd_show(self.mc_vis, cropped_parent_start_pcd, (0, 255, 0), f'{mc_load_name}/cropped_parent_start_pcd_pre_ds')

        # downsampling
        cropped_parent_start_pcd, rix_p = util.downsample_pcd_perm(cropped_parent_start_pcd, parent_shape_pcd_n, return_perm=True)
        parent_final_pcd, rix_p_full = util.downsample_pcd_perm(parent_final_pcd, parent_shape_pcd_n, return_perm=True)
        child_start_pcd, rix_c = util.downsample_pcd_perm(child_start_pcd, child_shape_pcd_n, return_perm=True)
        child_final_pcd = util.downsample_pcd_perm(child_final_pcd, child_shape_pcd_n, rix=rix_c)

        # add back points for regular batching
        cropped_parent_start_pcd = train_util.check_enough_points(cropped_parent_start_pcd, parent_shape_pcd_n)[:parent_shape_pcd_n]
        parent_final_pcd = train_util.check_enough_points(parent_final_pcd, parent_shape_pcd_n)[:parent_shape_pcd_n]
        child_start_pcd = train_util.check_enough_points(child_start_pcd, child_shape_pcd_n)[:child_shape_pcd_n]
        child_final_pcd = train_util.check_enough_points(child_final_pcd, child_shape_pcd_n)[:child_shape_pcd_n]
        
        if cropped_parent_start_pcd.shape[0] != parent_shape_pcd_n:
            print('\n\n\n!!! [pose pert] cropped_parent_start_pcd not enough points !!!\n\n\n')
        if parent_final_pcd.shape[0] != parent_shape_pcd_n:
            print('\n\n\n!!! [pose pert] parent_final_pcd not enough points !!!\n\n\n')
        if child_start_pcd.shape[0] != child_shape_pcd_n:
            print('\n\n\n!!! [pose pert] child_start_pcd not enough points !!!\n\n\n')
        if child_final_pcd.shape[0] != child_shape_pcd_n:
            print('\n\n\n!!! [pose pert] child_final_pcd not enough points !!!\n\n\n')

        if self.debug_viz:
            util.meshcat_pcd_show(self.mc_vis, parent_start_pcd, (255, 0, 0), f'{mc_load_name}/parent_start_pcd')
            util.meshcat_pcd_show(self.mc_vis, child_start_pcd, (0, 0, 255), f'{mc_load_name}/child_start_pcd')
            util.meshcat_pcd_show(self.mc_vis, child_final_pcd, (0, 255, 255), f'{mc_load_name}/child_final_pcd')
            # util.meshcat_trimesh_show(self.mc_vis, f'{mc_load_name}/child_bb', child_bb, opacity=0.3)
            util.meshcat_pcd_show(self.mc_vis, cropped_parent_start_pcd, (0, 255, 0), f'{mc_load_name}/cropped_parent_start_pcd')

        # policy model input
        # we provide the full parent point cloud and the segmented child point cloud
        # 3D coordinates of the child point cloud
        pose_mi = dict(
            parent_start_pcd=util.center_pcd(cropped_parent_start_pcd),
            child_start_pcd=util.center_pcd(child_start_pcd),
            parent_start_pcd_mean=np.mean(cropped_parent_start_pcd, axis=0),
            child_start_pcd_mean=np.mean(child_start_pcd, axis=0))
            # parent_start_pcd_offset=util.center_pcd(parent_start_pcd))

        pose_gt = dict(
            trans=(-1.0 * small_trans),
            rot_mat=np.linalg.inv(small_rotmat),
            parent_final_pcd=parent_final_pcd,  # parent_final_pcd[rix_p[:self.shape_pcd_n]]
            child_final_pcd=child_final_pcd,
            parent_start_pcd_mean=np.mean(cropped_parent_start_pcd, axis=0),
            child_start_pcd_mean=np.mean(child_start_pcd, axis=0),
            parent_final_pcd_mean=np.mean(parent_final_pcd, axis=0),
            child_final_pcd_mean=np.mean(child_final_pcd, axis=0),
            parent_to_child_offset=(np.mean(cropped_parent_start_pcd, axis=0) - np.mean(child_start_pcd, axis=0)))

        if pose_args.predict_offset:
            # obtain the ground truth offset 
            # trans_offset = data['trans_offset']
        
            if 'mug' in self.data_args.dataset_path and 'rack' in self.data_args.dataset_path:
                # placeholder while we get labels for ground truth trans offset (mug/rack-multi)
                mug_pose_list = data['multi_obj_final_obj_pose'].item()['child'][0]
                mug_pose_mat = util.matrix_from_list(mug_pose_list)
                offset_axis_pre_rot = mug_pose_mat[:-1, 0]  # mug x-axis, don't know forward/backward
                pt_near_handle = np.mean(child_final_pcd, axis=0) + (-1.0 * mug_pose_mat[:-1, 2] * 0.075)
                rack_to_handle_vec = pt_near_handle - np.mean(cropped_parent_start_pcd, axis=0)
                rack_to_handle_vec = rack_to_handle_vec / np.linalg.norm(rack_to_handle_vec)
                if np.dot(offset_axis_pre_rot, rack_to_handle_vec) < 0.0:
                    offset_axis_pre_rot = -1.0 * offset_axis_pre_rot
                if rot_aug is not None:
                    offset_axis_post_rot = np.matmul(random_large_rot, offset_axis_pre_rot.reshape(3, 1)).reshape(1, 3)
                else:
                    offset_axis_post_rot = offset_axis_pre_rot
            elif 'book' in self.data_args.dataset_path and 'shelf' in self.data_args.dataset_path:
                # placeholder while we get the labels for ground truth trans offset (book/bookshelf)
                shelf_pose_list = data['multi_obj_start_obj_pose'].item()['parent']
                shelf_pose_mat = util.matrix_from_list(shelf_pose_list)
                offset_axis_pre_rot = shelf_pose_mat[:-1, 0]  # shelf x-axis
                if rot_aug is not None:
                    offset_axis_post_rot = np.matmul(random_large_rot, offset_axis_pre_rot.reshape(3, 1)).reshape(1, 3)
                else:
                    offset_axis_post_rot = offset_axis_pre_rot
            else:
                raise ValueError('Unrecognized dataset path strings')

            if self.debug_viz:
                util.meshcat_pcd_show(
                    self.mc_vis, 
                    (offset_axis_pre_rot*0.25).reshape(1, 3), 
                    (255, 0, 0), name='scene/offset_axis_pre_rot', size=0.1)
                util.meshcat_pcd_show(
                    self.mc_vis, 
                    (offset_axis_post_rot*0.25).reshape(1, 3), 
                    (0, 255, 0), name='scene/offset_axis_post_rot', size=0.1)
                util.meshcat_pcd_show(
                    self.mc_vis, 
                    (child_final_pcd + offset_axis_post_rot*0.25),
                    (255, 0, 255), name='scene/child_final_pcd_offset', size=0.0025)

            trans_offset = offset_axis_post_rot
            pose_gt['trans_offset'] = trans_offset
        
        return pose_mi, pose_gt

    def get_success_input_gt(self, data: dict,
                             parent_final_pcd: np.ndarray, child_final_pcd: np.ndarray) -> Tuple[dict]:

        mc_load_name = 'scene/dataio/success'

        # get pose args
        success_args = self.data_args.success

        if self.debug_viz:
            util.meshcat_pcd_show(self.mc_vis, parent_final_pcd, (255, 0, 0), f'{mc_load_name}/parent_final_pcd')
            util.meshcat_pcd_show(self.mc_vis, child_final_pcd, (0, 0, 255), f'{mc_load_name}/child_final_pcd')
            # util.meshcat_pcd_show(self.mc_vis, parent_full_pcd, (255, 0, 0), 'scene/parent_final_pcd_pre_rot_full')
            # util.meshcat_pcd_show(self.mc_vis, child_full_pcd, (0, 0, 255), 'scene/child_final_pcd_pre_rot_full')
        
        # pcd aug
        apply_pcd_aug = self.apply_pcd_aug
        if success_args.aug.apply_pcd_aug is not None:
            apply_pcd_aug = success_args.aug.apply_pcd_aug

        pcd_aug_prob = self.pcd_aug_prob
        if success_args.aug.pcd_aug_prob is not None:
            pcd_aug_prob = success_args.aug.pcd_aug_prob

        # get required number of points before augmentation (ensure we don't remove too many)
        shape_pcd_n = self.shape_pcd_n
        parent_shape_pcd_n = self.parent_shape_pcd_n
        child_shape_pcd_n = self.child_shape_pcd_n
        if success_args.shape_pcd_n is not None:
            shape_pcd_n = success_args.shape_pcd_n
        if success_args.parent_shape_pcd_n is not None:
            parent_shape_pcd_n = success_args.parent_shape_pcd_n
        if success_args.child_shape_pcd_n is not None:
            child_shape_pcd_n = success_args.child_shape_pcd_n

        if apply_pcd_aug and (np.random.random() > (1 - pcd_aug_prob)):
            
            parent_final_pcd_aug, child_final_pcd_aug = self.apply_general_pcd_aug(
                data, parent_final_pcd, child_final_pcd, min_p_pts=parent_shape_pcd_n, min_c_pts=child_shape_pcd_n)

            parent_final_pcd = parent_final_pcd_aug
            child_final_pcd = child_final_pcd_aug

        # rot aug
        rot_aug = success_args.aug.rot_aug

        if rot_aug is not None:
            if rot_aug == 'rot':
                random_large_rot = self.rot_grid[np.random.randint(self.rot_grid.shape[0])]
                pcd_ref2 = child_final_pcd
            elif rot_aug == 'yaw':
                random_yaw = np.array([0, 0, np.random.random() * 2 * np.pi])
                random_large_rot = R.from_euler('xyz', random_yaw).as_matrix()
                pcd_ref2 = parent_final_pcd
            elif rot_aug == 'yaw_rot':
                # yaw_rot mean only yaw with a very small amount of additional roll/pitch
                random_yaw_pr = np.array([
                    np.deg2rad(30)*(np.random.random() - 0.5), 
                    np.deg2rad(30)*(np.random.random() - 0.5), 
                    np.random.random() * 2 * np.pi])
                random_large_rot = R.from_euler('xyz', random_yaw_pr).as_matrix()
                pcd_ref2 = parent_final_pcd
            parent_final_pcd = util.rotate_pcd_center(parent_final_pcd, random_large_rot, pcd_ref=pcd_ref2)
            child_final_pcd = util.rotate_pcd_center(child_final_pcd, random_large_rot, pcd_ref=pcd_ref2)

        # sample ground truth success/failure
        if np.random.random() > (1 - self.success_fail_prob):
            small_rpy = self.success_rot_scale * (np.random.random(3) - 0.5)
            small_trans = self.success_trans_scale * (np.random.random(3) - 0.5)
            success = True
        else:
            if np.random.random() > (1 - success_args.rot_only_prob):
                # large rot, small trans error
                small_rpy = (2 * self.fail_rot_scale * (np.random.random(3) - 0.5)) + self.fail_rot_min
                if np.random.random() > 0.5:
                    small_trans = (2 * self.fail_trans_min * (np.random.random(3) - 0.5))
                else:
                    small_trans = np.zeros(3)
            else:
                # large trans, small rot error
                if np.random.random() > 0.5:
                    small_rpy = (2 * self.fail_rot_min * (np.random.random(3) - 0.5))
                else:
                    small_rpy = np.zeros(3)

                small_trans = (2 * self.fail_trans_scale * (np.random.random(3) - 0.5)) + self.fail_trans_min
                if np.random.random() > 0.5:
                    small_trans = (2 * self.fail_trans_scale * (np.random.random(3) - 0.5)) + self.fail_trans_min
                else:
                    bb = trimesh.PointCloud(parent_final_pcd).bounding_box_oriented
                    pt = bb.sample_volume(1)[0]
                    small_trans = pt - np.mean(parent_final_pcd, axis=0) 

            success = False

        small_rotmat = R.from_euler('xyz', small_rpy).as_matrix()

        child_start_pcd = util.rotate_pcd_center(child_final_pcd, small_rotmat, pcd_ref=child_final_pcd)
        child_start_pcd = child_start_pcd + small_trans
        parent_start_pcd = copy.deepcopy(parent_final_pcd)

        # downsampling 
        parent_start_pcd, rix_p = util.downsample_pcd_perm(parent_start_pcd, parent_shape_pcd_n, return_perm=True)
        child_start_pcd, rix_c = util.downsample_pcd_perm(child_start_pcd, child_shape_pcd_n, return_perm=True)
        parent_final_pcd = util.downsample_pcd_perm(parent_final_pcd, parent_shape_pcd_n, rix=rix_p)
        child_final_pcd = util.downsample_pcd_perm(child_final_pcd, child_shape_pcd_n, rix=rix_c)

        if self.debug_viz:
            util.meshcat_pcd_show(self.mc_vis, parent_start_pcd, (255, 0, 0), f'{mc_load_name}/parent_start_pcd')
            util.meshcat_pcd_show(self.mc_vis, child_start_pcd, (0, 0, 255), f'{mc_load_name}/child_start_pcd')
            print(f'Succes: {success}\nTrans offset: {small_trans}\nRot offset: \n{small_rotmat}')

        # prepare model input (mi) dictionaries
        success_mi = dict(
            parent_start_pcd=util.center_pcd(parent_start_pcd),
            child_start_pcd=util.center_pcd(child_start_pcd),
            parent_start_pcd_mean=np.mean(parent_start_pcd, axis=0),
            child_start_pcd_mean=np.mean(child_start_pcd, axis=0))

        success_gt = dict(
            trans=-1.0*small_trans,
            rot_mat=np.linalg.inv(small_rotmat),
            parent_final_pcd=parent_final_pcd, 
            child_final_pcd=child_final_pcd,
            parent_start_pcd_mean=np.mean(parent_start_pcd, axis=0),
            child_start_pcd_mean=np.mean(child_start_pcd, axis=0),
            parent_final_pcd_mean=np.mean(parent_final_pcd, axis=0),
            child_final_pcd_mean=np.mean(child_final_pcd, axis=0),
            parent_to_child_offset=(np.mean(parent_start_pcd, axis=0) - np.mean(child_start_pcd, axis=0)),
            success=success)
        
        return success_mi, success_gt

    def get_diff_pose_input_gt(self, data: dict, 
                               parent_final_pcd: np.ndarray, child_final_pcd: np.ndarray) -> Tuple[dict]:
        
        mc_load_name = 'scene/dataio/pose_diff'

        # get pose args
        pose_args = self.data_args.refine_pose

        if self.debug_viz:
            # new
            # print("here in get_diff_pose_input_gt")
            # from IPython import embed; embed()
            # parent_final_pcd = o3d_fps(parent_final_pcd, int(parent_final_pcd.shape[0]/10))
            # child_final_pcd = o3d_fps(child_final_pcd, int(child_final_pcd.shape[0]/6))

            util.meshcat_pcd_show(self.mc_vis, parent_final_pcd, (255, 0, 0), f'{mc_load_name}/parent_final_pcd_pre_rot', size=self.MC_SIZE)
            util.meshcat_pcd_show(self.mc_vis, child_final_pcd, (0, 0, 255), f'{mc_load_name}/child_final_pcd_pre_rot', size=self.MC_SIZE)
            # util.meshcat_pcd_show(self.mc_vis, parent_full_pcd, (255, 0, 0), 'scene/parent_final_pcd_pre_rot_full')
            # util.meshcat_pcd_show(self.mc_vis, child_full_pcd, (0, 0, 255), 'scene/child_final_pcd_pre_rot_full')
            
            # new
            # util.meshcat_pcd_show(self.mc_vis, o3d_fps(parent_final_pcd), (0, 0, 0), f'{mc_load_name}/parent_final_pcd_pre_rot_ds', size=self.MC_SIZE)
            # util.meshcat_pcd_show(self.mc_vis, o3d_fps(child_final_pcd), (0, 0, 255), f'{mc_load_name}/child_final_pcd_pre_rot_ds', size=self.MC_SIZE)
            util.meshcat_pcd_show(self.mc_vis, parent_final_pcd, (0, 0, 0), f'{mc_load_name}/parent_final_pcd_pre_rot', size=self.MC_SIZE/2)
            util.meshcat_pcd_show(self.mc_vis, child_final_pcd, (0, 0, 255), f'{mc_load_name}/child_final_pcd_pre_rot', size=self.MC_SIZE)

        # get n points for pcd aug and downsampling
        shape_pcd_n = self.shape_pcd_n
        parent_shape_pcd_n = self.parent_shape_pcd_n
        child_shape_pcd_n = self.child_shape_pcd_n
        if pose_args.shape_pcd_n is not None:
            shape_pcd_n = pose_args.shape_pcd_n
        if pose_args.parent_shape_pcd_n is not None:
            parent_shape_pcd_n = pose_args.parent_shape_pcd_n
        if pose_args.child_shape_pcd_n is not None:
            child_shape_pcd_n = pose_args.child_shape_pcd_n
        
        # pcd aug
        apply_pcd_aug = self.apply_pcd_aug
        if pose_args.aug.apply_pcd_aug is not None:
            apply_pcd_aug = pose_args.aug.apply_pcd_aug

        pcd_aug_prob = self.pcd_aug_prob
        if pose_args.aug.pcd_aug_prob is not None:
            pcd_aug_prob = pose_args.aug.pcd_aug_prob

        if apply_pcd_aug and (np.random.random() > (1 - pcd_aug_prob)):

            parent_final_pcd_aug, child_final_pcd_aug = self.apply_general_pcd_aug(
                data, parent_final_pcd, child_final_pcd, min_p_pts=parent_shape_pcd_n, min_c_pts=child_shape_pcd_n)

            parent_final_pcd = parent_final_pcd_aug
            child_final_pcd = child_final_pcd_aug

        # rot aug
        rot_aug = pose_args.aug.rot_aug

        if rot_aug is not None:
            if rot_aug == 'rot':
                random_large_rot = self.rot_grid[np.random.randint(self.rot_grid.shape[0])]
                pcd_ref2 = child_final_pcd
            elif rot_aug == 'yaw':
                random_yaw = np.array([0, 0, np.random.random() * 2 * np.pi])
                random_large_rot = R.from_euler('xyz', random_yaw).as_matrix()
                pcd_ref2 = parent_final_pcd
            elif rot_aug == 'yaw_rot':
                # yaw_rot mean only yaw with a very small amount of additional roll/pitch
                random_yaw_pr = np.array([
                    np.deg2rad(30)*(np.random.random() - 0.5), 
                    np.deg2rad(30)*(np.random.random() - 0.5), 
                    np.random.random() * 2 * np.pi])
                random_large_rot = R.from_euler('xyz', random_yaw_pr).as_matrix()
                pcd_ref2 = parent_final_pcd
            parent_final_pcd = util.rotate_pcd_center(parent_final_pcd, random_large_rot, pcd_ref=pcd_ref2)
            child_final_pcd = util.rotate_pcd_center(child_final_pcd, random_large_rot, pcd_ref=pcd_ref2)
        
        if pose_args.child_start_ori_init:
            # sample trans perturbation, use demonstration to get initial orientation
            child_start_pose = np.asarray(data['multi_obj_start_obj_pose'].item()['child'])
            child_final_pose = np.asarray(data['multi_obj_final_obj_pose'].item()['child'])
            if child_start_pose.ndim > 1:
                child_start_pose = child_start_pose[0]
            if child_final_pose.ndim > 1:
                child_final_pose = child_final_pose[0]
            final_to_start = np.matmul(
                util.matrix_from_list(child_start_pose),
                np.linalg.inv(util.matrix_from_list(child_final_pose)))
            small_rotmat = final_to_start[:-1, :-1]

            start_pcds, perturb_pose = self.sample_pose_perturbation(parent_final_pcd, child_final_pcd)
            parent_start_pcd, _ = start_pcds
            _, small_trans = perturb_pose

            child_start_pcd = util.rotate_pcd_center(child_final_pcd, small_rotmat, pcd_ref=child_final_pcd)
            child_start_pcd = child_start_pcd + small_trans
        else:
            pass

        # create list of perturbation transformations
        indiv_perturb_pose_list = []
        indiv_full_perturb_pose_list = []
        cumul_perturb_pose_list = []
        cumul_full_perturb_pose_list = []

        diff_steps = self.data_args.refine_pose.n_diffusion_steps
        precise_prob_sampling = self.data_args.refine_pose.precise_diff_prob  # if true, be biased toward smaller steps (more precise - be closer to training on a single step of noise)
        
        tf_so_far = np.eye(4)
        child_pcd_so_far = child_final_pcd.copy()
        rotmat_so_far = np.eye(3)
        trans_so_far = np.zeros(3)
        
        if self.debug_viz:
            cmap = plt.get_cmap('inferno')
            color_list = cmap(np.linspace(0.1, 0.9, diff_steps+1, dtype=np.float32))[::-1]
            # RED = np.array([255, 20, 10])
            # BLUE = np.array([10, 20, 255])
            # color_list = np.linspace(RED, BLUE, diff_steps, dtype=np.float32)

            cmap2 = plt.get_cmap('Spectral')
            color_list2 = cmap2(np.linspace(0.1, 0.9, diff_steps+1, dtype=np.float32))[::-1]
        
        start_scene_bb = self.data_args.refine_pose.init_scene_bounding_box
        if self.data_args.refine_pose.interp_diffusion_traj: # sample large pert and interpolate

            start_pcds, perturb_pose = self.sample_pose_perturbation(parent_final_pcd, child_pcd_so_far, normal=False, start_scene_bb=start_scene_bb)

            # split up the tf
            rotmat_pert, trans_pert = perturb_pose
            quat_pert = R.from_matrix(rotmat_pert).as_quat()

            # interpolate the positions
            trans_interp = np.linspace(np.zeros(3), trans_pert, diff_steps+1)

            # interpolate the rotations
            slerp = Slerp(np.arange(2), R.from_quat([np.array([0, 0, 0, 1]), quat_pert]))
            interp_rots = slerp(np.linspace(0, 1, diff_steps+1))
            rotmat_interp = interp_rots.as_matrix()
            
            for d_idx in range(1, diff_steps+1):
                small_trans = trans_interp[d_idx] - trans_interp[d_idx-1]
                small_rotmat = np.matmul(rotmat_interp[d_idx], np.linalg.inv(rotmat_interp[d_idx-1]))
                
                tf_this_step_raw = np.eye(4); tf_this_step_raw[:-1, :-1] = small_rotmat; tf_this_step_raw[:-1, -1] = small_trans
                tf_this_step = util.form_tf_mat_cent_pcd_rot(tf_this_step_raw, child_pcd_so_far)
                tf_so_far = np.matmul(tf_this_step, tf_so_far)

                child_pcd_so_far = util.transform_pcd(child_pcd_so_far, tf_this_step)

                indiv_perturb_pose_list.append(tf_this_step_raw)
                cumul_perturb_pose_list.append(tf_so_far)
                
                # also save the "full" tf
                tf_this_step_full_raw = np.eye(4); tf_this_step_full_raw[:-1, :-1] = rotmat_interp[d_idx]; tf_this_step_full_raw[:-1, -1] = trans_interp[d_idx]
                tf_this_step_full = util.form_tf_mat_cent_pcd_rot(tf_this_step_full_raw, child_final_pcd)
                indiv_full_perturb_pose_list.append(tf_this_step_full_raw)
                cumul_full_perturb_pose_list.append(tf_this_step_full)

                if self.debug_viz:
                    color2 = tuple((color_list2[d_idx][:3] * 255).astype(np.uint8).tolist())
                    tf_child_start_pcd = util.transform_pcd(child_final_pcd, tf_so_far)
                    util.meshcat_pcd_show(self.mc_vis, tf_child_start_pcd, color2, f'{mc_load_name}/diff_pcds_so_far/pcd_{d_idx}', size=self.MC_SIZE)

        else: # random walk
            for d_idx in range(diff_steps):
                # sample perturbation
                start_pcds, perturb_pose = self.sample_pose_perturbation(parent_final_pcd, child_pcd_so_far, normal=True, start_scene_bb=start_scene_bb)
                # start_pcds, perturb_pose = self.sample_pose_perturbation(parent_final_pcd, child_final_pcd)
                parent_start_pcd, child_start_pcd = start_pcds
                # _, perturb_pose = self.sample_pose_perturbation(parent_final_pcd, child_final_pcd)
                small_rotmat, small_trans = perturb_pose

                rotmat_so_far = np.matmul(small_rotmat, rotmat_so_far)
                trans_so_far = small_trans + trans_so_far
                
                # tf_this_step = np.eye(4); tf_this_step[:-1, :-1] = small_rotmat; tf_this_step[:-1, -1] = small_trans
                # tf_so_far = np.matmul(tf_this_step, tf_so_far)

                tf_this_step_raw = np.eye(4); tf_this_step_raw[:-1, :-1] = small_rotmat; tf_this_step_raw[:-1, -1] = small_trans
                tf_this_step = util.form_tf_mat_cent_pcd_rot(tf_this_step_raw, child_pcd_so_far)
                tf_so_far = np.matmul(tf_this_step, tf_so_far)

                child_pcd_so_far = child_start_pcd.copy()

                indiv_perturb_pose_list.append(tf_this_step_raw)
                cumul_perturb_pose_list.append(tf_so_far)

                # also save the "full" tf
                tf_this_step_full_raw = np.eye(4); tf_this_step_full_raw[:-1, :-1] = rotmat_so_far; tf_this_step_full_raw[:-1, -1] = trans_so_far
                tf_this_step_full = util.form_tf_mat_cent_pcd_rot(tf_this_step_full_raw, child_final_pcd)
                indiv_full_perturb_pose_list.append(tf_this_step_full_raw)
                cumul_full_perturb_pose_list.append(tf_this_step_full)

                if self.debug_viz:
                    color = tuple((color_list[d_idx][:3] * 255).astype(np.uint8).tolist())
                    util.meshcat_pcd_show(self.mc_vis, child_start_pcd, color, f'{mc_load_name}/diff_pcds/pcd_{d_idx}', debug=False, size=self.MC_SIZE)

                    color2 = tuple((color_list2[d_idx][:3] * 255).astype(np.uint8).tolist())
                    child_final_mean = np.mean(child_final_pcd, axis=0)
                    # tf_child_start_pcd_rot = util.rotate_pcd_center(child_final_pcd, tf_so_far[:-1, :-1], leave_centered=True)
                    # tf_child_start_pcd = tf_child_start_pcd_rot + child_final_mean + tf_so_far[:-1, -1]
                    # tf_child_start_pcd = util.rotate_pcd_center(child_final_pcd, tf_so_far[:-1, :-1]) + tf_so_far[:-1, -1]
                    tf_child_start_pcd = util.transform_pcd(child_final_pcd, tf_so_far)
                    util.meshcat_pcd_show(self.mc_vis, tf_child_start_pcd, color2, f'{mc_load_name}/diff_pcds_so_far/pcd_{d_idx}', size=self.MC_SIZE)
        
        # sample the transition we want to train on
        if precise_prob_sampling:
            diff_vals = np.exp(-1.0*np.arange(diff_steps))
            total = diff_vals.sum()
            probs = diff_vals / total
            diff_sample_idx = np.where(np.random.multinomial(1, probs))[0][0]
        else:
            diff_sample_idx = np.random.randint(0, diff_steps)
        # diff_sample_idx = np.random.randint(0, diff_steps)

        start_pert_pose = cumul_perturb_pose_list[diff_sample_idx]  # this is the full cumulation, to get the proper start pcd
        target_pert_pose = indiv_perturb_pose_list[diff_sample_idx]  # this is the tf to be decomposed as (R_body, t)

        start_pert_full_pose = cumul_full_perturb_pose_list[diff_sample_idx]  # this is the full cumulation, to get the proper start pcd
        target_pert_full_pose = indiv_full_perturb_pose_list[diff_sample_idx]  # this is the tf to be decomposed as (R_body, t)
        
        if self.data_args.refine_pose.diffusion_full_pose_target: # sample large pert and interpolate
            small_rotmat = target_pert_full_pose[:-1, :-1]; small_trans = target_pert_full_pose[:-1, -1]
            start_small_rotmat = start_pert_pose[:-1, :-1]; start_small_trans = start_pert_pose[:-1, -1]
            to_start_tf_to_use = start_pert_full_pose.copy()
        else:
            small_rotmat = target_pert_pose[:-1, :-1]; small_trans = target_pert_pose[:-1, -1]
            start_small_rotmat = start_pert_pose[:-1, :-1]; start_small_trans = start_pert_pose[:-1, -1]
            to_start_tf_to_use = start_pert_pose.copy()
        # small_rotmat = target_pert_pose[:-1, :-1]; small_trans = target_pert_pose[:-1, -1]
        # start_small_rotmat = start_pert_pose[:-1, :-1]; start_small_trans = start_pert_pose[:-1, -1]

        # create the start point clouds
        # child_pcd_mean = np.mean(child_final_pcd, axis=0)
        # child_pcd_cent_rot = util.rotate_pcd_center(child_final_pcd, start_small_rotmat, leave_centered=True)
        # child_start_pcd = child_pcd_cent_rot + child_pcd_mean + start_small_trans
        child_start_pcd = util.transform_pcd(child_final_pcd, to_start_tf_to_use)
        parent_start_pcd = copy.deepcopy(parent_final_pcd)
        
        child_final_pcd_orig = child_final_pcd.copy()
        child_final_pcd = util.rotate_pcd_center(child_start_pcd, np.linalg.inv(small_rotmat)) + (-1.0 * small_trans)

        # form the cropped parent point cloud point clouds here
        # get the mean of the perturbed child point cloud
        start_child_mean = np.mean(child_start_pcd, axis=0)

        # cropping
        variable_size_crop = self.data_args.refine_pose.vary_crop_size_diffusion

        if variable_size_crop:
            min_crop_length = self.max_length
            # max_crop_length = np.linalg.norm([parent_final_pcd.max(0) - parent_final_pcd.min(0)], axis=1)[0] * 1.5
            max_crop_length = np.linalg.norm([parent_final_pcd.max(0) - parent_final_pcd.min(0)], axis=1)[0] / np.sqrt(2)
            # max_crop_length = np.linalg.norm([parent_final_pcd.max(0) - parent_final_pcd.min(0)], axis=1)[0]
            max_length = min_crop_length + ((diff_sample_idx / diff_steps)**2) * (max_crop_length - min_crop_length)
        else:
            max_length = self.max_length

        xmin, xmax = start_child_mean[0] - max_length, start_child_mean[0] + max_length
        ymin, ymax = start_child_mean[1] - max_length, start_child_mean[1] + max_length
        zmin, zmax = start_child_mean[2] - max_length, start_child_mean[2] + max_length
        if self.parent_crop: # self.parent_crop_refine
            cropped_parent_start_pcd = util.crop_pcd(
                parent_start_pcd, 
                x=[xmin, xmax],
                y=[ymin, ymax],
                z=[zmin, zmax])
        else:
            cropped_parent_start_pcd = parent_start_pcd

        if cropped_parent_start_pcd.shape[0] == 0:
            print(f'[Refine pose data loader] Parent point cloud crop led to zero points')
            return {}, {}

        if self.debug_viz:
            util.meshcat_pcd_show(self.mc_vis, parent_start_pcd, (255, 0, 0), f'{mc_load_name}/parent_start_pcd_pre_ds', size=self.MC_SIZE)
            util.meshcat_pcd_show(self.mc_vis, child_start_pcd, (0, 0, 255), f'{mc_load_name}/child_start_pcd_pre_ds', size=self.MC_SIZE)
            util.meshcat_pcd_show(self.mc_vis, child_final_pcd, (0, 255, 255), f'{mc_load_name}/child_final_pcd_pre_ds', size=self.MC_SIZE)
            # util.meshcat_trimesh_show(self.mc_vis, f'{mc_load_name}/child_bb', child_bb, opacity=0.3)
            util.meshcat_pcd_show(self.mc_vis, cropped_parent_start_pcd, (0, 255, 0), f'{mc_load_name}/cropped_parent_start_pcd_pre_ds', size=self.MC_SIZE)
            
            # new
            util.meshcat_pcd_show(self.mc_vis, cropped_parent_start_pcd, (255, 0, 0), f'{mc_load_name}/cropped_parent_start_pcd_pre_ds', size=self.MC_SIZE*1.5)

        # downsampling
        cropped_parent_start_pcd, rix_p = util.downsample_pcd_perm(cropped_parent_start_pcd, parent_shape_pcd_n, return_perm=True)
        parent_final_pcd, rix_p_full = util.downsample_pcd_perm(parent_final_pcd, parent_shape_pcd_n, return_perm=True)
        child_start_pcd, rix_c = util.downsample_pcd_perm(child_start_pcd, child_shape_pcd_n, return_perm=True)
        child_final_pcd = util.downsample_pcd_perm(child_final_pcd, child_shape_pcd_n, rix=rix_c)

        # add back points for regular batching
        cropped_parent_start_pcd = train_util.check_enough_points(cropped_parent_start_pcd, parent_shape_pcd_n)[:parent_shape_pcd_n]
        parent_final_pcd = train_util.check_enough_points(parent_final_pcd, parent_shape_pcd_n)[:parent_shape_pcd_n]
        child_start_pcd = train_util.check_enough_points(child_start_pcd, child_shape_pcd_n)[:child_shape_pcd_n]
        child_final_pcd = train_util.check_enough_points(child_final_pcd, child_shape_pcd_n)[:child_shape_pcd_n]
        
        if cropped_parent_start_pcd.shape[0] != parent_shape_pcd_n:
            print('\n\n\n!!! [pose pert] cropped_parent_start_pcd not enough points !!!\n\n\n')
        if parent_final_pcd.shape[0] != parent_shape_pcd_n:
            print('\n\n\n!!! [pose pert] parent_final_pcd not enough points !!!\n\n\n')
        if child_start_pcd.shape[0] != child_shape_pcd_n:
            print('\n\n\n!!! [pose pert] child_start_pcd not enough points !!!\n\n\n')
        if child_final_pcd.shape[0] != child_shape_pcd_n:
            print('\n\n\n!!! [pose pert] child_final_pcd not enough points !!!\n\n\n')

        if self.debug_viz:
            util.meshcat_pcd_show(self.mc_vis, parent_start_pcd, (255, 0, 0), f'{mc_load_name}/parent_start_pcd', size=self.MC_SIZE)
            util.meshcat_pcd_show(self.mc_vis, child_start_pcd, (0, 0, 255), f'{mc_load_name}/child_start_pcd', size=self.MC_SIZE)
            util.meshcat_pcd_show(self.mc_vis, child_final_pcd, (0, 255, 255), f'{mc_load_name}/child_final_pcd', size=self.MC_SIZE)
            # util.meshcat_trimesh_show(self.mc_vis, f'{mc_load_name}/child_bb', child_bb, opacity=0.3)
            util.meshcat_pcd_show(self.mc_vis, cropped_parent_start_pcd, (0, 255, 0), f'{mc_load_name}/cropped_parent_start_pcd', size=self.MC_SIZE)

        # policy model input
        # we provide the full parent point cloud and the segmented child point cloud
        # 3D coordinates of the child point cloud
        pose_mi = dict(
            parent_start_pcd=util.center_pcd(cropped_parent_start_pcd),
            child_start_pcd=util.center_pcd(child_start_pcd),
            parent_start_pcd_mean=np.mean(cropped_parent_start_pcd, axis=0),
            child_start_pcd_mean=np.mean(child_start_pcd, axis=0),
            diffusion_timestep=diff_sample_idx)
            # parent_start_pcd_offset=util.center_pcd(parent_start_pcd))

        pose_gt = dict(
            trans=(-1.0 * small_trans),
            rot_mat=np.linalg.inv(small_rotmat),
            parent_final_pcd=parent_final_pcd,  # parent_final_pcd[rix_p[:self.shape_pcd_n]]
            child_final_pcd=child_final_pcd,
            parent_start_pcd_mean=np.mean(cropped_parent_start_pcd, axis=0),
            child_start_pcd_mean=np.mean(child_start_pcd, axis=0),
            parent_final_pcd_mean=np.mean(parent_final_pcd, axis=0),
            child_final_pcd_mean=np.mean(child_final_pcd, axis=0),
            parent_to_child_offset=(np.mean(cropped_parent_start_pcd, axis=0) - np.mean(child_start_pcd, axis=0)))

        if pose_args.predict_offset:
            # obtain the ground truth offset 
            # trans_offset = data['trans_offset']
        
            if 'mug' in self.data_args.dataset_path and 'rack' in self.data_args.dataset_path:
                # placeholder while we get labels for ground truth trans offset (mug/rack-multi)
                mug_pose_list = data['multi_obj_final_obj_pose'].item()['child'][0]
                mug_pose_mat = util.matrix_from_list(mug_pose_list)
                offset_axis_pre_rot = mug_pose_mat[:-1, 0]  # mug x-axis, don't know forward/backward
                pt_near_handle = np.mean(child_final_pcd, axis=0) + (-1.0 * mug_pose_mat[:-1, 2] * 0.075)
                rack_to_handle_vec = pt_near_handle - np.mean(cropped_parent_start_pcd, axis=0)
                rack_to_handle_vec = rack_to_handle_vec / np.linalg.norm(rack_to_handle_vec)
                if np.dot(offset_axis_pre_rot, rack_to_handle_vec) < 0.0:
                    offset_axis_pre_rot = -1.0 * offset_axis_pre_rot
                if rot_aug is not None:
                    offset_axis_post_rot = np.matmul(random_large_rot, offset_axis_pre_rot.reshape(3, 1)).reshape(1, 3)
                else:
                    offset_axis_post_rot = offset_axis_pre_rot
            elif 'book' in self.data_args.dataset_path and 'shelf' in self.data_args.dataset_path:
                # placeholder while we get the labels for ground truth trans offset (book/bookshelf)
                shelf_pose_list = data['multi_obj_start_obj_pose'].item()['parent']
                shelf_pose_mat = util.matrix_from_list(shelf_pose_list)
                offset_axis_pre_rot = shelf_pose_mat[:-1, 0]  # shelf x-axis
                if rot_aug is not None:
                    offset_axis_post_rot = np.matmul(random_large_rot, offset_axis_pre_rot.reshape(3, 1)).reshape(1, 3)
                else:
                    offset_axis_post_rot = offset_axis_pre_rot
            else:
                raise ValueError('Unrecognized dataset path strings')

            if self.debug_viz:
                util.meshcat_pcd_show(
                    self.mc_vis, 
                    (offset_axis_pre_rot*0.25).reshape(1, 3), 
                    (255, 0, 0), name='scene/offset_axis_pre_rot', size=0.1)
                util.meshcat_pcd_show(
                    self.mc_vis, 
                    (offset_axis_post_rot*0.25).reshape(1, 3), 
                    (0, 255, 0), name='scene/offset_axis_post_rot', size=0.1)
                util.meshcat_pcd_show(
                    self.mc_vis, 
                    (child_final_pcd + offset_axis_post_rot*0.25),
                    (255, 0, 255), name='scene/child_final_pcd_offset', size=0.0025)

            trans_offset = offset_axis_post_rot
            pose_gt['trans_offset'] = trans_offset
        
        return pose_mi, pose_gt

    def __getitem__(self, index):
        return self.get_item(index)

