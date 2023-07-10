import os, os.path as osp
import argparse
from tqdm import tqdm 
import csv
import numpy as np
import trimesh
import trimesh.creation as cr
import time
from mesh_to_sdf import mesh_to_voxels
from multiprocessing import Pool
import meshcat

from airobot.utils import common

from rpdiff_robot.utils import util, path_util, trimesh_util
from rpdiff_robot.utils.mesh_util import inside_mesh

# from syn_bookshelf_cfg import get_syn_bookshelf_default_cfg
from syn_can_cabinet_cfg import get_syn_can_cabinet_default_cfg
from can_cabinet_dev import make_can_cabinet

VOXEL_RES = 128

parser = argparse.ArgumentParser()
parser.add_argument('--n_pts', type=int, default=100000)
parser.add_argument('--voxel_resolution', type=int, default=128)
# parser.add_argument('--voxel_pitch', type=float, default=0.005)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--mesh_save_dir', type=str, required=True)
parser.add_argument('--obj_name', type=str, required=True)
parser.add_argument('--can_obj_name', type=str, required=True)
parser.add_argument('--n_objs', type=int, default=200)
parser.add_argument('--difficulty', type=str, default='easy')
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--only_occ', action='store_true')
parser.add_argument('--only_center', action='store_true')
parser.add_argument('--dense', action='store_true')

args = parser.parse_args()

cfg = get_syn_can_cabinet_default_cfg()

mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
mc_vis['scene'].delete()

# path that we will save synthesized meshes in
mesh_save_dir = osp.join(path_util.get_rpdiff_obj_descriptions(), args.mesh_save_dir)
unnormalized_mesh_save_dir = osp.join(path_util.get_rpdiff_obj_descriptions(), args.mesh_save_dir + '_unnormalized')
open_slot_poses_dir = osp.join(unnormalized_mesh_save_dir, 'open_slot_poses')
unposed_can_meshes_dir = osp.join(unnormalized_mesh_save_dir, 'syn_cans_obj')

# path that  we will save occupancy data in
# occ_save_dir = osp.join(path_util.get_rpdiff_obj_descriptions(), args.occ_save_dir)

util.safe_makedirs(mesh_save_dir)
util.safe_makedirs(unnormalized_mesh_save_dir)
util.safe_makedirs(open_slot_poses_dir)
util.safe_makedirs(unposed_can_meshes_dir)
# util.safe_makedirs(occ_save_dir)


def rhl(bounds):
    high, low = max(bounds), min(bounds)
    sample = np.random.random() * (high - low) + low
    return sample


def get_raster_points(voxel_resolution):
    points = np.meshgrid(
        np.linspace(-0.5, 0.5, voxel_resolution),
        np.linspace(-0.5, 0.5, voxel_resolution),
        np.linspace(-0.5, 0.5, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    return points


def main_mp(args, cfg):
    obj_name = args.obj_name
    can_obj_name = args.can_obj_name
    n_objs = args.n_objs

    global_normalizing_factor = 0.0
    mp_args = [(obj_name, str(i), cfg) for i in range(n_objs)]
    for mp_arg in mp_args:
        obj_name,  obj_number, cfg = mp_arg

        # sample dimensions
        # bl = rhl(cfg.BASE_LENGTH_LOW_HIGH) 
        bl = 0.0
        bw = rhl(cfg.BASE_WIDTH_LOW_HIGH)
        bt = rhl(cfg.BASE_THICKNESS_LOW_HIGH)
        wt = rhl(cfg.WALL_THICKNESS_LOW_HIGH)
        # wh = rhl(cfg.WALL_HEIGHT_LOW_HIGH)
        wh = 0.0
        theta = rhl(cfg.WALL_THETA_LOW_HIGH)

        cr = cfg.CAN_RADIUS_LOW_HIGH
        ch = cfg.CAN_HEIGHT_LOW_HIGH

        # bkl, bkw, bkt = 0.18, 0.25, 0.04
        per_can_prob = rhl(cfg.PER_BOOK_PROB_LOW_HIGH)
        top_bool = np.random.random() > (1 - cfg.TOP_BOOL_PROB)
        eps = rhl(cfg.BOOK_SPACE_LOW_HIGH)
        n_shelves = np.random.randint(1, cfg.N_SHELVES_MAX)
        n_cans_per_shelf_max = np.random.randint(2, cfg.N_CANS_PER_SHELF_MAX+1)

        new_stack_can_prob = 0.75
        remove_stack_prob = 0.05
        nonuniform = True
        dense = args.dense

        # sample canshelf
        # print(f'Making canshelf with dimensions: {bl:.3f}, {bw:.3f}, {bt:.3f}, {wt:.3f}, {wh:.3f}, {theta:.3f}')
        try:
            full_can_and_cabinet_mesh, full_avail_can_top_poses, unposed_can_mesh = make_can_cabinet(
                bl=bl, bw=bw, bt=bt, n_shelves=n_shelves, n_cans_per_shelf_max=n_cans_per_shelf_max,
                cr=cr, ch=ch,
                wt=wt, wh=wh, top_bool=top_bool, per_can_prob=per_can_prob, eps=eps,
                remove_stack_prob=remove_stack_prob, new_stack_can_prob=new_stack_can_prob, 
                dense=dense, nonuniform=nonuniform,
                show=True, mc_vis=mc_vis)
        except ValueError:
            print(f'\n\n\nSkipping\n\n\n')
            continue
        
        # create mesh
        obj_mesh = full_can_and_cabinet_mesh
        # obj_mesh.vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
        new_obj_name = obj_name + '_' + obj_number
        new_can_obj_name = can_obj_name + '_' + obj_number
        saved_obj_fname = osp.join(unnormalized_mesh_save_dir, new_obj_name + '.obj')
        saved_can_obj_fname = osp.join(unposed_can_meshes_dir, new_can_obj_name + '.obj')

        obj_mesh.export(saved_obj_fname)
        unposed_can_mesh.export(saved_can_obj_fname)

        print(f'Saved to {saved_obj_fname}')
        
        # saved_available_poses_fname = osp.join(unnormalized_mesh_save_dir, new_obj_name + '_open_slot_poses.txt')
        # saved_available_poses_fname = osp.join(open_slot_poses_dir, new_obj_name + '_open_slot_poses.txt')
        # with open(saved_available_poses_fname, 'w') as f:
        #     np.savetxt(f, available_can_poses)

        saved_available_poses_fname = osp.join(open_slot_poses_dir, new_obj_name + '_open_slot_poses.npz')
        np.savez(saved_available_poses_fname, avail_top_poses=full_avail_can_top_poses)

        # test loading
        # loaded_poses = np.loadtxt(saved_available_poses_fname)
        loaded_poses = np.load(saved_available_poses_fname, allow_pickle=True)
        
        avail_pose_info_all = loaded_poses['avail_top_poses']
        p_idx = np.random.randint(0, avail_pose_info_all.shape[0])
        avail_pose_info = avail_pose_info_all[p_idx]

        avail_pose_top_pose = avail_pose_info['pose']
        avail_pose_dims = avail_pose_info['dims']

        cl = trimesh.creation.cylinder(radius=avail_pose_dims['r'], height=avail_pose_dims['h'])
        body_pose = avail_pose_top_pose; body_pose[2, -1] += avail_pose_dims['h']/2
        cl.apply_transform(body_pose)

        util.meshcat_trimesh_show(mc_vis, f'scene/full_can_cabinet', full_can_and_cabinet_mesh, opacity=0.8)
        util.meshcat_trimesh_show(mc_vis, f'scene/new_can', cl, color=(255, 0, 0), opacity=1.0)

        # print("here with can/cabinet")
        # from IPython import embed; embed()

        # from IPython import embed; embed()
        mc_vis['scene/make_can_cab'].delete()


if __name__ == '__main__':
    # main(args)
    main_mp(args, cfg)

