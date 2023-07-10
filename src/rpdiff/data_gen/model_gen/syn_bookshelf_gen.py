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

from ndf_robot.utils import util, path_util, trimesh_util
from ndf_robot.utils.mesh_util import inside_mesh

from syn_bookshelf_cfg import get_syn_bookshelf_default_cfg
from bookshelf_dev import make_bookshelf

VOXEL_RES = 128

parser = argparse.ArgumentParser()
parser.add_argument('--n_pts', type=int, default=100000)
parser.add_argument('--voxel_resolution', type=int, default=128)
# parser.add_argument('--voxel_pitch', type=float, default=0.005)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--occ_save_dir', type=str, required=True)
parser.add_argument('--mesh_save_dir', type=str, required=True)
parser.add_argument('--obj_name', type=str, required=True)
parser.add_argument('--book_obj_name', type=str, required=True)
parser.add_argument('--n_objs', type=int, default=200)
parser.add_argument('--difficulty', type=str, default='easy')
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--only_occ', action='store_true')
parser.add_argument('--only_center', action='store_true')

args = parser.parse_args()

cfg = get_syn_bookshelf_default_cfg()

mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
mc_vis['scene'].delete()

# path that we will save synthesized meshes in
mesh_save_dir = osp.join(path_util.get_ndf_obj_descriptions(), args.mesh_save_dir)
unnormalized_mesh_save_dir = osp.join(path_util.get_ndf_obj_descriptions(), args.mesh_save_dir + '_unnormalized')
open_slot_poses_dir = osp.join(unnormalized_mesh_save_dir, 'open_slot_poses')
unposed_book_meshes_dir = osp.join(unnormalized_mesh_save_dir, 'syn_books_obj')

# path that  we will save occupancy data in
occ_save_dir = osp.join(path_util.get_ndf_obj_descriptions(), args.occ_save_dir)

util.safe_makedirs(mesh_save_dir)
util.safe_makedirs(unnormalized_mesh_save_dir)
util.safe_makedirs(open_slot_poses_dir)
util.safe_makedirs(unposed_book_meshes_dir)
util.safe_makedirs(occ_save_dir)


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


def make_bookshelf_save_occ_f(obj_name, obj_number, cfg, normalizing_factor):
    if isinstance(obj_number, int):
        obj_number = str(obj_number)
    print('Running for object name: %s, object number: %s' % (obj_name, obj_number))
    sample_points = get_raster_points(VOXEL_RES)

    # create mesh
    obj_fname = osp.join(unnormalized_mesh_save_dir, obj_name + '_' + obj_number + '.obj')
    obj_mesh = trimesh.load(obj_fname, process=False)
    vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
    norm_factor = 1 / np.max(obj_mesh.bounding_box.extents)
    vertices *= norm_factor
    # vertices *= 1 / normalizing_factor
    obj_mesh = trimesh.Trimesh(vertices=vertices, faces=obj_mesh.faces)

    # get voxelized SDF and occupancies
    # voxels_sdf = mesh_to_voxels(obj_mesh, VOXEL_RES, pad=False)
    # occ = voxels_sdf <= 0.0
    occ = inside_mesh.check_mesh_contains(obj_mesh, sample_points)
    sdf = occ
    print('got occ...')
    
    occ_pts = sample_points[np.where(occ)[0]]
    non_occ_pts = sample_points[np.where(np.logical_not(occ))[0]]
    # util.meshcat_pcd_show(mc_vis, occ_pts, color=[255, 0, 0], name='scene/occ_pts')
    # util.meshcat_pcd_show(mc_vis, non_occ_pts, color=[255, 0, 255], name='scene/non_occ_pts')

    # save
    new_obj_name = obj_name + '_' + obj_number

    occ_save_fname = osp.join(occ_save_dir, new_obj_name + '_occupancy.npz')
    normalized_saved_obj_fname = osp.join(mesh_save_dir, new_obj_name + '.obj')

    normalized_saved_obj_fname_relative = normalized_saved_obj_fname.split(path_util.get_ndf_obj_descriptions())[1].lstrip('/')
    obj_fname_relative = obj_fname.split(path_util.get_ndf_obj_descriptions())[1].lstrip('/')

    print(f'saving to... \nnpz file: {occ_save_fname}\nmesh_file: {normalized_saved_obj_fname_relative}')

    np.savez(
        occ_save_fname,
        mesh_fname=obj_fname_relative,
        normalized_mesh_fname=normalized_saved_obj_fname_relative,
        points=sample_points,
        occupancy=occ.reshape(-1),
        sdf=occ.reshape(-1),
        norm_factor=norm_factor
    )

    obj_mesh.export(normalized_saved_obj_fname)
    print('Done with object name: %s, object number: %s' % (obj_name, obj_number))


def main_mp(args, cfg):
    obj_name = args.obj_name
    book_obj_name = args.book_obj_name
    n_objs = args.n_objs

    global_normalizing_factor = 0.0
    mp_args = [(obj_name, str(i), cfg) for i in range(n_objs)]
    for mp_arg in mp_args:
        obj_name,  obj_number, cfg = mp_arg

        # sample dimensions
        bl = rhl(cfg.BASE_LENGTH_LOW_HIGH) 
        bw = rhl(cfg.BASE_WIDTH_LOW_HIGH)
        bt = rhl(cfg.BASE_THICKNESS_LOW_HIGH)
        wt = rhl(cfg.WALL_THICKNESS_LOW_HIGH)
        wh = rhl(cfg.WALL_HEIGHT_LOW_HIGH)
        theta = rhl(cfg.WALL_THETA_LOW_HIGH)

        bkl = rhl(cfg.BOOK_LENGTH_LOW_HIGH)
        bkw = rhl(cfg.BOOK_WIDTH_LOW_HIGH)
        bkt = rhl(cfg.BOOK_THICKNESS_LOW_HIGH)

        # bkl, bkw, bkt = 0.18, 0.25, 0.04
        per_book_prob = rhl(cfg.PER_BOOK_PROB_LOW_HIGH)
        top_bool = np.random.random() > (1 - cfg.TOP_BOOL_PROB)
        eps = rhl(cfg.BOOK_SPACE_LOW_HIGH)
        n_shelves = np.random.randint(1, cfg.N_SHELVES_MAX)

        # sample bookshelf
        print(f'Making bookshelf with dimensions: {bl:.3f}, {bw:.3f}, {bt:.3f}, {wt:.3f}, {wh:.3f}, {theta:.3f}')
        full_bookshelf_mesh, unposed_book_mesh, _, available_book_poses = make_bookshelf(
            bl=bl, bw=bw, bt=bt, n_shelves=n_shelves,
            bkl=bkl, bkw=bkw, bkt=bkt,
            bkl_range=0.0, bkw_range=0.0, bkt_range=0.0,
            wt=wt, top_bool=top_bool, per_book_prob=per_book_prob, eps=eps,
            show=True, mc_vis=mc_vis)
        
        # create mesh
        obj_mesh = full_bookshelf_mesh
        # obj_mesh.vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
        new_obj_name = obj_name + '_' + obj_number
        new_book_obj_name = book_obj_name + '_' + obj_number
        saved_obj_fname = osp.join(unnormalized_mesh_save_dir, new_obj_name + '.obj')
        saved_book_obj_fname = osp.join(unposed_book_meshes_dir, new_book_obj_name + '.obj')

        obj_mesh.export(saved_obj_fname)
        unposed_book_mesh.export(saved_book_obj_fname)
        
        # saved_available_poses_fname = osp.join(unnormalized_mesh_save_dir, new_obj_name + '_open_slot_poses.txt')
        saved_available_poses_fname = osp.join(open_slot_poses_dir, new_obj_name + '_open_slot_poses.txt')
        with open(saved_available_poses_fname, 'w') as f:
            np.savetxt(f, available_book_poses)

        # test loading
        # loaded_poses = np.loadtxt(saved_available_poses_fname)

        print('Done with object name: %s, object number: %s' % (obj_name, obj_number))
        normalizing_factor = np.max(obj_mesh.bounding_box.extents)
        print('norm factor: %.3f' % normalizing_factor)
        if normalizing_factor > global_normalizing_factor:
            global_normalizing_factor = normalizing_factor
            print('new norm factor: %.3f' % global_normalizing_factor)


    # mp_args = [(obj_name, i, cfg, global_normalizing_factor) for i in range(n_objs)]
    # with Pool(args.workers) as p:
    #     p.starmap(make_bookshelf_save_occ_f, mp_args)

if __name__ == '__main__':
    # main(args)
    main_mp(args, cfg)

