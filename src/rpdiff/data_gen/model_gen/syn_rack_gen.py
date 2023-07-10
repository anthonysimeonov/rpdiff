import os, os.path as osp
import argparse
from tqdm import tqdm 
import numpy as np
import trimesh
import trimesh.creation as cr
import time
from mesh_to_sdf import mesh_to_voxels
from multiprocessing import Pool
import meshcat

from airobot.utils import common

from ndf_robot.utils import util, path_util, trimesh_util

from syn_rack_cfg import get_syn_rack_default_cfg

VOXEL_RES = 128

parser = argparse.ArgumentParser()
parser.add_argument('--n_pts', type=int, default=100000)
parser.add_argument('--voxel_resolution', type=int, default=128)
# parser.add_argument('--voxel_pitch', type=float, default=0.005)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--occ_save_dir', type=str, required=True)
parser.add_argument('--mesh_save_dir', type=str, required=True)
parser.add_argument('--obj_name', type=str, default='syn_rack')
parser.add_argument('--n_objs', type=int, default=200)
parser.add_argument('--difficulty', type=str, default='easy')
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--only_occ', action='store_true')
parser.add_argument('--only_center', action='store_true')

args = parser.parse_args()

cfg = get_syn_rack_default_cfg()
if args.difficulty == 'easy':
    rack_cfg = cfg.EASY
elif args.difficulty == 'med':
    rack_cfg = cfg.MED
else:
    rack_cfg = cfg.HARD

mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')

# path that we will save synthesized rack meshes in
mesh_save_dir = osp.join(path_util.get_ndf_obj_descriptions(), args.mesh_save_dir)
unnormalized_mesh_save_dir = osp.join(path_util.get_ndf_obj_descriptions(), args.mesh_save_dir + '_unnormalized')

# path that  we will save occupancy data in
occ_save_dir = osp.join(path_util.get_ndf_obj_descriptions(), args.occ_save_dir)

util.safe_makedirs(mesh_save_dir)
util.safe_makedirs(unnormalized_mesh_save_dir)
util.safe_makedirs(occ_save_dir)


def rhl(high, low):
    sample = np.random.random() * (high - low) + low
    return sample


def rhlb(bounds):
    sample = np.random.random() * (max(bounds) - min(bounds)) + low
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


def sample_peg(peg_cyl_radius, peg_cyl_height, base_cyl_radius, base_cyl_height, rack_cfg, top_bottom_half=None):
    peg_cyl = cr.cylinder(peg_cyl_radius, peg_cyl_height)

    # create peg pose
    # half_min = (max(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH) + min(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH)) / 2.0
    half_min = 0.5
    if top_bottom_half is None:
        peg_base_height_frac = rhl(max(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH), min(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH))
    elif top_bottom_half == 'top':
        peg_base_height_frac = rhl(max(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH), half_min)
    elif top_bottom_half == 'bottom':
        peg_base_height_frac = rhl(half_min, min(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH))
    else:
        peg_base_height_frac = rhl(max(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH), min(rack_cfg.PEG_BASE_HEIGHT_FRAC_LOW_HIGH))
    peg_base_height = peg_base_height_frac * base_cyl_height
    peg_angle = rhl(np.deg2rad(max(rack_cfg.PEG_ANGLE_LOW_HIGH)), np.deg2rad(min(rack_cfg.PEG_ANGLE_LOW_HIGH)))
    peg_yaw = rhl(np.deg2rad(max(rack_cfg.PEG_YAW_LOW_HIGH)), np.deg2rad(min(rack_cfg.PEG_YAW_LOW_HIGH)))

    peg_rot1 = common.euler2rot([peg_angle, 0, 0])
    peg_rot2 = common.euler2rot([0, 0, peg_yaw])
    peg_tf1 = np.eye(4)
    peg_tf1[:-1, :-1] = peg_rot1
    peg_tf1[2, -1] = peg_base_height
    peg_tf1[1, -1] = -(peg_cyl_height / 2.0) * np.sin(peg_angle)
    peg_tf2 = np.eye(4)
    peg_tf2[:-1, :-1] = peg_rot2
    peg_tf = np.matmul(peg_tf2, peg_tf1)

    peg_cyl.apply_transform(peg_tf)
    return peg_cyl


def sample_syn_rack(rack_cfg):
    base_cyl_radius = rhl(max(rack_cfg.BASE_RADIUS_LOW_HIGH), min(rack_cfg.BASE_RADIUS_LOW_HIGH))
    base_cyl_height = rhl(max(rack_cfg.BASE_LENGTH_LOW_HIGH), min(rack_cfg.BASE_LENGTH_LOW_HIGH))
    base_cyl = cr.cylinder(base_cyl_radius, base_cyl_height)
    base_cyl.apply_translation([0, 0, base_cyl_height/2.0])

    peg_cyl_list = []
    if rack_cfg.N_PEGS < 0:
        n_pegs = np.random.randint(1, rack_cfg.MAX_PEGS+1) 
    else:
        n_pegs = rack_cfg.N_PEGS
    for i in range(n_pegs):
        top_bottom = 'top' if i == 0 else 'bottom'
        peg_cyl_radius = rhl(max(rack_cfg.PEG_RADIUS_LOW_HIGH), min(rack_cfg.PEG_RADIUS_LOW_HIGH))
        peg_cyl_height = rhl(max(rack_cfg.PEG_LENGTH_LOW_HIGH), min(rack_cfg.PEG_LENGTH_LOW_HIGH))
        peg_cyl = sample_peg(
            peg_cyl_radius, 
            peg_cyl_height, 
            base_cyl_radius, 
            base_cyl_height, 
            rack_cfg,
            top_bottom_half=top_bottom)
        peg_cyl_list.append(peg_cyl)

    rack_mesh_list = [base_cyl] + peg_cyl_list

    if rack_cfg.WITH_BOTTOM and (np.random.random() > 0.5):
        if np.random.random() > 0.5:
            bottom_cyl_radius = rhl(max(rack_cfg.BOTTOM_CYLINDER_RADIUS_LOW_HIGH), min(rack_cfg.BOTTOM_CYLINDER_RADIUS_LOW_HIGH))
            bottom_cyl_height = rhl(max(rack_cfg.BOTTOM_CYLINDER_LENGTH_LOW_HIGH), min(rack_cfg.BOTTOM_CYLINDER_LENGTH_LOW_HIGH))
            bottom_cyl = cr.cylinder(bottom_cyl_radius, bottom_cyl_height)
            bottom_cyl.apply_translation([0, 0, bottom_cyl_height/2.0])
            bottom_mesh = bottom_cyl
        else:
            bottom_box_side = rhl(max(rack_cfg.BOTTOM_BOX_SIDE_LOW_HIGH), min(rack_cfg.BOTTOM_BOX_SIDE_LOW_HIGH))
            bottom_box_height = rhl(max(rack_cfg.BOTTOM_BOX_HEIGHT_LOW_HIGH), min(rack_cfg.BOTTOM_BOX_HEIGHT_LOW_HIGH))
            bottom_box = cr.box([bottom_box_side, bottom_box_side, bottom_box_height])
            bottom_box.apply_translation([0, 0, bottom_box_height/2.0])
            bottom_mesh = bottom_box
        rack_mesh_list.append(bottom_mesh)

    return rack_mesh_list


def make_rack_save_occ_f(obj_name, obj_number, rack_cfg, normalizing_factor):
    if isinstance(obj_number, int):
        obj_number = str(obj_number)
    print('Running for object name: %s, object number: %s' % (obj_name, obj_number))
    sample_points = get_raster_points(VOXEL_RES)

    # # sample rack mesh
    # rack_mesh_list = sample_syn_rack(rack_cfg)
    # merged_rack = trimesh.util.concatenate(rack_mesh_list)
    # print('made mesh...')
    
    
    # create mesh
    # obj_mesh = merged_rack.copy()
    obj_fname = osp.join(unnormalized_mesh_save_dir, obj_name + '_' + obj_number + '.obj')
    obj_mesh = trimesh.load(obj_fname, process=False)
    vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
    norm_factor = 1 / np.max(obj_mesh.bounding_box.extents)
    vertices *= norm_factor
    # vertices *= 1 / normalizing_factor
    obj_mesh = trimesh.Trimesh(vertices=vertices, faces=obj_mesh.faces)

    # get voxelized SDF and occupancies
    voxels_sdf = mesh_to_voxels(obj_mesh, VOXEL_RES, pad=False)
    occ = voxels_sdf <= 0.0
    print('got occ...')

    # save
    new_obj_name = obj_name + '_' + obj_number

    occ_save_fname = osp.join(occ_save_dir, new_obj_name + '_occupancy.npz')
    normalized_saved_obj_fname = osp.join(mesh_save_dir, new_obj_name + '.obj')
    print('saving to... \n%s\n%s' % (occ_save_fname, normalized_saved_obj_fname))

    np.savez(
        occ_save_fname,
        mesh_fname=obj_fname,
        normalized_mesh_fname=normalized_saved_obj_fname,
        points=sample_points,
        occupancy=occ.reshape(-1),
        sdf=voxels_sdf.reshape(-1),
        norm_factor=norm_factor
    )

    obj_mesh.export(normalized_saved_obj_fname)
    print('Done with object name: %s, object number: %s' % (obj_name, obj_number))


def main_mp(args, rack_cfg):
    obj_name = args.obj_name
    n_objs = args.n_objs

    global_normalizing_factor = 0.0
    mp_args = [(obj_name, str(i), rack_cfg) for i in range(n_objs)]
    for mp_arg in mp_args:
        obj_name,  obj_number, rack_cfg = mp_arg
        # sample rack mesh
        rack_mesh_list = sample_syn_rack(rack_cfg)
        merged_rack = trimesh.util.concatenate(rack_mesh_list)
        print('made mesh...')
        util.meshcat_trimesh_show(mc_vis, 'scene/rack', merged_rack)
        # input('press enter to continue')
        
        # create mesh
        obj_mesh = merged_rack
        obj_mesh.vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
        new_obj_name = obj_name + '_' + obj_number
        saved_obj_fname = osp.join(unnormalized_mesh_save_dir, new_obj_name + '.obj')
        obj_mesh.export(saved_obj_fname)
        print('Done with object name: %s, object number: %s' % (obj_name, obj_number))
        normalizing_factor = np.max(obj_mesh.bounding_box.extents)
        print('norm factor: %.3f' % normalizing_factor)
        if normalizing_factor > global_normalizing_factor:
            global_normalizing_factor = normalizing_factor
            print('new norm factor: %.3f' % global_normalizing_factor)


    mp_args = [(obj_name, i, rack_cfg, global_normalizing_factor) for i in range(n_objs)]
    with Pool(args.workers) as p:
        p.starmap(make_rack_save_occ_f, mp_args)

if __name__ == '__main__':
    # main(args)
    main_mp(args, rack_cfg)

