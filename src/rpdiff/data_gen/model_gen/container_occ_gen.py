import os, os.path as osp
import argparse
from tqdm import tqdm 
import numpy as np
import trimesh
import time
from mesh_to_sdf import mesh_to_voxels
from multiprocessing import Pool

from ndf_robot.utils import util, path_util, trimesh_util

VOXEL_RES = 128

parser = argparse.ArgumentParser()
parser.add_argument('--n_pts', type=int, default=100000)
parser.add_argument('--datadir', type=str, default='cuboids')
parser.add_argument('--voxel_resolution', type=int, default=128)
# parser.add_argument('--voxel_pitch', type=float, default=0.005)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--occ_save_dir', type=str, required=True)
parser.add_argument('--centered_save_dir', type=str, required=True)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--only_occ', action='store_true')
parser.add_argument('--only_center', action='store_true')

args = parser.parse_args()

datadir = osp.join(path_util.get_ndf_obj_descriptions(), args.datadir)
occ_savedir = osp.join(path_util.get_ndf_obj_descriptions(), args.occ_save_dir)
centered_savedir = osp.join(path_util.get_ndf_obj_descriptions(), args.centered_save_dir)
util.safe_makedirs(occ_savedir)
util.safe_makedirs(centered_savedir)

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

sample_points = get_raster_points(VOXEL_RES)

def save_occ_f(obj_fname, obj_number):
    obj_name = obj_fname.split('/')[-1]
    new_obj_name = obj_name.split('.')[0] + '_normalized'
    normalized_saved_obj_fname = osp.join(centered_savedir, new_obj_name + '.stl')
    occ_save_fname = osp.join(occ_savedir, new_obj_name + '_occupancy.npz')
    print('Running for object name: %s, object number: %s' % (obj_name, obj_number))
    print(occ_savedir, occ_save_fname)

    # sample_points = get_raster_points(VOXEL_RES)
    obj_mesh = trimesh.load(osp.join(datadir, obj_name), process=False)
    vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
    norm_factor = 1 / np.max(obj_mesh.bounding_box.extents)
    vertices *= norm_factor
    obj_mesh = trimesh.Trimesh(vertices=vertices, faces=obj_mesh.faces)

    # get voxelized SDF and occupancies
    # occ = bb.contains(sample_points)
    # bb = obj_mesh.bounding_box_oriented
    occ = obj_mesh.contains(sample_points)
    voxels_sdf = occ

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
    print('Done with object name: %s, object number: %s to occ file: %s' % (obj_name, obj_number, occ_save_fname))

def main_mp(args):
    obj_fnames = [osp.join(datadir, fname) for fname in os.listdir(datadir)]
    # obj_fnames = os.listdir(datadir)

    # loop through to generate per-object occupancies
    mp_args = [(obj_fnames[i], str(i)) for i in range(len(obj_fnames))]
    with Pool(args.workers) as p:
        p.starmap(save_occ_f, mp_args)


if __name__ == '__main__':
    # main(args)
    main_mp(args)
