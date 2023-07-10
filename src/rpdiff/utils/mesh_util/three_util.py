from typing import List
import numpy as np
import trimesh
from mesh_to_sdf import mesh_to_voxels
from torch import Tensor, FloatTensor, LongTensor

from rpdiff.utils import fork_pdb

from typing import List, Tuple, Union

def trimesh_show(
        np_pcd_list: List[np.ndarray], 
        color_list: List[np.ndarray]=None, 
        show: bool=True) -> trimesh.Scene:
    if color_list is None:
        color_list = []
        for i in range(len(np_pcd_list)):
            color_list.append((np.random.rand(3) * 255).astype(np.int32).tolist() + [255])
    
    tpcd_list = []
    for i, pcd in enumerate(np_pcd_list):
        tpcd = trimesh.PointCloud(pcd)
        tpcd.colors = np.tile(color_list[i], (tpcd.vertices.shape[0], 1))

        tpcd_list.append(tpcd)
    
    scene = trimesh.Scene()
    scene.add_geometry(tpcd_list)
    if show:
        scene.show() 

    return scene


def trimesh_combine(
        mesh_files: List[str], 
        mesh_poses: List[np.ndarray],
        mesh_scales: List[np.ndarray]=None) -> trimesh.Trimesh:
    meshes = []
    if mesh_scales is None:
        default = [1.0, 1.0, 1.0]
        mesh_scales = [default] * len(mesh_files)
    for i, mesh in enumerate(mesh_files):
        tmesh = trimesh.load(mesh, process=False)
        if isinstance(tmesh, trimesh.Scene):
            # converts Scene into individual meshes
            sep_meshes = tmesh.dump()
            for sep_mesh in sep_meshes:
                sep_mesh.apply_scale(mesh_scales[i])
                sep_mesh.apply_transform(mesh_poses[i])
                meshes.append(sep_mesh)
        else:
            tmesh.apply_scale(mesh_scales[i])
            tmesh.apply_transform(mesh_poses[i])
            meshes.append(tmesh) 
    
    concat_mesh = trimesh.util.concatenate(meshes)
    return concat_mesh


def get_raster_points(voxel_resolution: int, padding: float=0.0) -> np.ndarray:
    points = np.meshgrid(
        np.linspace(-0.5-padding/2, 0.5+padding/2, voxel_resolution),
        np.linspace(-0.5-padding/2, 0.5+padding/2, voxel_resolution),
        np.linspace(-0.5-padding/2, 0.5+padding/2, voxel_resolution)
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(np.float32)
    return points


def coordinate2index(x: FloatTensor, reso: int, coord_type: str='2d') -> LongTensor:
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


def normalize_3d_coordinate(p: FloatTensor, padding: float=0.1) -> FloatTensor:
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def coordinate2index_np(x: np.ndarray, reso: int, coord_type: str='2d') -> np.ndarray:
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).astype(np.int64)
    if x.ndim < 3:
        x = x.reshape(1, -1, 3)
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        # index = x[:, :, 2] + reso * (x[:, :, 1] + reso * x[:, :, 0])
        index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
    index = index[:, None, :]
    return index


def normalize_3d_coordinate_np(p: np.ndarray, padding: float=0.1) -> np.ndarray:
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''
    
    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def scale_mesh(mesh: trimesh.Trimesh, offset: np.ndarray=None, 
               scaling: float=None) -> trimesh.Trimesh:
    if offset is None:
        vertices = mesh.vertices - mesh.bounding_box.centroid
    else:
        vertices = mesh.vertices - offset
    
    if scaling is None:
        vertices *= 1 / np.max(mesh.bounding_box.extents)
    else:
        vertices *= scaling

    scaled_mesh = trimesh.Trimesh(vertices=vertices, faces=mesh.faces)
    return scaled_mesh


def get_occ(obj_mesh: trimesh.Trimesh, VOXEL_RES: int, 
            offset: np.ndarray=None, scaling: float=None, 
            sample_points: np.ndarray=None) -> Tuple[np.ndarray]:
    if sample_points is None:
        sample_points = get_raster_points(VOXEL_RES)

    if offset is None:
        vertices = obj_mesh.vertices - obj_mesh.bounding_box.centroid
    else:
        vertices = obj_mesh.vertices - offset
    
    if scaling is None:
        vertices *= 1 / np.max(obj_mesh.bounding_box.extents)
    else:
        vertices *= scaling

    obj_mesh = trimesh.Trimesh(vertices=vertices, faces=obj_mesh.faces)

    # get voxelized SDF and occupancies
    voxels_sdf = mesh_to_voxels(obj_mesh, VOXEL_RES, pad=False)
    occ = voxels_sdf <= 0.0

    return sample_points, occ.reshape(-1), voxels_sdf.reshape(-1)
