import os, os.path as osp
import argparse
import numpy as np
import trimesh
import meshcat

from airobot.utils import common

from ndf_robot.utils import util, path_util, trimesh_util

mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')

def make_container(bl, bw, bt, wt, wh, th, show=False):
    """
    Args:
        bl (float): Base length
        bw (float): Base width
        bt (float): Base thickness
        wt (float): Wall thickness
        wh (float): Wall height 
        th (float): Wall angle (degrees)

    Returns:
        trimesh.Trimesh: concatenated mesh of the whole container
        dict: Dictionary with all the part meshes
    """
    scene = trimesh.Scene()
    # l, h, t corresponds to [x, y, z]

    # base 
    # bl, bw, bt = 0.06, 0.12, 0.005
    base_dims = [bl, bw, bt]
    base = trimesh.creation.box(base_dims)

    # global wall dims
    # wt = 0.005  # wall thickness
    # wh = 0.04  # wall height

    # th = 10  # let's assume the angles are always the same for now
    fb_theta = np.deg2rad(th)
    side_theta = np.deg2rad(th)

    # front/back walls
    fbl, fbw, fbt = wh, bw, wt
    fb_dims = [fbl, fbw, fbt]
    fb_d1 = (wh/2)*np.sin(fb_theta)
    fb_d2 = (wh/2)*(1 - np.cos(fb_theta))
    front_wall = trimesh.creation.box(fb_dims)
    back_wall = trimesh.creation.box(fb_dims)
    front_back_walls = dict(front=front_wall, back=back_wall)
    # scene.add_geometry([base, front_wall])

    # side walls
    sl, sw, st = bl, wh, wt
    side_dims = [sl, sw, st]
    s_d1 = (wh/2)*np.sin(side_theta)
    s_d2 = (wh/2)*(1 - np.cos(side_theta))
    left_wall = trimesh.creation.box(side_dims)
    right_wall = trimesh.creation.box(side_dims)
    side_walls = dict(left=left_wall, right=right_wall)
    # scene.add_geometry([base, right_wall])

    # transformations for all the walls
    # front/back need to translate in x and rotate about y (pitch)
    front_trans = np.array([bl/2 + fb_d1, 0, wh/2 + fb_d2])
    front_rot = common.euler2rot([0, (np.pi/2 + fb_theta), 0])
    front_trans_mat = np.eye(4); front_trans_mat[:-1, :-1] = front_rot; front_trans_mat[:-1, -1] = front_trans

    back_trans = np.array([-bl/2 - fb_d1, 0, wh/2 + fb_d2])
    back_rot = common.euler2rot([0, -(np.pi/2 + fb_theta), 0])
    back_trans_mat = np.eye(4); back_trans_mat[:-1, :-1] = back_rot; back_trans_mat[:-1, -1] = back_trans

    fb_trans = dict(front=front_trans_mat, back=back_trans_mat)
    for k, v in front_back_walls.items():
        v.apply_transform(fb_trans[k])
        
    # left/right need to translate in y and rotate about x (roll)
    left_trans = np.array([0, bw/2 + s_d1, wh/2 + s_d2])
    left_rot = common.euler2rot([-(np.pi/2 + side_theta), 0, 0])
    left_trans_mat = np.eye(4); left_trans_mat[:-1, :-1] = left_rot; left_trans_mat[:-1, -1] = left_trans

    right_trans = np.array([0, -bw/2 - s_d1, wh/2 + s_d2])
    right_rot = common.euler2rot([(np.pi/2 + side_theta), 0, 0])
    right_trans_mat = np.eye(4); right_trans_mat[:-1, :-1] = right_rot; right_trans_mat[:-1, -1] = right_trans

    side_trans = dict(left=left_trans_mat, right=right_trans_mat)
    for k, v in side_walls.items():
        v.apply_transform(side_trans[k])

    # cones to fill in the corners
    cone_names = ['fl', 'fr', 'bl', 'br']
    cr, ch = 1.0*(wh/2)*np.sin(fb_theta), wh
    # pre_rot = common.euler2rot([np.pi/2, 0, 0])
    cones = {k: trimesh.creation.cone(cr, ch) for k in cone_names}
    for k, v in cones.items():
        if k == 'fl':
            rot = common.euler2rot([(np.pi - fb_theta), (side_theta), 0])
            trans = np.array([bl/2 + fb_d1, bw/2 + s_d1, wh + fb_d2])
        elif k == 'fr':
            rot = common.euler2rot([-(np.pi - fb_theta), (side_theta), 0])
            trans = np.array([bl/2 + fb_d1, -(bw/2 + s_d1), wh + fb_d2])
        elif k == 'bl':
            rot = common.euler2rot([(np.pi - fb_theta), -(side_theta), 0])
            trans = np.array([-(bl/2 + fb_d1), bw/2 + s_d1, wh + fb_d2])
        else: # br
            rot = common.euler2rot([-(np.pi - fb_theta), -(side_theta), 0])
            trans = np.array([-(bl/2 + fb_d1), -(bw/2 + s_d1), wh + fb_d2])
        
        trans_mat = np.eye(4); trans_mat[:-1, :-1] = rot; trans_mat[:-1, -1] = trans
        # trans_mat = np.eye(4); trans_mat[:-1, -1] = trans
        v.apply_transform(trans_mat)

    # scene.add_geometry([base] + list(front_back_walls.values()))
    # scene.add_geometry([base] + list(front_back_walls.values()) + list(side_walls.values()))

    # merge
    mesh_dict = {'base': base}
    for k, v in front_back_walls.items():
        mesh_dict[k] = v
    for k, v in side_walls.items():
        mesh_dict[k] = v
    for k, v in cones.items():
        mesh_dict[k] = v
    full_container_mesh = trimesh.util.concatenate(list(mesh_dict.values()))

    if show:
        scene.add_geometry([base] + list(front_back_walls.values()) + list(side_walls.values()) + list(cones.values()))
        util.meshcat_trimesh_show(mc_vis, 'scene/full_container', full_container_mesh)
        # input('waiting for user to view in meshcat')
        # scene.show()

    return full_container_mesh, mesh_dict


if __name__ == "__main__":
    bl, bw, bt = 0.06, 0.12, 0.005
    wt = 0.005  # wall thickness
    wh = 0.04  # wall height

    th = 10  # let's assume the angles are always the same for now

    full_container_mesh, container_part_mesh_dict = make_container(bl=bl, bw=bw, bt=bt, wt=wt, wh=wh, th=th, show=False)
    full_container_mesh.show()
    from IPython import embed; embed()
