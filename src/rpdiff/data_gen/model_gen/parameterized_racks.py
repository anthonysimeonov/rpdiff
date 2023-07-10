import os, os.path as osp
import copy
import numpy as np
import trimesh
cr = trimesh.creation

from airobot.utils import common

from eof_robot.utils import util, path_util, trimesh_util

simple_rack_file = osp.join(path_util.get_eof_descriptions(), 'hanging/table/simple_rack.obj')
simple_rack_mesh = trimesh.load(simple_rack_file)

new_racks = []
for i in range(10):
    new_rack =  simple_rack_mesh.copy()
    rand_scale = np.random.random(3)
    new_rack.apply_scale(rand_scale)
    # new_rack.show()
    new_racks.append(new_rack)

# from IPython import embed; embed()

base_cyl_height, base_cyl_radius = 0.3, 0.01
peg_cyl_height, peg_cyl_radius = 0.1, 0.005

base_cyl = cr.cylinder(base_cyl_radius, base_cyl_height)
base_cyl.apply_translation([0, 0, base_cyl_height/2.0])

def rand_high_low(high, low):
    sample = np.random.random() * (high - low) + low
    return sample


def sample_peg(peg_cyl_radius, peg_cyl_height, base_cyl_radius, base_cyl_height):
    peg_cyl = cr.cylinder(peg_cyl_radius, peg_cyl_height)

    # create peg pose
    peg_base_height_frac = np.random.random() * (0.9 - 0.1) + 0.1
    peg_base_height = peg_base_height_frac * base_cyl_height
    peg_angle = rand_high_low(np.deg2rad(90), np.deg2rad(15))
    peg_yaw = rand_high_low(2*np.pi, 0)

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

# scene = trimesh.Scene()
# scene.add_geometry([base_cyl, peg_cyl])
# scene.show()
for _ in range(10):
    peg_cyl_list = []
    for _ in range(4):
        peg_cyl_radius, peg_cyl_height = rand_high_low(0.005, 0.005), rand_high_low(0.15, 0.05)
        cyl = sample_peg(peg_cyl_radius, peg_cyl_height, base_cyl_radius, base_cyl_height)
        peg_cyl_list.append(cyl)

    scene = trimesh.Scene()
    scene.add_geometry([base_cyl] + peg_cyl_list)
    scene.show()

from IPython import embed; embed()