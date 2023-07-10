import os, os.path as osp
import numpy as np
import random
import trimesh
import time
import pybullet as p

from pybullet_recorder import PyBulletRecorder
from airobot import Robot
from airobot.utils import common
from airobot import log_info
from airobot.utils.common import euler2quat
from airobot.utils.pb_util import create_pybullet_client

import meshcat
from meshcat import geometry as mcg
from meshcat import transformations as mctf

from eof_robot.utils import util, path_util

robot = Robot('franka', pb_cfg={'gui': False})
pb_client = robot.pb_client

TABLE_POS = [0.5, 0.0, 0.4]
TABLE_SCALING = 0.9

table_ori = euler2quat([0, 0, np.pi / 2])
table_id = pb_client.load_urdf('table/table.urdf',
    TABLE_POS,
    table_ori,
    scaling=TABLE_SCALING)

mesh_dir = osp.join(path_util.get_eof_obj_descriptions(), 'mug_centered_obj_normalized')
mesh_sid = random.sample(os.listdir(mesh_dir), 1)[0]
mesh_fname = osp.join(mesh_dir, mesh_sid, 'models/model_normalized.obj')

mesh = trimesh.load(mesh_fname)
#obj_scaling = [1.0, 0.5, 2.0]
obj_scaling = [1.0]*3
ori1 = [0, 0, 0, 1]
ori2 = np.random.rand(4)
ori2 = ori2 / np.linalg.norm(ori2)
ori3 = common.euler2quat([np.pi/2, 0, 0, ])
obj_id = pb_client.load_geom(
        'mesh',
        mass=0.01,
        mesh_scale=obj_scaling,
        visualfile=mesh_fname,
        collifile=mesh_fname,
        base_pos=[0.3, 0, 1.5],
        base_ori=ori2)
        

recorder = PyBulletRecorder()
recorder.register_object(obj_id, mesh_fname, scaling=obj_scaling)

vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')

pose = p.getBasePositionAndOrientation(obj_id)[:2]
pos, ori = list(pose[0]), list(pose[1])
pb_client.set_step_sim(True)

while True:
    pos[1] += 0.001
    if pos[1] > 1.5:
        pos[1] = 0.0
    pb_client.reset_body(obj_id, pos, ori)
    recorder.add_keyframe()
    recorder.update_meshcat_current_state(vis)
    time.sleep(0.01)

from IPython import embed; embed()

