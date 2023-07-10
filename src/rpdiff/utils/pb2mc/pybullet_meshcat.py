import os, os.path as osp
import pybullet as p
import threading
import copy
from urdfpy import URDF
from os.path import abspath, dirname, basename, splitext
import numpy as np
from scipy.spatial.transform import Rotation as R

from meshcat import geometry as mcg
from meshcat import Visualizer

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

from rpdiff.utils.pb2mc.obj2urdf import obj2urdf
from rpdiff.utils import util, path_util

from typing import List, Tuple, Union

class PyBulletMeshcat:
    class LinkTracker:
        def __init__(self,
                     name: str,
                     body_id: int,
                     link_id: int,
                     link_origin: np.ndarray,
                     mesh_path: str,
                     mesh_scale: Tuple[float],
                     client_id: int,
                     pb_client=None,
                     debug: bool=False):
            self.body_id = body_id
            self.link_id = link_id
            self.mesh_path = mesh_path
            self.mesh_scale = mesh_scale
            self.pb_client = p
            self.client_id = client_id

            origin_pos = link_origin[:-1, -1]
            origin_rot_mat = link_origin[:-1, :-1]
            origin_ori = R.from_matrix(origin_rot_mat).as_quat()
            self.link_pose = [origin_pos, origin_ori]
            self.name = name
            self.debug = debug

        def transform(self, position, orientation):
            return p.multiplyTransforms(
                position, orientation,
                self.link_pose[0], self.link_pose[1],
            )

        def get_keyframe(self):
            if self.link_id == -1:
                position, orientation = self.pb_client.getBasePositionAndOrientation(
                    self.body_id, physicsClientId=self.client_id)
                position, orientation = self.transform(position=position, orientation=orientation)
            else:
                link_state = self.pb_client.getLinkState(
                    self.body_id, self.link_id,
                    computeForwardKinematics=True, physicsClientId=self.client_id)
                position, orientation = self.transform(
                    position=link_state[4], orientation=link_state[5])
            return {'position': list(position), 'orientation': list(orientation)}

    def __init__(self, pb_client=None, tmp_urdf_dir: str=None, debug: bool=False):
        self.client_id = p.connect(p.SHARED_MEMORY, key=p.SHARED_MEMORY_KEY2)
        self.pb_client = p
        self.states = []
        self.links = []
        self.links_dict = {}
        self.current_state = None
        self.known_meshcat_objs = []
        
        self.urdf_lock = threading.RLock()
        if tmp_urdf_dir is None:
            self.tmp_urdf_dir = osp.join(path_util.get_rpdiff_obj_descriptions(), 'tmp_urdf')
        else:
            self.tmp_urdf_dir = tmp_urdf_dir
        util.safe_makedirs(self.tmp_urdf_dir)
        self.current_state_lock = threading.RLock()
        self.links_lock = threading.RLock()
        self.keyframe_lock = threading.RLock()
        self.meshcat_scene_lock = threading.RLock()

        self.debug = debug

    def acq(self):
        self.current_state_lock.acquire()
        self.keyframe_lock.acquire()
        self.links_lock.acquire()

    def rel(self):
        self.links_lock.release()
        self.current_state_lock.release()
        self.keyframe_lock.release()

    def clear(self):
        # self.current_state_lock.acquire()
        # self.keyframe_lock.acquire()
        # self.links_lock.acquire()
        self.acq()
        self.states = []
        self.links = []
        self.links_dict = {}
        self.current_state = None
        self.known_meshcat_objs = []
        # self.links_lock.release()
        # self.current_state_lock.release()
        # self.keyframe_lock.release()
        self.rel()

    def remove_object(self, body_id: int, mc_handler: Visualizer):
        # self.current_state_lock.acquire()
        # self.keyframe_lock.acquire()
        # self.links_lock.acquire()
        self.acq()
        self.meshcat_scene_lock.acquire()
        link_keys = copy.deepcopy(list(self.links_dict.keys()))
        for k in link_keys:
            if f'{body_id}::' in k:
                link_name = self.links_dict[k].name
                mc_name = f'scene/{link_name}'
                if mc_name in self.known_meshcat_objs:
                    self.known_meshcat_objs.remove(mc_name)
                    mc_handler[mc_name].delete()
                del self.links_dict[k]
        # self.links_lock.release()
        # self.keyframe_lock.release()
        # self.current_state_lock.release()
        self.meshcat_scene_lock.release()
        self.rel()

    def register_object(self, body_id: int, 
                        model_path: str, scaling: Union[List[float], float]=None, 
                        global_scaling: float=1, link_geom_type: str='collision'):
        """
        Register the pybullet body with the tracker so we can obtain updated simulator states,
        map them to the correct mesh asset, and send them to meshcat

        Args:
            obj_id (int): pybullet body id
            model_path (str): path to either URDF or .obj file containing the object geometry. 
                Warning -- this does not accept .stl files. 
            scaling (int or list): todo  
            global_scaling (int): todo
            link_geom_type (str): either 'visual' or 'collision'. sometimes the .obj files in
                a 'visual' mesh asset directory will get loaded as a type we cannot easily load
                into meshcat, whereas the 'collision' version will almost always be compatible,
                at the loss of some geometric fidelity
        """
        link_id_map = dict()
        n = self.pb_client.getNumJoints(body_id)
        if self.debug:
            log_debug(f'[Register Object] Body_id: {body_id}, n: {n}, client_id: {self.client_id}')
        if n == 0:
            link_id_map['base_link'] = -1
        else:
            link_id_map[self.pb_client.getBodyInfo(body_id)[0].decode('gb2312')] = -1
            for link_id in range(0, n):
                link_id_map[self.pb_client.getJointInfo(body_id, link_id)[12].decode('gb2312')] = link_id

        if not model_path.endswith('.urdf'):
            # build a temp URDf if we are passed just a single .obj or .stl file
            obj_name = model_path.split('/')[-1].split('.')[0]
            self.urdf_lock.acquire()
            urdf_str, urdf_path = obj2urdf(model_path, obj_name, save_dir=self.tmp_urdf_dir, scaling=scaling)
            self.urdf_lock.release()
        else:
            urdf_path = model_path

        dir_path = dirname(abspath(urdf_path))
        file_name = splitext(basename(urdf_path))[0]
        if self.debug:
            log_debug(f'[Register Object] dir_path: {dir_path}, file_name: {file_name}')
        robot = URDF.load(urdf_path)
        for link in robot.links:
            link_id = link_id_map[link.name]
            links_to_use = link.visuals if link_geom_type == 'visual' else link.collisions
            if len(links_to_use) > 0:
                for i, link_visual in enumerate(links_to_use):
                    if link_visual.geometry.mesh is None:
                        continue
                    mesh_scale = [global_scaling, global_scaling, global_scaling] if link_visual.geometry.mesh.scale is None \
                        else link_visual.geometry.mesh.scale * global_scaling

                    # process the filename of the meshes, in case they contain relative paths
                    mesh_file_path = link_visual.geometry.mesh.filename 
                    # if the file doesn't exist, let's guess that it's in the same directory as the URDF
                    if self.debug:
                        log_debug(f'[Register Object] mesh_file_path: {mesh_file_path}')
                    if not osp.exists(mesh_file_path):
                        mesh_file_path = osp.join(dir_path, mesh_file_path)
                        if self.debug:
                            log_debug(f'[Register Object] mesh_file_path: {mesh_file_path}')
                    
                    # let's remove any package:// prefixes
                    if 'package://' in mesh_file_path:
                        mesh_file_path.replace('package://', '')
                        if self.debug:
                            log_debug(f'[Register Object] mesh_file_path: {mesh_file_path}')

                    self.links_lock.acquire()
                    # If link_id == -1 then is base link,
                    # PyBullet will return
                    # inertial_origin @ visual_origin,
                    # so need to undo that transform
                    if link_id == -1:
                        link_origin = np.linalg.inv(link.inertial.origin) @ link_visual.origin * global_scaling
                    else:
                        link_origin = link_visual.origin * global_scaling

                    link_element = PyBulletMeshcat.LinkTracker(
                            name=f'{file_name}_{body_id}_{link.name}_{i}',
                            body_id=body_id,
                            link_id=link_id,
                            link_origin=link_origin,
                            mesh_path=mesh_file_path,
                            mesh_scale=mesh_scale,
                            client_id=self.client_id,
                            pb_client=self.pb_client)
                    if False:
                        link_element = PyBulletRecorder.LinkTracker(
                                name=f'{file_name}_{body_id}_{link.name}_{i}',
                                body_id=body_id,
                                link_id=link_id,
                                link_origin=  # If link_id == -1 then is base link,
                                # PyBullet will return
                                # inertial_origin @ visual_origin,
                                # so need to undo that transform
                                (np.linalg.inv(link.inertial.origin)
                                if link_id == -1
                                else np.identity(4)) @
                                link_visual.origin * global_scaling,
                                mesh_path=mesh_file_path,
                                mesh_scale=mesh_scale,
                                client_id=self.client_id,
                                pb_client=self.pb_client)
                    link_dict_key = f'{body_id}::{link_id}'
                    self.links_dict[link_dict_key] = link_element
                    self.links_lock.release()

    def add_keyframe(self):
        # Ideally, call every p.stepSimulation()
        self.current_state_lock.acquire()

        current_state = {}

        self.links_lock.acquire()
        for link in self.links_dict.values():
            self.keyframe_lock.acquire()
            current_state[link.name] = link.get_keyframe()
            self.keyframe_lock.release()
        self.links_lock.release()

        self.states.append(current_state)
        self.current_state = copy.deepcopy(current_state)

        self.current_state_lock.release()

    def update_meshcat_current_state(self, mc_handler: Visualizer):
        self.current_state_lock.acquire()
        if self.current_state is None:
            log_info('Don"t yet have a current state, please call "add_keyframe()" to get one')
            return
        self.links_lock.acquire()
        for link in self.links_dict.values():
            mc_name = f'scene/{link.name}'
            obj_file_to_load = link.mesh_path
            mesh_scale = link.mesh_scale
            obj_pos = self.current_state[link.name]['position']
            obj_ori = self.current_state[link.name]['orientation']
            obj_pose_world_np = np.asarray(obj_pos + obj_ori)
            if self.debug:
                log_debug(f'Object name: {link.name}, Object mesh file: {obj_file_to_load}, Object pose world (meshcat): {",".join(list(map(str, obj_pose_world_np.tolist())))}')
            
            self.meshcat_scene_lock.acquire()
            if mc_name not in self.known_meshcat_objs:
                mc_handler[mc_name].delete()
                self.known_meshcat_objs.append(mc_name)
                mc_handler[mc_name].set_object(mcg.ObjMeshGeometry.from_file(obj_file_to_load))
            mc_mat = np.matmul(util.matrix_from_pose(util.list2pose_stamped(obj_pose_world_np)), util.scale_matrix(mesh_scale))
            mc_handler[mc_name].set_transform(mc_mat)
            self.meshcat_scene_lock.release()
        self.links_lock.release()
        self.current_state_lock.release()

    def reset(self):
        self.states = []

    def get_formatted_output(self):
        retval = {}
        for link in self.links_dict.values():
            retval[link.name] = {
                'type': 'mesh',
                'mesh_path': link.mesh_path,
                'mesh_scale': link.mesh_scale,
                'frames': [state[link.name] for state in self.states]
            }
        return retval

    def get_formatted_current_state(self):
        retval = {}
        for link in self.links_dict.values():
            retval[link.name] = {
                'type': 'mesh',
                'mesh_path': link.mesh_path,
                'mesh_scale': link.mesh_scale,
                'current_state': self.current_state[link.name]
            }
        return retval
