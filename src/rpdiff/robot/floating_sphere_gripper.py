import os, os.path as osp
from typing import List, Union, Tuple
import numpy as np
import copy
import time

import pybullet as p

from airobot.utils.pb_util import BulletClient
import airobot.utils.common as arutil
from airobot.utils.arm_util import wait_to_reach_jnt_goal
from airobot import set_log_level, log_debug, log_info, log_warn, log_critical

from rpdiff.utils import util, path_util

class InverseKinematicsError(Exception):
    pass

class FloatingSphereGripper:
    def __init__(self, 
                 pb_client: BulletClient, 
                 urdf_file: str=None, 
                 init_pos: List[float]=[0.0, 0.0, 0.0], 
                 home_pos: List[float]=[0.0, 0.0, 0.0],
                 base_body_link_id=(0, -1)): 

        if urdf_file is None:
            self.urdf_file = osp.join(path_util.get_rpdiff_descriptions(), 'sphere_gripper_floating.urdf')
        else:
            self.urdf_file = urdf_file

        self.init_pos = init_pos
        self.base_body_id, self.base_link_id = base_body_link_id
        self.pb_client = pb_client
        self.home_position = home_pos

        self.hand = FloatingHandDriven(self.urdf_file, pb_client=self.pb_client, init_pos=self.init_pos)

        self.hand.set_home_position(self.home_position)
        self.hand.go_home(ignore_physics=True)
        self.hand_id = self.hand.hand_id

        self._setup()
        self.reset_hand = self.reset_hand_position
        self.drive_hand = self.drive_hand_joints
        self.get_ee_pose = self._get_ee_pose_joints

    def _setup(self):
        self.color_info = p.getVisualShapeData(self.hand_id)

    def hide_hand(self):
        for j in range(len(self.color_info)):
            link_id = self.color_info[j][1]
            color_data = self.color_info[j][7]
            p.changeVisualShape(
                self.hand_id,
                link_id,
                rgbaColor=[color_data[0], color_data[1], color_data[2], 0.0])

    def show_hand(self):
        for j in range(len(self.color_info)):
            link_id = self.color_info[j][1]
            color_data = self.color_info[j][7]
            p.changeVisualShape(
                self.hand_id,
                link_id,
                rgbaColor=color_data)

    def to_safe_pos(self):
        ik_trials = 0
        while True:
            ik_trials += 1
            if ik_trials > 10:
                raise InverseKinematicsError('Floating Plams to_safe_pos: Tried to solve IK 10 times, and failed')
            try:
                rand_ori = np.random.rand(4); ori = rand_ori / np.linalg.norm(rand_ori)
                self.reset_hand(pos=self.safe_position_r, ori=ori.tolist())
                break
            except InverseKinematicsError as e:
                log_warn(e)

    def reset_hand_position(self, pos, ori):
        """Function to reset the pose of the palms by reseting the joint positions with 
        ignore_physics=True (reseting the state by overriding the physics simulation)
        and then sending a position command that matches the state we have reset to. 
        This function assumes that input position and orientation are positions/orientations
        of the WRIST of the palms. Depending on whether we are controlling in wrist mode or tip
        mode, we may manually perform some transformations of this input within this function

        Args:
            pos (list or np.ndarray): [x, y, z] position of the wrist to reset to 
            ori (list or np.ndarray): [x, y, z, w] orientation of the wrist to reset to
        """
        hand = self.hand

        self.pb_client.set_step_sim(True)
        hand.set_ee_pose(pos, ori, ignore_physics=True)
        self.pb_client.set_step_sim(False)
        hand.set_ee_pose(pos, ori, wait=True, ignore_physics=False)

        self.set_compliant_jpos()

    def drive_hand_joints(self, pos, ori, *args, **kwargs):
        self.hand.set_ee_pose(pos, ori, wait=False)

    def _get_ee_pose_joints(self):
        ee_pose = np.concatenate(self.hand.get_ee_pose()[:2])
        return ee_pose[:3], ee_pose[3:]

class FloatingHandDriven:
    def __init__(self, urdf_file, pb_client, init_pos=[0.0, 0.0, 0.0]):
        self.pb_client = pb_client
        self.init_pos = init_pos
        self.urdf_file = urdf_file
        self.hand_id = p.loadURDF(self.urdf_file, [self.init_pos[0], self.init_pos[1], self.init_pos[2]])

        with open(self.urdf_file, 'r') as f:
            self.urdf_str = f.read()
        # self.num_ik = trac_ik.IK('base_world', 'base_link', urdf_string=self.urdf_str)
        self.ik_pos_tol, self.ik_ori_tol = 0.001, 0.01
        self.grasp_obj_cid = None

        self._setup_joints()            

    def _setup_joints(self):    
        max_force = 1e3
        self.arm_jnt_names = [
            'base_joint_pris_x', 
            'base_joint_pris_y', 
            'base_joint_pris_z',
        ]
        self._home_position = [0.0, 0.0, 0.0]
        self._max_torques = [max_force] * 3 

        self.arm_dof = len(self.arm_jnt_names)
        self._home_position = [0.0] * self.arm_dof

        # joint damping for inverse kinematics
        self._ik_jd = 0.0005

        self.ee_link_jnt = 'sphere_target_joint'

        self._build_jnt_id()

    def _build_jnt_id(self):
        """
        Build the mapping from the joint name to joint index.
        """
        self.jnt_to_id = {}
        self.non_fixed_jnt_names = []
        for i in range(p.getNumJoints(self.hand_id)):
            info = p.getJointInfo(self.hand_id, i)
            jnt_name = info[1].decode('UTF-8')
            self.jnt_to_id[jnt_name] = info[0]
            if info[2] != p.JOINT_FIXED:
                self.non_fixed_jnt_names.append(jnt_name)

        self._ik_jds = [self._ik_jd] * len(self.non_fixed_jnt_names)
        self.ee_link_id = self.jnt_to_id[self.ee_link_jnt]
        self.arm_jnt_ids = [self.jnt_to_id[jnt] for jnt in self.arm_jnt_names]
        self.arm_jnt_ik_ids = [self.non_fixed_jnt_names.index(jnt)
                               for jnt in self.arm_jnt_names]

    def go_home(self, ignore_physics=False):
        self.set_jpos(self._home_position, ignore_physics=ignore_physics)

    def set_home_position(self, pos):
        self._home_position = pos

    def set_jpos(self, position, joint_name=None,
                 wait=True, ignore_physics=False, *args, **kwargs):
        """
        Move the arm to the specified joint position(s).

        Args:
            position (float or list): desired joint position(s).
            joint_name (str): If not provided, position should be a list
                and all the actuated joints will be moved to the specified
                positions. If provided, only the specified joint will
                be moved to the desired joint position.
            wait (bool): whether to block the code and wait
                for the action to complete.
            ignore_physics (bool): hard reset the joints to the target joint
                positions. It's best only to do this at the start,
                while not running the simulation. It will overrides
                all physics simulation.

        Returns:
            bool: A boolean variable representing if the action is successful
            at the moment when the function exits.
        """
        position = copy.deepcopy(position)
        success = False
        if joint_name is None:
            if len(position) != self.arm_dof:
                raise ValueError('Position should contain %d '
                                 'elements if the joint_name'
                                 ' is not provided' % self.arm_dof)
            tgt_pos = position
            if ignore_physics:
                # we need to set the joints to velocity control mode
                # so that the reset takes effect. Otherwise, the joints
                # will just go back to the original positions
                self.set_jvel([0.] * self.arm_dof)
                for idx, jnt in enumerate(self.arm_jnt_names):
                    self.reset_joint_state(
                        jnt,
                        tgt_pos[idx]
                    )
                success = True
            else:
                p.setJointMotorControlArray(self.hand_id,
                                                   self.arm_jnt_ids,
                                                   p.POSITION_CONTROL,
                                                   targetPositions=tgt_pos,
                                                   forces=self._max_torques)

                # p.setJointMotorControlArray(self.hand_id,
                #                                    self.arm_jnt_ids,
                #                                    p.POSITION_CONTROL,
                #                                    targetPositions=tgt_pos,
                #                                    forces=[1e7]*self.arm_dof)

                # p.setJointMotorControlArray(self.hand_id,
                #                                    self.arm_jnt_ids,
                #                                    p.POSITION_CONTROL,
                #                                    targetPositions=tgt_pos,
                #                                    forces=[1e2]*3 + [1e7]*3)
        else:
            if joint_name not in self.arm_jnt_names:
                raise TypeError('Joint name [%s] is not in the arm'
                                ' joint list!' % joint_name)
            else:
                tgt_pos = position
                arm_jnt_idx = self.arm_jnt_names.index(joint_name)
                max_torque = self._max_torques[arm_jnt_idx]
                jnt_id = self.jnt_to_id[joint_name]
            if ignore_physics:
                self.set_jvel(0., joint_name)
                self.reset_joint_state(joint_name, tgt_pos)
                success = True
            else:
                p.setJointMotorControl2(self.hand_id,
                                               jnt_id,
                                               p.POSITION_CONTROL,
                                               targetPosition=tgt_pos,
                                               force=max_torque)
        if wait and not ignore_physics:
            success = wait_to_reach_jnt_goal(
                tgt_pos,
                get_func=self.get_jpos,
                joint_name=joint_name,
                get_func_derv=self.get_jvel,
                timeout=5.0,
                max_error=0.001
            )
        return success

    def set_jvel(self, velocity, joint_name=None, wait=False, *args, **kwargs):
        """
        Move the arm with the specified joint velocity(ies).

        Args:
            velocity (float or list): desired joint velocity(ies).
            joint_name (str): If not provided, velocity should be a list
                and all the actuated joints will be moved in the specified
                velocities. If provided, only the specified joint will
                be moved in the desired joint velocity.
            wait (bool): whether to block the code and wait
                for the action to complete.

        Returns:
            bool: A boolean variable representing if the action is successful
            at the moment when the function exits.
        """
        velocity = copy.deepcopy(velocity)
        success = False
        if joint_name is None:
            velocity = copy.deepcopy(velocity)
            if len(velocity) != self.arm_dof:
                raise ValueError('Velocity should contain %d elements '
                                 'if the joint_name is not '
                                 'provided' % self.arm_dof)
            tgt_vel = velocity
            p.setJointMotorControlArray(self.hand_id,
                                               self.arm_jnt_ids,
                                               p.VELOCITY_CONTROL,
                                               targetVelocities=tgt_vel,
                                               forces=self._max_torques)
        else:
            if joint_name not in self.arm_jnt_names:
                raise TypeError('Joint name [%s] is not in the arm'
                                ' joint list!' % joint_name)
            else:
                tgt_vel = velocity
                arm_jnt_idx = self.arm_jnt_names.index(joint_name)
                max_torque = self._max_torques[arm_jnt_idx]
                jnt_id = self.jnt_to_id[joint_name]
            p.setJointMotorControl2(self.hand_id,
                                           jnt_id,
                                           p.VELOCITY_CONTROL,
                                           targetVelocity=tgt_vel,
                                           force=max_torque)
        if wait:
            success = wait_to_reach_jnt_goal(
                tgt_vel,
                get_func=self.get_jvel,
                joint_name=joint_name,
                timeout=5.0,
                max_error=0.1
            )
        else:
            success = True
        return success

    def reset_joint_state(self, jnt_name, jpos, jvel=0):
        """
        Reset the state of the joint. It's best only to do
        this at the start, while not running the simulation.
        It will overrides all physics simulation.

        Args:
            jnt_name (str): joint name.
            jpos (float): target joint position.
            jvel (float): optional, target joint velocity.

        """
        p.resetJointState(self.hand_id,
                                 self.jnt_to_id[jnt_name],
                                 targetValue=jpos,
                                 targetVelocity=jvel)

    def get_jpos(self, joint_name=None):
        """
        Return the joint position(s) of the arm.

        Args:
            joint_name (str, optional): If it's None,
                it will return joint positions
                of all the actuated joints. Otherwise, it will
                return the joint position of the specified joint.

        Returns:
            One of the following

            - float: joint position given joint_name.
            - list: joint positions if joint_name is None
              (shape: :math:`[DOF]`).
        """
        if joint_name is None:
            states = p.getJointStates(self.hand_id,
                                      self.arm_jnt_ids)
            pos = [state[0] for state in states]
        else:
            jnt_id = self.jnt_to_id[joint_name]
            pos = p.getJointState(self.hand_id, jnt_id)[0]
        return pos

    def get_jvel(self, joint_name=None):
        """
        Return the joint velocity(ies) of the arm.

        Args:
            joint_name (str, optional): If it's None, it will return
                joint velocities of all the actuated joints. Otherwise,
                it will return the joint velocity of the specified joint.

        Returns:
            One of the following

            - float: joint velocity given joint_name.
            - list: joint velocities if joint_name is None
              (shape: :math:`[DOF]`).
        """
        if joint_name is None:
            states = p.getJointStates(self.hand_id,
                                             self.arm_jnt_ids)
            vel = [state[1] for state in states]
        else:
            jnt_id = self.jnt_to_id[joint_name]
            vel = p.getJointState(self.hand_id,
                                         jnt_id)[1]
        return vel

    def get_ee_pose(self):
        """
        Return the end effector pose.

        Returns:
            4-element tuple containing

            - np.ndarray: x, y, z position of the EE (shape: :math:`[3,]`).
            - np.ndarray: quaternion representation of the
              EE orientation (shape: :math:`[4,]`).
            - np.ndarray: rotation matrix representation of the
              EE orientation (shape: :math:`[3, 3]`).
            - np.ndarray: euler angle representation of the
              EE orientation (roll, pitch, yaw with
              static reference frame) (shape: :math:`[3,]`).
        """
        info = p.getLinkState(self.hand_id, self.ee_link_id)
        pos = info[4]
        quat = info[5]

        rot_mat = arutil.quat2rot(quat)
        euler = arutil.quat2euler(quat, axes='xyz')  # [roll, pitch, yaw]
        return np.array(pos), np.array(quat), rot_mat, euler

    def set_ee_pose(self, pos=None, ori=None, wait=True, seed=None, ignore_physics=False, try_multiple_ik=True, *args, **kwargs):
        """
        Move the end effector to the specifed pose.

        Args:
            pos (list or np.ndarray): Desired x, y, z positions in the robot's
                base frame to move to (shape: :math:`[3,]`).
            ori (list or np.ndarray, optional): It can be euler angles
                ([roll, pitch, yaw], shape: :math:`[4,]`),
                or quaternion ([qx, qy, qz, qw], shape: :math:`[4,]`),
                or rotation matrix (shape: :math:`[3, 3]`). If it's None,
                the solver will use the current end effector
                orientation as the target orientation.

        Returns:
            bool: A boolean variable representing if the action is successful
            at the moment when the function exits.
        """
        if pos is None:
            pose = self.get_ee_pose()
            pos = pose[0]
        original_pos, original_ori = copy.deepcopy(pos), copy.deepcopy(ori)
        if try_multiple_ik:
            ik_trials = 0
            while True:
                ik_trials += 1
                if ik_trials > 10:
                    raise InverseKinematicsError('set_ee_pose in Floating Palms: Failed IK attempt 10 times with perturbed ')
                try:
                    # jnt_pos = self.compute_ik(pos, ori, seed=seed)
                    jnt_pos = self.compute_ik(pos, ori)
                    break
                except InverseKinematicsError as e:
                    log_warn(e)
                    log_warn('set_ee_pose in Floating Palms: Failed IK attempt, trying again with noise')
                    # add some noise to the command 
                    pos = original_pos + np.random.rand(3)*0.001
                    ori = original_ori + np.random.rand(4)*0.001
                    ori = ori / np.linalg.norm(ori)
        else:
            jnt_pos = self.compute_ik(pos, ori, seed=seed)

        # jnt_pos = self.compute_ik(pos, ori, seed=seed)
        success = self.set_jpos(jnt_pos, wait=wait, ignore_physics=ignore_physics)
        return success

    def move_ee_xyz(self, delta_xyz, eef_step=0.005, *args, **kwargs):
        """
        Move the end-effector in a straight line without changing the
        orientation.

        Args:
            delta_xyz (list or np.ndarray): movement in x, y, z
                directions (shape: :math:`[3,]`).
            eef_step (float): interpolation interval along delta_xyz.
                Interpolate a point every eef_step distance
                between the two end points.

        Returns:
            bool: A boolean variable representing if the action is successful
            at the moment when the function exits.
        """
        # if not self.pb_client.in_realtime_mode():
        #     raise AssertionError('move_ee_xyz() can '
        #                          'only be called in realtime'
        #                          ' simulation mode')
        pos, quat, rot_mat, euler = self.get_ee_pose()
        cur_pos = np.array(pos)
        cur_rev_ori = self.get_jpos()[3:]
        delta_xyz = np.array(delta_xyz)

        waypoints = arutil.linear_interpolate_path(cur_pos,
                                                   delta_xyz,
                                                   eef_step)
        way_jnt_positions = []
        for i in range(waypoints.shape[0]):
            # tgt_jnt_poss = waypoints[i, :3].tolist() + cur_rev_ori
            tgt_jnt_poss = self.compute_ik(waypoints[i, :].flatten().tolist(),
                                           quat,
                                           **kwargs.get('ik_kwargs', dict()))
            way_jnt_positions.append(copy.deepcopy(tgt_jnt_poss))
        success = False
        for jnt_poss in way_jnt_positions:
            success = self.set_jpos(jnt_poss, **kwargs)
        return success

    def move_ee_xyz_until_touch(self, delta_xyz, eef_step=0.005, 
                                force_thresh=0.1, coll_id_pairs=None, 
                                use_force=True, return_stop_motion=False,
                                *args, **kwargs):
        """
        Move the end-effector in a straight line without changing the
        orientation.

        Args:
            delta_xyz (list or np.ndarray): movement in x, y, z
                directions (shape: :math:`[3,]`).
            eef_step (float): interpolation interval along delta_xyz.
                Interpolate a point every eef_step distance
                between the two end points.

        Returns:
            bool: A boolean variable representing if the action is successful
            at the moment when the function exits.
        """
        pos, quat, rot_mat, euler = self.get_ee_pose()
        cur_pos = np.array(pos)
        cur_rev_ori = self.get_jpos()[3:]
        delta_xyz = np.array(delta_xyz)

        waypoints = arutil.linear_interpolate_path(cur_pos,
                                                   delta_xyz,
                                                   eef_step)
        way_jnt_positions = []
        for i in range(waypoints.shape[0]):
            # tgt_jnt_poss = waypoints[i, :3].tolist() + cur_rev_ori
            tgt_jnt_poss = self.compute_ik(waypoints[i, :].flatten().tolist(),
                                           quat,
                                           **kwargs.get('ik_kwargs', dict()))
            way_jnt_positions.append(copy.deepcopy(tgt_jnt_poss))

        success = False
        stop_motion = False
        for jnt_poss in way_jnt_positions:
            success = self.set_jpos(jnt_poss, **kwargs)

            contacts = self.pb_client.getContactPoints(
                coll_id_pairs[0][0], # body_id
                coll_id_pairs[1][0], # body_id,
                coll_id_pairs[0][1], # link_id,
                coll_id_pairs[1][1]) # link_id

            normal_force_list = []
            for pt in contacts:
                normal_force_list.append(pt[-5])

            if use_force:
                log_debug(f'[FloatingHandDriven move_ee_xyz_until_touch] Checking touch via force thresh') 
                if len(contacts) > 0:
                    log_debug(f'[FloatingHandDriven move_ee_xyz_until_touch] Max force: {max(normal_force_list)}') 
                    stop_motion = max(normal_force_list) > force_thresh
                else:
                    stop_motion = False
            else:
                log_debug(f'[FloatingHandDriven move_ee_xyz_until_touch] Checking touch via contact')
                stop_motion = len(contact) > 0

            if stop_motion:
                log_debug(f'[FloatingHandDriven move_ee_xyz_until_touch] Stopping motion') 
                break
        
        if return_stop_motion:
            return success, stop_motion
        return success

    def _correct_ik(self, next_joints):
        last_joints = self.get_jpos()
        if not isinstance(next_joints, np.ndarray):
            next_joints = np.asarray(next_joints)
        if not isinstance(last_joints, np.ndarray):
            last_joints = np.asarray(last_joints)
        delta_joints = next_joints - last_joints
        abs_delta = np.abs(delta_joints)

        correction_idxs = np.where(abs_delta > np.pi)[0]

        corrected_jnts = copy.deepcopy(next_joints)
        for idx in correction_idxs:
            # print(idx, last_joints[idx], next_joints[idx], delta_joints[idx])
            # print(delta_joints[idx])
            if delta_joints[idx] > 0:
                new_next_jnt = last_joints[idx] - (2*np.pi - delta_joints[idx])
                # new_next_jnt = last_joints[idx] - ((2*np.pi - delta_joints[idx]) % np.pi)
                # new_next_jnt = last_joints[idx] - (abs_delta[idx] % np.pi)
            else:
                new_next_jnt = last_joints[idx] + (2*np.pi + delta_joints[idx])
                # new_next_jnt = last_joints[idx] + ((2*np.pi + delta_joints[idx]) % np.pi)
                # new_next_jnt = last_joints[idx] + (abs_delta[idx] % np.pi)
            # print(new_next_jnt)
            corrected_jnts[idx] = new_next_jnt
        return corrected_jnts

    def compute_ik(self, pos, ori=None, ns=False, *args, **kwargs):
        """
        Compute the inverse kinematics solution given the
        position and orientation of the end effector.

        Args:
            pos (list or np.ndarray): position (shape: :math:`[3,]`).
            ori (list or np.ndarray): orientation. It can be euler angles
                ([roll, pitch, yaw], shape: :math:`[3,]`), or
                quaternion ([qx, qy, qz, qw], shape: :math:`[4,]`),
                or rotation matrix (shape: :math:`[3, 3]`).
            ns (bool): whether to use the nullspace options in pybullet,
                True if nullspace should be used. Defaults to False.

        Returns:
            list: solution to inverse kinematics, joint angles which achieve
            the specified EE pose (shape: :math:`[DOF]`).
        """
        kwargs.setdefault('jointDamping', self._ik_jds)

        if ori is not None:
            ori = arutil.to_quat(ori)
            jnt_poss = self.pb_client.calculateInverseKinematics(
                self.hand_id,
                self.ee_link_id,
                pos,
                ori,
                **kwargs)
        else:
            jnt_poss = self.pb_client.calculateInverseKinematics(
                self.hand_id,
                self.ee_link_id,
                pos,
                **kwargs)
        jnt_poss = list(map(arutil.ang_in_mpi_ppi, jnt_poss))
        arm_jnt_poss = [jnt_poss[i] for i in self.arm_jnt_ik_ids]
        return arm_jnt_poss

    def constraint_grasp_close(self, obj_id):
        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))

        ee_link_id = self.ee_link_id
        ee_pose_world = np.concatenate(self.get_ee_pose()[:2]).tolist()
        ee_pose_world = util.list2pose_stamped(ee_pose_world)

        obj_pose_ee = util.convert_reference_frame(
            pose_source=obj_pose_world,
            pose_frame_target=ee_pose_world,
            pose_frame_source=util.unit_pose()
        )
        obj_pose_ee_list = util.pose_stamped2list(obj_pose_ee)

        cid = p.createConstraint(
            parentBodyUniqueId=self.hand_id,
            parentLinkIndex=ee_link_id,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=obj_pose_ee_list[:3],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=obj_pose_ee_list[3:])

        self.grasp_obj_cid = cid
        return cid

    def constraint_grasp_open(self, cid=None):
        if cid is None:
            if self.grasp_obj_cid is not None:
                p.removeConstraint(self.grasp_obj_cid)
                self.grasp_obj_cid = None
        else:
            if cid is not None:
                p.removeConstraint(cid)
