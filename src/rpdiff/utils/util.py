import os, os.path as osp
from typing import List, Tuple, Union
import copy
import sys
import yaml
from PIL import Image
import numpy as np
from numpy.lib.npyio import NpzFile
from yacs.config import CfgNode
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import math
from matplotlib import cm
import meshcat
import meshcat.geometry as mcg
from trimesh.base import Trimesh
from trimesh.scene.scene import Scene
from trimesh.util import concatenate as trimesh_concatenate


# General config
def load_config(path: str, default_path: str=None) -> dict:
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


def npz2dict(npz: NpzFile) -> dict:
    out_dict = {}
    for k in npz.files:
        out_dict[k] = npz[k]
    return out_dict


def set_if_not_none(var, cfg_input):
    if cfg_input is not None:
        var = cfg_input
    return var


def exists_and_true(d: dict, st: str) -> bool:
    if st in d:
        if d[st]:
            return True 
    return False


def hasattr_and_not_none(o, st: str) -> bool:
    if hasattr(o, st):
        attr = getattr(o, st)
        if attr is not None:
            return True
    return False


def hasattr_and_true(o, st: str) -> bool:
    if hasattr(o, st):
        attr = getattr(o, st)
        if attr:
            return True
    return False


def np2img(np_array: np.ndarray, img_file: str) -> None:
    im = Image.fromarray(np_array)
    im.save(img_file)


def safe_makedirs(dirname: str) -> None:
    if not osp.exists(dirname):
        os.makedirs(dirname)


def safe_join_path(path1, path2, return_none_str=True):
    if path1 is not None and path2 is not None:
        return osp.join(path1, path2)
    else:
        if return_none_str:
            return 'None'
        else:
            return None


def signal_handler(sig, frame):
    """
    Capture exit signal from keyboard
    """
    print('Exit')
    sys.exit(0)


def cn2dict(config):
    """
    Convert a YACS CfgNode config object into a
    dictionary

    Args:
        config (CfgNode): Config object

    Returns:
        dict: Dictionary version of config
    """
    out_dict = {}
    items = config.items()
    for key, val in items:
        if isinstance(val, CfgNode):
            ret = cn2dict(val)
        else:
            ret = val
        out_dict[key] = ret
    return out_dict


def crop_pcd(
        raw_pts: np.ndarray, 
        x: List[float]=[0.0, 0.7], 
        y: List[float]=[-0.4, 0.4], 
        z: List[float]=[0.9, 1.5]) -> np.ndarray:
    npw = np.where(
            (raw_pts[:, 0] > min(x)) & (raw_pts[:, 0] < max(x)) &
            (raw_pts[:, 1] > min(y)) & (raw_pts[:, 1] < max(y)) &
            (raw_pts[:, 2] > min(z)) & (raw_pts[:, 2] < max(z)))
    return raw_pts[npw[0], :]


class Position:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.


class Orientation:
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 0.
        self.w = 0.


class Pose:
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation


class Header:
    def __init__(self):
        self.frame_id = "world"


class PoseStamped():
    def __init__(self):
        position = Position()
        orientation = Orientation()
        pose = Pose(position, orientation)
        header = Header()
        self.pose = pose
        self.header = header


def angle_from_3d_vectors(u: np.ndarray, v: np.ndarray) -> float:
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    u_dot_v = np.dot(u, v)
    return np.arccos(u_dot_v) / (u_norm * v_norm)


def pose_from_matrix(matrix: np.ndarray) -> PoseStamped:
    # trans = tf.transformations.translation_from_matrix(matrix)
    # quat = tf.transformations.quaternion_from_matrix(matrix)
    quat = R.from_matrix(matrix[:3, :3]).as_quat()
    trans = matrix[:-1, -1]
    pose = list(trans) + list(quat)
    pose = list2pose_stamped(pose)
    return pose


def list2pose_stamped(pose: List[float]) -> PoseStamped:
    msg = PoseStamped()
    msg.pose.position.x = pose[0]
    msg.pose.position.y = pose[1]
    msg.pose.position.z = pose[2]
    msg.pose.orientation.x = pose[3]
    msg.pose.orientation.y = pose[4]
    msg.pose.orientation.z = pose[5]
    msg.pose.orientation.w = pose[6]
    return msg


def unit_pose() -> PoseStamped:
    return list2pose_stamped([0, 0, 0, 0, 0, 0, 1])


def convert_reference_frame(
        pose_source: PoseStamped, 
        pose_frame_target: PoseStamped, 
        pose_frame_source: PoseStamped) -> PoseStamped:
    T_pose_source = matrix_from_pose(pose_source)
    pose_transform_target2source = get_transform(
        pose_frame_source, pose_frame_target)
    T_pose_transform_target2source = matrix_from_pose(
        pose_transform_target2source)
    T_pose_target = np.matmul(T_pose_transform_target2source, T_pose_source)
    pose_target = pose_from_matrix(T_pose_target)
    return pose_target


def pose_stamped2list(msg: PoseStamped) -> List:
    return [float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
            ]


def pose_stamped2np(msg: PoseStamped) -> np.ndarray:
    return np.asarray(pose_stamped2list(msg))


def get_transform(
        pose_frame_target: PoseStamped, 
        pose_frame_source: PoseStamped) -> PoseStamped:
    """
    Find transform that transforms pose source to pose target
    :param pose_frame_target:
    :param pose_frame_source:
    :return:
    """
    #both poses must be expressed in same reference frame
    T_target_world = matrix_from_pose(pose_frame_target)
    T_source_world = matrix_from_pose(pose_frame_source)
    T_relative_world = np.matmul(T_target_world, np.linalg.inv(T_source_world))
    pose_relative_world = pose_from_matrix(T_relative_world)
    return pose_relative_world


def matrix_from_pose(pose: PoseStamped) -> np.ndarray:
    pose_list = pose_stamped2list(pose)
    return matrix_from_list(pose_list)


def matrix_from_list(pose_list: List[float]) -> np.ndarray:
    trans = pose_list[:3]
    quat = pose_list[3:]

    T = np.eye(4)
    T[:-1, :-1] = R.from_quat(quat).as_matrix()
    T[:-1, -1] = trans
    return T


def scale_matrix(factor: Union[float, List, np.ndarray] , origin: np.ndarray=None) -> np.ndarray:
    """Return matrix to scale by factor around origin in direction.
    Use factor -1 for point symmetry.
    """
    if not isinstance(factor, list) and not isinstance(factor, np.ndarray):
        M = np.diag([factor, factor, factor, 1.0])
    else:
        assert len(factor) == 3, 'If applying different scaling per dimension, must pass in 3-element list or array'
        #M = np.diag([factor[0], factor[1], factor[2], 1.0])
        M = np.eye(4)
        M[0, 0] = factor[0]
        M[1, 1] = factor[1]
        M[2, 2] = factor[2]
    if origin is not None:
        M[:3, 3] = origin[:3]
        M[:3, 3] *= 1.0 - factor
    return M


def interpolate_pose(
        pose_initial: PoseStamped, 
        pose_final: PoseStamped, 
        N: int) -> List[PoseStamped]:
    pose_initial_list = pose_stamped2list(pose_initial)
    pose_final_list = pose_stamped2list(pose_final)
    trans_initial = pose_initial_list[0:3]
    quat_initial = pose_initial_list[3:7]
     # onvert to pyquaterion convertion (w,x,y,z)
    trans_final = pose_final_list[0:3]
    quat_final = pose_final_list[3:7]

    trans_interp_total = [np.linspace(trans_initial[0], trans_final[0], num=N),
                          np.linspace(trans_initial[1], trans_final[1], num=N),
                          np.linspace(trans_initial[2], trans_final[2], num=N)]
    
    key_rots = R.from_quat([quat_initial, quat_final])
    slerp = Slerp(np.arange(2), key_rots)
    interp_rots = slerp(np.linspace(0, 1, N))
    quat_interp_total = interp_rots.as_quat()    

    pose_interp = []
    for counter in range(N):
        pose_tmp = [
            trans_interp_total[0][counter],
            trans_interp_total[1][counter],
            trans_interp_total[2][counter],
            quat_interp_total[counter][0], #return in ROS ordering w,x,y,z
            quat_interp_total[counter][1],
            quat_interp_total[counter][2],
            quat_interp_total[counter][3],
        ]
        pose_interp.append(list2pose_stamped(pose_tmp))
    return pose_interp


def transform_pose(pose_source: PoseStamped, pose_transform: PoseStamped) -> PoseStamped:
    T_pose_source = matrix_from_pose(pose_source)
    T_transform_source = matrix_from_pose(pose_transform)
    T_pose_final_source = np.matmul(T_transform_source, T_pose_source)
    pose_final_source = pose_from_matrix(T_pose_final_source)
    return pose_final_source


def vec_from_pose(pose: PoseStamped) -> Tuple[np.ndarray]:
    #get unit vectors of rotation from pose
    quat = pose.pose.orientation
    # T = tf.transformations.quaternion_matrix([quat.x, quat.y, quat.z, quat.w])
    T = np.zeros((4, 4,))
    T[-1, -1] = 1
    T[:3, :3] = R.from_quat([quat.x, quat.y, quat.z, quat.w]).as_matrix()

    x_vec = T[0:3, 0]
    y_vec = T[0:3, 1]
    z_vec = T[0:3, 2]
    return x_vec, y_vec, z_vec


def list_to_pose(pose_list: List[float]) -> Pose:
    msg = Pose()
    msg.position.x = pose_list[0]
    msg.position.y = pose_list[1]
    msg.position.z = pose_list[2]
    msg.orientation.x = pose_list[3]
    msg.orientation.y = pose_list[4]
    msg.orientation.z = pose_list[5]
    msg.orientation.w = pose_list[6]
    return msg


def pose_to_list(pose: Pose) -> List[float]:
    pose_list = []
    pose_list.append(pose.position.x)
    pose_list.append(pose.position.y)
    pose_list.append(pose.position.z)
    pose_list.append(pose.orientation.x)
    pose_list.append(pose.orientation.y)
    pose_list.append(pose.orientation.z)
    pose_list.append(pose.orientation.w)
    return pose_list


def quat_multiply(quat1: np.ndarray, quat2: np.ndarray) -> np.ndarray:
    """
    Quaternion mulitplication.

    Args:
        quat1 (list or np.ndarray): first quaternion [x,y,z,w]
            (shape: :math:`[4,]`).
        quat2 (list or np.ndarray): second quaternion [x,y,z,w]
            (shape: :math:`[4,]`).

    Returns:
        np.ndarray: quat1 * quat2 (shape: :math:`[4,]`).
    """
    r1 = R.from_quat(quat1)
    r2 = R.from_quat(quat2)
    r = r1 * r2
    return r.as_quat()


def quat_inverse(quat: np.ndarray) -> np.ndarray:
    """
    Return the quaternion inverse.

    Args:
        quat (list or np.ndarray): quaternion [x,y,z,w] (shape: :math:`[4,]`).

    Returns:
        np.ndarray: inverse quaternion (shape: :math:`[4,]`).
    """
    r = R.from_quat(quat)
    return r.inv().as_quat()


def pose_difference_np(pose: np.ndarray, pose_ref: np.ndarray, rs: bool=False) -> Tuple[float]:
    """
    Compute the approximate difference between two poses, by comparing
    the norm between the positions and using the quaternion difference
    to compute the rotation similarity

    Args:
        pose (np.ndarray): pose 1, in form [pos, ori], where
            pos (shape: [3,]) is of the form [x, y, z], and ori (shape: [4,])
            if of the form [x, y, z, w]
        pose_ref (np.ndarray): pose 2, in form [pos, ori], where
            pos (shape: [3,]) is of the form [x, y, z], and ori (shape: [4,])
            if of the form [x, y, z, w]
        rs (bool): If True, use rotation_similarity metric for orientation error.
            Otherwise use geodesic distance. Defaults to False

    Returns:
        2-element tuple containing:
        - np.ndarray: Euclidean distance between positions
        - np.ndarray: Quaternion difference between the orientations
    """
    if isinstance(pose, list):
        pose = np.asarray(pose)
    if isinstance(pose_ref, list):
        pose_ref = np.asarray(pose_ref)
    if len(pose.shape) < 2:
        pose = pose[None, :]
    
    pos_1, pos_2 = pose[:, :3], pose_ref[:3]
    ori_1, ori_2 = pose[:, 3:], pose_ref[3:]

    pos_diff = pos_1 - pos_2
    pos_error = np.linalg.norm(pos_diff, axis=1)

    quat_diff = quat_multiply(quat_inverse(ori_1), ori_2)
    rot_similarity = np.abs(quat_diff[:, 3])

    # dot_prod = np.dot(ori_1, ori_2)
    dot_prod1 = np.clip(np.dot(ori_1, ori_2), 0, 1)
    angle_diff1 = np.arccos(2*dot_prod1**2 - 1)

    dot_prod2 = np.clip(np.dot(ori_1, -ori_2), 0, 1)
    angle_diff2 = np.arccos(2*dot_prod2**2 - 1)    

    if rs:
        angle_diff1 = 1 - rot_similarity
        angle_diff2 = np.inf
    return pos_error, np.min(np.vstack((angle_diff1, angle_diff2)), axis=0)


def ori_difference(ori_1: np.ndarray, ori_2: np.ndarray) -> float:
    dot_prod1 = np.clip(np.dot(ori_1, ori_2), 0, 1)
    angle_diff1 = np.arccos(2*dot_prod1**2 - 1)

    dot_prod2 = np.clip(np.dot(ori_1, -ori_2), 0, 1)
    angle_diff2 = np.arccos(2*dot_prod2**2 - 1)    
    return min(angle_diff1, angle_diff2)


def pose_from_vectors(x_vec: np.ndarray, y_vec: np.ndarray, z_vec: np.ndarray, trans: np.ndarray) -> np.ndarray:
    # Normalized frame
    hand_orient_norm = np.vstack((x_vec, y_vec, z_vec))
    hand_orient_norm = hand_orient_norm.transpose()
    quat = R.from_matrix(hand_orient_norm).as_quat()
    pose = list2pose_stamped(list(trans) + list(quat))
    return pose


def transform_vectors(vectors: np.ndarray, pose_transform: PoseStamped) -> np.ndarray:
    """Transform a set of vectors

    Args:
        vectors (np.ndarray): Numpy array of vectors, size
            [N, 3], where each row is a vector [x, y, z]
        pose_transform (PoseStamped): PoseStamped object defining the transform

    Returns:
        np.ndarray: Size [N, 3] with transformed vectors in same order as input
    """
    vectors_homog = np.ones((4, vectors.shape[0]))
    vectors_homog[:-1, :] = vectors.T

    T_transform = matrix_from_pose(pose_transform)

    vectors_trans_homog = np.matmul(T_transform, vectors_homog)
    vectors_trans = vectors_trans_homog[:-1, :].T
    return vectors_trans


def sample_orthogonal_vector(reference_vector: np.ndarray) -> np.ndarray:
    """Sample a random unit vector that is orthogonal to the specified reference

    Args:
        reference_vector (np.ndarray): Numpy array with
            reference vector, [x, y, z]. Cannot be all zeros

    Return:
        np.ndarray: Size [3,] that is orthogonal to specified vector
    """
    # y_unnorm = np.zeros(reference_vector.shape)

    # nonzero_inds = np.where(reference_vector)[0]
    # ind_1 = random.sample(nonzero_inds, 1)[0]
    # while True:
    #     ind_2 = np.random.randint(3)
    #     if ind_1 != ind_2:
    #         break

    # y_unnorm[ind_1] = reference_vector[ind_2]
    # y_unnorm[ind_2] = -reference_vector[ind_1]
    # y = y_unnorm / np.linalg.norm(y_unnorm)
    rand_vec = np.random.rand(3) * 2 - 1
    # rand_vec = np.random.rand(3) * -1.0
    y_unnorm = project_point2plane(rand_vec, reference_vector, [0, 0, 0])[0]
    y = y_unnorm / np.linalg.norm(y_unnorm)
    return y


def project_point2plane(
        point: np.ndarray, 
        plane_normal: np.ndarray, 
        plane_points: np.ndarray) -> np.ndarray:
    '''project a point to a plane'''
    point_plane = plane_points[0]
    w = point - point_plane
    dist = (np.dot(plane_normal, w) / np.linalg.norm(plane_normal))
    projected_point = point - dist * plane_normal / np.linalg.norm(plane_normal)
    return projected_point, dist


def rand_body_yaw_transform(
        pos: np.ndarray, min_theta: float=0.0, 
        max_theta: float=2*np.pi) -> np.ndarray:
    """Given some initial position, sample a Transform that is
    a pure yaw about the world frame orientation, with
    the origin at the current pose position

    Args:
        pos (np.ndarray): Current position in the world frame 
        min (float, optional): Minimum boundary for sample
        max (float, optional): Maximum boundary for sample

    Returns:
        np.ndarray: Transformation matrix
    """
    if isinstance(pos, list):
        pos = np.asarray(pos)    
    trans_to_origin = pos
    theta = np.random.random() * (max_theta - min_theta) + min_theta
    yaw = R.from_euler('xyz', [0, 0, theta]).as_matrix()[:3, :3]

    # translate the source to the origin
    T_0 = np.eye(4)
    T_0[:-1, -1] = -trans_to_origin

    # apply pure rotation in the world frame
    T_1 = np.eye(4)
    T_1[:-1, :-1] = yaw

    # translate in [x, y] back away from origin
    T_2 = np.eye(4)
    T_2[0, -1] = trans_to_origin[0]
    T_2[1, -1] = trans_to_origin[1]
    T_2[2, -1] = trans_to_origin[2]
    yaw_trans = np.matmul(T_2, np.matmul(T_1, T_0))
    return yaw_trans


def transform_pcd(pcd: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if pcd.shape[1] != 4:
        pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
    pcd_new = np.matmul(transform, pcd.T)[:-1, :].T
    return pcd_new


def center_pcd(pcd: np.ndarray, ref_pcd: np.ndarray=None) -> np.ndarray:
    """
    Args:
        pcd (np.ndarray): Point cloud to center, either N x 3 or B x N x 3
        ref_pcd (np.ndarray): Point cloud whose mean should be used to center, 
            either N x 3 or B x N x 3. If None, will use pcd as the reference

    Returns:
        np.ndarray: Centered point clouds, either N x 3 or B x N x 3
    """
    if ref_pcd is None:
        ref_pcd = pcd

    if ref_pcd.ndim < 3:
        pcd_means = np.mean(ref_pcd, 0)
    else:
        pcd_means = np.mean(ref_pcd, 1)

    if pcd.ndim < 3:
        # a single N x 3 point cloud
        return pcd - pcd_means
    else:
        # over a batch, B x N x 3 point clouds
        return pcd - pcd_means


def rotate_pcd_center(
        pcd: np.ndarray, rot_mat: np.ndarray, 
        pcd_ref: np.ndarray=None, leave_centered: bool=False, 
        pcd_mean: np.ndarray=None) -> np.ndarray:
    """
    Applied a rotation about the centroid of a point cloud

    Args:
        pcd (np.ndarray): N x 3
        rot_mat (np.ndarray): 3 x 3

    Returns:
        np.ndarray: N x 3
    """
    if pcd_mean is None:
        if pcd_ref is None:
            pcd_ref = pcd
        pcd_mean = np.mean(pcd_ref, axis=0)
    pcd_cent = pcd - pcd_mean
    pcd_cent_rot = np.matmul(rot_mat, pcd_cent.T).T
    if leave_centered:
        return pcd_cent_rot
    pcd_new = pcd_cent_rot + pcd_mean
    return pcd_new


def form_tf_mat_cent_pcd_rot(tf_mat: np.ndarray, pcd: np.ndarray) -> np.ndarray:
    """
    Create 4x4 homogenous transformation matrix that rotates a point cloud
    in place and then translates the rotated shape. This occurs by mean-centering,
    rotating, and then translating back + adding an additional traslation

    Args:
        tf_mat (np.ndarray): 4x4
        pcd (np.ndarray): Nx3

    Returns: 
        np.ndarray: 4x4
    """
    tf1 = np.eye(4); tf1[:-1, -1] = -1.0 * np.mean(pcd, axis=0)
    tf2 = np.eye(4); tf2[:-1, :-1] = tf_mat[:-1, :-1]
    tf3 = np.eye(4); tf3[:-1, -1] = np.mean(pcd, axis=0) + tf_mat[:-1, -1]

    tf_full = np.matmul(tf3, np.matmul(tf2, tf1))

    return tf_full


def downsample_pcd_perm(
        pcd: np.ndarray, 
        n_pts: int, 
        rix: np.ndarray=None, 
        return_perm: bool=False) -> Union[np.ndarray, Tuple[np.ndarray]]:
    """
    Downsample a 3D point cloud by uniformly sampling point indices

    Args: 
        pcd (np.ndarray): Nx3
        n_pts (int): Number of points to select
        rix (np.ndarray): Array of integers that are pre-computed
        return_perm (bool): If True, also output the 
    """
    if rix is None:
        rix = np.random.permutation(pcd.shape[0])
    pcd_ds = pcd[rix[:n_pts]]
    if return_perm:
        return pcd_ds, rix
    return pcd_ds


# import healpy as hp
# from https://github.com/google-research/google-research/blob/3ed7475fef726832c7288044c806481adc6de827/implicit_pdf/models.py#L381

def generate_healpix_grid(recursion_level: int=None, size: int=None) -> np.ndarray:
  """Generates an equivolumetric grid on SO(3) following Yershova et al. (2010).
  Uses a Healpix grid on the 2-sphere as a starting point and then tiles it
  along the 'tilt' direction 6*2**recursion_level times over 2pi.
  Args:
    recursion_level: An integer which determines the level of resolution of the
      grid.  The final number of points will be 72*8**recursion_level.  A
      recursion_level of 2 (4k points) was used for training and 5 (2.4M points)
      for evaluation.
    size: A number of rotations to be included in the grid.  The nearest grid
      size in log space is returned.
  Returns:
    (N, 3, 3) array of rotation matrices, where N=72*8**recursion_level.
  """
  import healpy as hp  # pylint: disable=g-import-not-at-top
  # from airobot.utils import common

  assert not(recursion_level is None and size is None)
  if size:
    recursion_level = max(int(np.round(np.log(size/72.)/np.log(8.))), 0)
  number_per_side = 2**recursion_level
  number_pix = hp.nside2npix(number_per_side)
  s2_points = hp.pix2vec(number_per_side, np.arange(number_pix))
  s2_points = np.stack([*s2_points], 1)

  # Take these points on the sphere and
  azimuths = np.arctan2(s2_points[:, 1], s2_points[:, 0])
  tilts = np.linspace(0, 2*np.pi, 6*2**recursion_level, endpoint=False)
  polars = np.arccos(s2_points[:, 2])
  grid_rots_mats = []
  for tilt in tilts:
    # Build up the rotations from Euler angles, zyz format
    # rot_mats = tfg.rotation_matrix_3d.from_euler(
    #     np.stack([azimuths,
    #               np.zeros(number_pix),
    #               np.zeros(number_pix)], 1))
    # rot_mats = rot_mats @ tfg.rotation_matrix_3d.from_euler(
    #     np.stack([np.zeros(number_pix),
    #               np.zeros(number_pix),
    #               polars], 1))
    # rot_mats = rot_mats @ tf.expand_dims(
    #     tfg.rotation_matrix_3d.from_euler([tilt, 0., 0.]), 0)

    euler = np.stack([azimuths, np.zeros(number_pix), np.zeros(number_pix)], 1)
    # rot_mats = common.euler2rot(euler)
    rot_mats = R.from_euler('xyz', euler).as_matrix()

    euler2 = np.stack([np.zeros(number_pix), np.zeros(number_pix), polars], 1)
    # rot_mats = rot_mats @ common.euler2rot(euler2)
    rot_mats = rot_mats @ R.from_euler('xyz', euler2).as_matrix()

    euler3 = [tilt, 0, 0]
    # rot_mats = rot_mats @ common.euler2rot(euler3)
    rot_mats = rot_mats @ R.from_euler('xyz', euler3).as_matrix()

    grid_rots_mats.append(rot_mats)

  grid_rots_mats = np.concatenate(grid_rots_mats, 0)
  return grid_rots_mats


def meshcat_pcd_show(
        mc_vis: meshcat.Visualizer, point_cloud: np.ndarray, 
        color: Tuple[int]=None, name: str=None, 
        size: float=0.001, debug: bool=False) -> None:
    """
    Function to show a point cloud using meshcat. 

    Args:
        mc_vis (meshcat.Visualizer): Interface to the visualizer 
        point_cloud (np.ndarray): Shape Nx3 or 3xN
        color (np.ndarray or list): Shape (3,), using 0-255 RGB colors
        name (str): Label to give point cloud in meshcat
        size (float): Size of the points

    """
    # color_orig = copy.deepcopy(color)
    if point_cloud.shape[0] != 3:
        point_cloud = point_cloud.swapaxes(0, 1)
    if color is None:
        color = np.zeros_like(point_cloud)
    else:
        # color = int('%02x%02x%02x' % color, 16)
        if not isinstance(color, np.ndarray):
            color = np.asarray(color).astype(np.float32)
        color = np.clip(color, 0, 255)
        color = np.tile(color.reshape(3, 1), (1, point_cloud.shape[1]))
        color = color.astype(np.float32)
    if name is None:
        name = 'scene/pcd'

    if debug:
        print("here in meshcat_pcd_show")
        from IPython import embed; embed()

    mc_vis[name].set_object(
        mcg.Points(
            mcg.PointsGeometry(point_cloud, color=(color / 255)),
            mcg.PointsMaterial(size=size)
    ))


def meshcat_multiple_pcd_show(
        mc_vis: meshcat.Visualizer, point_cloud_list: List[np.ndarray], 
        color_list: List[np.ndarray]=None, name_list: List[str]=None, 
        rand_color: bool=False) -> None:
    colormap = cm.get_cmap('brg', len(point_cloud_list))
    # colormap = cm.get_cmap('gist_ncar_r', len(point_cloud_list))
    colors = [
        (np.asarray(colormap(val)) * 255).astype(np.int32) for val in np.linspace(0.05, 0.95, num=len(point_cloud_list))
    ]
    if color_list is None:
        if rand_color:
            color_list = []
            for i in range(len(point_cloud_list)):
                color_list.append((np.random.rand(3) * 255).astype(np.int32).tolist())
        else:
            color_list = colors

    if name_list is None:
        name_list = [f'scene/pcd_list_{i}' for i in range(len(point_cloud_list))]
    
    for i, pcd in enumerate(point_cloud_list):
        if pcd.shape[0] > 0:
            meshcat_pcd_show(mc_vis, pcd, color=color_list[i][:3].astype(np.int8).tolist(), name=name_list[i])

    
def meshcat_frame_show(
        mc_vis: meshcat.Visualizer, name: str, 
        transform: np.ndarray=None, length: float=0.1, 
        radius: float=0.008, opacity: float=1.) -> None:
    """
    Initializes coordinate axes of a frame T. The x-axis is drawn red,
    y-axis green and z-axis blue. The axes point in +x, +y and +z directions,
    respectively.
    Args:
        mc_vis: a meshcat.Visualizer object.
        name: (string) the name of the triad in meshcat.
        transform (np.ndarray): 4 x 4 matrix representing the pose
        length: the length of each axis in meters.
        radius: the radius of each axis in meters.
        opacity: the opacity of the coordinate axes, between 0 and 1.
    """
    delta_xyz = np.array([[length / 2, 0, 0],
    [0, length / 2, 0],
    [0, 0, length / 2]])

    axes_name = ['x', 'y', 'z']
    colors = [0xff0000, 0x00ff00, 0x0000ff]
    rotation_axes = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]

    for i in range(3):
        material = meshcat.geometry.MeshLambertMaterial(
        color=colors[i], opacity=opacity)
        mc_vis[name][axes_name[i]].set_object(
        meshcat.geometry.Cylinder(length, radius), material)
        X = meshcat.transformations.rotation_matrix(
        np.pi/2, rotation_axes[i])
        X[0:3, 3] = delta_xyz[i]
        if transform is not None:
            X = np.matmul(transform, X)
        mc_vis[name][axes_name[i]].set_transform(X)


def meshcat_trimesh_show(
        mc_vis: meshcat.Visualizer, name: str, 
        trimesh_mesh: Trimesh, color: Tuple[int]=(128, 128, 128), 
        opacity: float=1.0) -> None:
    verts = trimesh_mesh.vertices
    faces = trimesh_mesh.faces

    if not isinstance(color, tuple):
        color = tuple(color)

    color = int('%02x%02x%02x' % color, 16)

    material = meshcat.geometry.MeshLambertMaterial(color=color, reflectivity=0.0, opacity=opacity)
    mcg_mesh = meshcat.geometry.TriangularMeshGeometry(verts, faces)
    # mc_vis[name].set_object(mcg_mesh)
    mc_vis[name].set_object(mcg_mesh, material)


def trimesh_scene_to_mesh(trimesh_scene: Scene) -> Trimesh:
    verts = []
    faces = []
    meshes = []
    for i, geom in enumerate(trimesh_scene.geometry.values()):
        verts.append(geom.vertices)
        faces.append(geom.faces)
        meshes.append(geom)

    # trimesh_mesh = trimesh.Trimesh(verts, faces)
    trimesh_mesh = trimesh_concatenate(meshes)
    return trimesh_mesh
