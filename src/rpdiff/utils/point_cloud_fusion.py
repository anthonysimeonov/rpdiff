import os, sys
import os.path as osp
import numpy as np
import torch
import copy
import time
import trimesh
from scipy.spatial import ConvexHull
import open3d

from rpdiff.utils import util, path_util, trimesh_util


def preprocess_point_cloud(pcd, voxel_size,
                           radius_normal=None,
                           radius_feature=None):
    """Preprocess point cloud by applying a voxel-based downsampling,
    estimating the point cloud normals, and computing the FPFH features
    that are used by the RANSAC registration algorithm

    Args:
        pcd (open3d.geometry.PointCloud): The pointcloud to be preprocessed
        voxel_size (float): Voxel size to downsample into

    Returns:
        open3d.geometry.PointCloud: The processed pointcloud (voxelized and with
            normals estimated)
        open3d.pipelines.registration.Feature: The FPFH feature for the pointcloud
    """
    # # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    if radius_normal is None:
        radius_normal = voxel_size * 2.0
    if radius_feature is None:
        radius_feature = voxel_size * 5.0

    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = open3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh



def refine_registration(source, target, init_trans, voxel_size, distance_threshold=0.001, max_iteration=100):
    """
    Function to refine pointcloud registration result with a local ICP-based
    registration. Takes as input the result of a global registration attempt
    for initialization.

    Args:
        source (open3d.geometry.PointCloud): source pointcloud to be aligned
        target (open3d.geometry.PointCloud): target pointcloud to align to
        init_trans (np.ndarray): Result from global registration method,
            transformation used as intialization to ICP
        voxel_size (float): Voxel size that point cloud is downsampled to

    Result:
        open3d.pipelines.registration.RegistrationResult: Result of ICP registration
    """
    convergence_criteria = open3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=max_iteration,
        relative_fitness=0.0,
        relative_rmse=0.0
    )
    result = open3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_trans,
        open3d.pipelines.registration.TransformationEstimationPointToPoint(),
        convergence_criteria)
    return result


def full_registration_np(source_np, target_np, init_trans=np.eye(4), viz=False):
    """Run full global + local registration routine, using numpy array
    pointclouds as input

    Args:
        source_np (np.ndarray): Source pointcloud, numpy array
        target_np (np.ndarray): Target pointcloud, numpy array

    Returns:
        np.ndarray: Homogeneous transformation matrix result of registration
    """
    start_time = time.time()
    source_pcd = open3d.geometry.PointCloud()
    target_pcd = open3d.geometry.PointCloud()
    source_pcd.points = open3d.utility.Vector3dVector(source_np)
    target_pcd.points = open3d.utility.Vector3dVector(target_np)

    voxel_size = 0.0025

    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size,
                                                      radius_normal=1.0,
                                                      radius_feature=1.0)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

    result_icp = refine_registration(
        source_down, target_down,
        init_trans, voxel_size)
    # result_icp = refine_registration(
    #     source_pcd, target_down,
    #     init_trans, voxel_size)

    # print('Time taken for registration: ' + str(time.time() - start_time))

    if viz:
        trimesh_util.trimesh_show([target_np, util.transform_pcd(source_np, result_icp.transformation)])
        trimesh_util.trimesh_show([np.asarray(target_down.points), util.transform_pcd(np.asarray(source_down.points), result_icp.transformation)])
        trimesh_util.trimesh_show([target_np, util.transform_pcd(source_np, init_trans)])
        trimesh_util.trimesh_show([np.asarray(target_down.points), util.transform_pcd(np.asarray(source_down.points), init_trans)])
    return result_icp.transformation



class SceneGraspedObjectCollChecker:
    def __init__(self, object_pcd, scene_pcd, start_ee_pose=[0, 0, 0, 0, 0, 0, 1], scene_geom_model=None, scene_offset=0.0, scene_scaling=1.0):
        self.gripper_mesh_file = osp.join(path_util.get_rpdiff_descriptions(), 'franka_panda/meshes/collision/hand.obj')
        self.gripper_tmesh = trimesh.load(self.gripper_mesh_file)
        self.gripper_tmesh.apply_translation([0, 0, -0.105])  # panda-hand specific offset
        
        self.n_gripper_query_points = 1000
        self.gripper_hand_query_points_origin = self.gripper_tmesh.sample(self.n_gripper_query_points)
        # self.gripper_tmesh.apply_transform(util.matrix_from_pose(util.list2pose_stamped(start_ee_pose)))
        
        self.scene_pcd = scene_pcd
        self.scene_pcd_2d = self.scene_pcd[:, :-1]

        self.object_pcd_origin = object_pcd
        self.object_pcd = copy.deepcopy(self.object_pcd_origin)
        self.object_pcd_2d = self.object_pcd[:, :-1]
        self.obj_ch_origin = ConvexHull(self.object_pcd_2d)
        self.obj_ch_origin = ConvexHull(self.obj_ch_origin.points[self.obj_ch_origin.vertices])
        self.gripper_ch_origin = ConvexHull(self.gripper_tmesh.vertices)
        self.gripper_ch_origin = ConvexHull(self.gripper_ch_origin.points[self.gripper_ch_origin.vertices])

        self.scene_geom_model = scene_geom_model
        if self.scene_geom_model is not None:
            self.scene_model_device = list(self.scene_geom_model.parameters())[0].device
        else:
            self.scene_model_device = None
        self.object_query_points = None
        self.gripper_hand_query_points = None

        self.scene_offset = scene_offset
        self.scene_scaling = 1.0

        self.init_obj()
        self.init_gripper()

    def init_obj(self, transform=np.eye(3)):
        pts = self.obj_ch_origin.points[self.obj_ch_origin.vertices]
        pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        pts_verts = np.matmul(transform, pts_h.T).T[:, :-1]
        self.obj_ch = ConvexHull(pts_verts)
        # self.obj_ch = ConvexHull(self.obj_ch_origin.points[self.obj_ch.vertices])
        self.obj_inward_normals = -1.0 * self.obj_ch.equations[:, :-1]
        self.obj_offsets = self.obj_ch.equations[:, -1]

        transform_full = np.eye(4)
        transform_full[:-2, :-2] = transform[:-1, :-1]
        transform_full[:-2, -1] = transform[:-1, -1]
        self.object_pcd = util.transform_pcd(copy.deepcopy(self.object_pcd_origin), transform_full)
        self.object_query_points_np = util.crop_pcd(self.object_pcd, z=[1.2*np.min(self.object_pcd[:, 2]), 1.0])
        self.n_object_query_points = self.object_query_points_np.shape[0]

        if self.scene_geom_model is not None:
            # points to be used for checking occupancy with respect to scene occupancy field
            # self.object_query_points = copy.deepcopy(self.object_pcd)
            self.object_query_points = torch.from_numpy(self.object_query_points_np).float().to(self.scene_model_device)
            self.object_query_points = self.object_query_points.reshape(1, -1, 3)

            # self.object_scene_offset = torch.ones_like(self.object_query_points).float().to(self.scene_model_device) * self.scene_offset
            self.object_scene_offset = torch.from_numpy(self.scene_offset).reshape(1, 1, 3).repeat((1, self.object_query_points.size(1), 1))
            self.object_scene_offset = self.object_scene_offset.float().to(self.scene_model_device)
            self.object_scene_scaling = self.scene_scaling

    def init_gripper(self, transform=np.eye(4)):
        pts = util.transform_pcd(self.gripper_ch_origin.points[self.gripper_ch_origin.vertices], transform)
        self.gripper_ch = ConvexHull(pts)
        # self.gripper_ch = ConvexHull(self.gripper_ch_origin.points[self.gripper_ch.vertices])
        self.gripper_inward_normals = -1.0 * self.gripper_ch.equations[:, :-1]
        self.gripper_offsets = self.gripper_ch.equations[:, -1]

        self.posed_gripper_tmesh = self.gripper_tmesh.copy().apply_transform(transform)
        self.gripper_hand_query_points_np = util.transform_pcd(self.gripper_hand_query_points_origin, transform)

        # points to be used for checking occupancy with respect to scene occupancy field
        if self.scene_geom_model is not None:
            # points to be used for checking occupancy with respect to scene occupancy field
            self.gripper_hand_query_points = torch.from_numpy(self.gripper_hand_query_points_np).float().to(self.scene_model_device)
            self.gripper_hand_query_points = self.gripper_hand_query_points.reshape(1, -1, 3)

            # self.gripper_scene_offset = torch.ones_like(self.gripper_hand_query_points).float().to(self.scene_model_device) * self.scene_offset
            self.gripper_scene_offset = torch.from_numpy(self.scene_offset).reshape(1, 1, 3).repeat((1, self.gripper_hand_query_points.size(1), 1))
            self.gripper_scene_offset = self.gripper_scene_offset.float().to(self.scene_model_device)
            self.gripper_scene_scaling = self.scene_scaling

    def obj_scene_collision(self, viz=False, normalize=False):
        return_dict = {}
        return_dict['object_in_collision'] = None
        return_dict['visible_scene_pts_in_collision'] = None
        return_dict['visible_object_pts_in_collision'] = None

        if self.scene_geom_model is None:
            print('Not using NN-based scene collision checking')
            obj_in_coll_comp_2d = np.matmul(self.obj_inward_normals, self.scene_pcd_2d.T) - np.tile(self.obj_offsets[:, None], (1, self.scene_pcd_2d.shape[0]))
            obj_in_coll_2d = np.all(obj_in_coll_comp_2d > 0, axis=0)

            if viz:
                trimesh_util.trimesh_show([self.scene_pcd[np.where(obj_in_coll_2d)], self.scene_pcd[np.where(np.logical_not(obj_in_coll_2d))], self.object_pcd])

            return_dict['object_in_collision'] = obj_in_coll_2d.any()
            return_dict['visible_scene_pts_in_collision'] = obj_in_coll_2d
        else:
            print('Using NN-based scene collision checking')
            if normalize:
                object_qp = (self.object_query_points - self.object_scene_offset) * self.object_scene_scaling
            else:
                object_qp = self.object_query_points
            occ_com = self.scene_geom_model.forward_latent(
                z=None, 
                coords=object_qp, 
                use_cached_latent=True, 
                return_occ=True)[1]
            # return occ_com.probs, occ_com.probs > thresh
            occ_probs = occ_com.probs.detach().cpu().numpy().squeeze()
            occ_idxs = np.where(occ_probs > 0.5)[0]
            # return_dict['object_in_collision'] = (occ_probs > 0.5).any()
            return_dict['object_in_collision'] = occ_idxs.shape[0] > (0.05 * self.n_object_query_points)
            # return_dict['visible_object_pts_in_collision'] = self.object_pcd[np.where(occ_probs > 0.5)[0]] 
            return_dict['visible_object_pts_in_collision'] = occ_idxs
        
        return return_dict
        
    def gripper_scene_collision(self, viz=False, normalize=False):
        return_dict = {}
        return_dict['gripper_in_collision'] = None
        return_dict['visible_scene_pts_in_collision'] = None
        return_dict['gripper_pts_in_collision'] = None

        if self.scene_geom_model is None:
            print('Not using NN-based scene collision checking')
            gripper_in_coll_comp = np.matmul(self.gripper_inward_normals, self.scene_pcd.T) - np.tile(self.gripper_offsets[:,  None], (1, self.scene_pcd.shape[0]))
            gripper_in_coll = np.all(gripper_in_coll_comp > 0, axis=0)

            if viz:
                scene = trimesh_util.trimesh_show([self.scene_pcd[np.where(gripper_in_coll)], self.scene_pcd[np.where(np.logical_not(gripper_in_coll))]], show=False)
                scene.add_geometry([self.posed_gripper_tmesh])
                scene.show()
            return_dict['gripper_in_collision'] = gripper_in_coll.any()
            return_dict['visible_scene_pts_in_collision'] = gripper_in_coll
        else:
            print('Using NN-based scene collision checking')
            if normalize:
                gripper_qp = (self.gripper_hand_query_points - self.gripper_scene_offset) * self.gripper_scene_scaling
            else:
                gripper_qp = self.gripper_hand_query_points
            occ_com = self.scene_geom_model.forward_latent(
                z=None, 
                coords=gripper_qp, 
                use_cached_latent=True, 
                return_occ=True)[1]
            # return occ_com.probs, occ_com.probs > thresh
            occ_probs = occ_com.probs.detach().cpu().numpy().squeeze()
            occ_idxs = np.where(occ_probs > 0.5)[0]
            # return_dict['gripper_in_collision'] = (occ_probs > 0.5).any()
            return_dict['gripper_in_collision'] = occ_idxs.shape[0] > (0.05 * self.n_gripper_query_points)
            # return_dict['gripper_pts_in_collision'] = self.gripper_hand_query_pts_np[np.where(occ_probs > 0.5)[0]] 
            return_dict['gripper_pts_in_collision'] = occ_idxs

        return return_dict
        

class PointCloudGraspedFused:
    def __init__(self):
        self.point_clouds = []  # list of PointCloudGraspedObject
        self.transformed_point_clouds = []  # list of PointCloudGraspedObject

        self.fused_point_cloud = None
        self.fused_point_cloud_list = []

    def add_new_pcd(self, pcd):
        self.point_clouds.append(pcd)

    def update_fused_pcd(self):
        # get the most recent point cloud
        latest_pcd = self.point_clouds[-1]
        self.transformed_point_clouds.append({
            'full_transformation': np.eye(4), 
            'point_cloud_raw': latest_pcd.point_cloud, 
            'point_cloud_transformed': latest_pcd.point_cloud,
            'transformation_sequence': [np.eye(4)] 
        })

        self.fused_point_cloud = latest_pcd.point_cloud
        self.fused_point_cloud_list.append(latest_pcd.point_cloud)

        for i, pcd in enumerate(self.transformed_point_clouds[:-1]):
            # pcd['full_transformation'] = np.matmul(latest_pcd.transformation, pcd['full_transformation'])
            # pcd['point_cloud_transformed'] = util.transform_pcd(pcd['point_cloud_raw'], pcd['full_transformation'])
            # pcd['transformation_sequence'].append(latest_pcd.transformation)
            init_transformation = np.matmul(latest_pcd.transformation, pcd['full_transformation'])
            refined_transformation = full_registration_np(
                source_np=pcd['point_cloud_raw'],
                target_np=self.fused_point_cloud,
                init_trans=init_transformation,
                viz=True
            )
            pcd['full_transformation'] = refined_transformation
            pcd['point_cloud_transformed'] = util.transform_pcd(pcd['point_cloud_raw'], refined_transformation)
            pcd['transformation_sequence'].append(latest_pcd.transformation)

            self.fused_point_cloud = np.concatenate([self.fused_point_cloud, pcd['point_cloud_transformed']], axis=0)
            self.fused_point_cloud_list.append(pcd['point_cloud_transformed'])

    def visualize_fused_pcd(self, separate=False):
        if separate:
            trimesh_util.trimesh_show(self.fused_point_cloud_list)
        else:
            trimesh_util.trimesh_show([self.fused_point_cloud])

    def get_fused_pcd(self):
        return self.fused_point_cloud


class PointCloudGraspedObject:
    """
    Attributes:
        parent (int): Index used to refer to previous state visited in planning
        point_cloud (np.ndarray): N X 3 pointcloud observed at this step (assumed to be after segmentation)
        point_cloud_scene (np.ndarray): N' X 3 pointcloud observed at this step (full scene point cloud)
        rgb_image (np.ndarray): HxWx3 array for RGB image observed at this pose
        depth_image (np.ndarray): HxW array for depth image observed at this pose
        object_seg_mask (np.ndarray): HxW array for True/False values indicating object mask 
        ee_pose (list or np.ndarray): [x, y, z, qx, qy, qz, qw] pose of the end effector for this point cloud
        transformation (np.ndarray): 4 X 4 homogenous transformation matrix representing what
            relative transformation was applied most recently to obtain the point cloud at this pose
        transformation_so_far (np.ndarray): 4 X 4 homogenous transformation matrix that tracks the composed
            sequence of transformations that have been used to lead up to this point cloud
    """
    def __init__(self, point_cloud, ee_pose, transformation, transformation_so_far=None, 
                 parent=-1, point_cloud_scene=None):
        self.parent = parent
        self.point_cloud = point_cloud
        self.point_cloud_scene = point_cloud_scene

        self.ee_pose = ee_pose
        self.transformation = transformation
        
        if transformation_so_far is None:
            self.transformation_so_far = transformation
        else:
            self.transformation_so_far = np.matmul(transformation, transformation_so_far)

        self.rgb_image = None
        self.depth_image = None
        self.object_seg_mask = None

        self.cam_ext_mat = None
        self.cam_int_mat = None

    def set_image_data(self, rgb, depth, object_seg_mask=None):
        """Setter function to directly set the transformed point cloud

        Args:
            rgb_image (np.ndarray): HxWx3 array for RGB image observed at this pose
            depth_image (np.ndarray): HxW array for depth image observed at this pose
            object_seg_mask (np.ndarray): HxW array for True/False values indicating object mask 
        """
        self.rgb_image = rgb
        self.depth_image = depth
        self.object_seg_mask = object_seg_mask


class PointCloudNode(object):
    """Class for representing object configurations based on point clouds,
    nodes in a search tree where edges represent rigid transformations between
    point cloud configurations, and bi-manual end effector poses that make contact
    with the object to execute the transformation edges.

    Attributes:
        parent (int): Index used to refer to previous state visited in planning
        pointcloud (np.ndarray): N X 3 pointcloud AFTER being transformed (downsampled # of points)
        pointcloud_full (np.ndarray): N' X 3 pointcloud AFTER being transformed (full # of points)
        pointcloud_mask (np.ndarray): N X 1 array of binary labels indicating which points are part
            of subgoal mask (i.e. for grasping)
        transformation (np.ndarray): 4 X 4 homogenous transformation matrix that is applied to initial
            pointcloud and transforms it into self.pointcloud.
        transformation_to_go (np.ndarray): 4 X 4 homogenous transformation matrix representing what
            transformation must be executed to satisfy the global desired transformation task specification
        transformation_so_far (np.ndarray): 4 X 4 homogenous transformation matrix that tracks the composed
            sequence of transformations that have been used to lead up to this point cloud
        palms (np.ndarray): TODO
        palms_corrected (np.ndarray): TODO
        palms_raw (np.ndarray): TODO
        planes (list): List of dictionaries, which have the following key value pairs:
            - 'planes' : np.ndarray of [x, y, z] points in the plane
            - 'normals' : np.ndarray of [x, y, z] normal vectors, for each point
            - 'mean_normal' : np.ndarray of [x, y, z] normal vector, which is the average of all the normals
            - 'antipodal_inds' : int, indicating the index in the list of the most likely oposite plane
        skill (str): Name of skill type that was used to transform the point cloud by the transformation
            specified in self.transformation
        surface (str): Name of the placement surface that the object touches after being transformed by
            self.transformation
    """
    def __init__(self):
        self.parent = None
        self.pointcloud = None
        self.pointcloud_full = None
        self.pointcloud_mask = None
        self.transformation = None
        self.transformation_to_go = np.eye(4)
        self.transformation_so_far = None
        self.palms = None
        self.palms_corrected = None
        self.palms_raw = None
        self.skill = None
        self.surface = None

        self.planes = None
        self.antipodal_thresh = 0.01

    def set_pointcloud(self, pcd, pcd_full=None, pcd_mask=None):
        """Setter function to directly set the transformed point cloud

        Args:
            pcd (np.ndarray): N X 3 transformed point cloud
            pcd_full (np.ndarray, optional): N' X 3 transformed pointcloud (full # of points). 
                Defaults to None.
            pcd_mask (np.ndarray, optional): N X 1 array of binary subgoal mask labels. Defaults to None.
        """
        self.pointcloud = pcd
        self.pointcloud_full = pcd_full
        self.pointcloud_mask = pcd_mask

    def set_trans_to_go(self, trans):
        """Setter function to set the transformation to go from this node on,
        if planning to solve some task specified by a desired global transformation

        Args:
            trans ([type]): [description]
        """
        self.transformation_to_go = trans

    def set_planes(self, planes):
        """Setter function to set the planes that correspond to a particular point cloud

        Args:
            planes (list): List of np.ndarrays of size N X 3 that contain set of points, where
                each array element in a list is a separately segmented plane 
        """
        # put planes and mean plane normals
        self.planes = []
        for i in range(len(planes)):
            plane_dict = {}
            plane_dict['plane'] = planes[i]['points']
            self.planes.append(plane_dict)

        # use mean plane normals to estimate pairs of antipodal planes, specified by index in the list
        for i in range(len(self.planes)):
            plane = self.planes[i]['plane']
            # estimate plane normals
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(plane)
            pcd.estimate_normals()
            normals = np.asarray(pcd.normals)
            self.planes[i]['normals'] = normals
            mean_normal = np.mean(normals, axis=0)
            self.planes[i]['mean_normal'] = mean_normal/np.linalg.norm(mean_normal)

        for i in range(len(self.planes)):
            # get average normal vector of this plane
            mean_normal = self.planes[i]['mean_normal']
            self.planes[i]['antipodal_inds'] = None
            for j in range(len(self.planes)):
                # don't check with self
                if i == j:
                    continue
                
                # get average normal vector with other planes
                mean_normal_check = self.planes[j]['mean_normal']

                # use dot product threshold to guess if it's an antipodal face, indicate based on index in list
                dot_prod = np.dot(mean_normal, mean_normal_check)
                if np.abs(1 - dot_prod) < self.antipodal_thresh:
                    self.planes[i]['antipodal_inds'] = j
                    break

    def init_state(self, state, transformation, *args, **kwargs):
        """Initialize the pointcloud and transformation attributes based on some 
        initial node (state) and sampled transformation.

        Args:
            state (PointCloudNode): Node containing the initial pointcloud, to be transformed
            transformation (np.ndarray): 4 X 4 homogenous transformation matrix, representing
                the transformation to be applied to the start pointcloud
        """
        # compute the pointcloud based on the previous pointcloud and specified trans
        pcd_homog = np.ones((state.pointcloud.shape[0], 4))
        pcd_homog[:, :-1] = state.pointcloud
        self.pointcloud = np.matmul(transformation, pcd_homog.T).T[:, :-1]

        # do the same for the full pointcloud, if it's there
        if state.pointcloud_full is not None:
            pcd_full_homog = np.ones((state.pointcloud_full.shape[0], 4))
            pcd_full_homog[:, :-1] = state.pointcloud_full
            self.pointcloud_full = np.matmul(transformation, pcd_full_homog.T).T[:, :-1]

        # node's one step transformation
        self.transformation = transformation

        # transformation to go based on previous transformation to go
        self.transformation_to_go = np.matmul(state.transformation_to_go, np.linalg.inv(transformation))

        # transformation so far, accounting for if this is the first step,
        # and parent node has no transformation so far
        if state.transformation is not None:
            self.transformation_so_far = np.matmul(transformation, state.transformation_so_far)
        else:
            self.transformation_so_far = transformation
        
        # save skill if it is specified
        if 'skill' in kwargs.keys():
            self.skill = kwargs['skill']

        # transform planes and plane normals, if we have them available
        if state.planes is not None:
            self.planes = []
            for i, plane_dict in enumerate(state.planes):
                new_plane_dict = {}

                new_plane = copy.deepcopy(plane_dict['plane'])
                new_normals = copy.deepcopy(plane_dict['normals'])

                # transform planes
                new_plane_homog = np.ones((new_plane.shape[0], 4))
                new_plane_homog[:, :-1] = new_plane
                new_plane = np.matmul(transformation, new_plane_homog.T).T[:, :-1]
                
                # transform normals
                new_normals = util.transform_vectors(new_normals, util.pose_from_matrix(transformation))
                
                # save new info
                new_plane_dict['plane'] = new_plane
                new_plane_dict['normals'] = new_normals
                new_plane_dict['mean_normal'] = np.mean(new_normals, axis=0)
                new_plane_dict['antipodal_inds'] = plane_dict['antipodal_inds']

                self.planes.append(new_plane_dict)

    def init_palms(self, palms, correction=False, prev_pointcloud=None, dual=True):
        """Initilize the palm attributes based on some sampled palm poses. Also implements
        heuristic corrections of the palm samples based on the transformed pointcloud, to make it
        more likely the palms will correctly contact the surface of the object.

        Args:
            palms (np.ndarray): 1 x 14
            correction (bool, optional): Whether or not to apply heuristic correction. Correction is
                based on searching along the vector normal to the intially specified palm plane for the
                closest point in the pointcloud, so that the position component of the palm pose can be
                refined while maintaining the same orientation. Must have access to the initial pointcloud
                used to sample the palm poses. Defaults to False.
            prev_pointcloud (np.ndarray, optional): N X 3, intial pointcloud used to sample the palms. Used
                here for palm pose refinement. Defaults to None.
            dual (bool, optional): True if this node corresponds to a two-arm skill, else False.
                Defaults to True.
        """
        if correction and prev_pointcloud is not None:
            palms_raw = palms
            palms_positions = {}
            palms_positions['right'] = palms_raw[:3]
            palms_positions['left'] = palms_raw[7:7+3]
            pcd_pts = prev_pointcloud
            if dual:
                # palms_positions_corr = correct_grasp_pos(palms_positions,
                #                                         pcd_pts)
                # palm_right_corr = np.hstack([
                #     palms_positions_corr['right'],
                #     palms_raw[3:7]])
                # palm_left_corr = np.hstack([
                #     palms_positions_corr['left'],
                #     palms_raw[7+3:]
                # ])
                # self.palms_corrected = np.hstack([palm_right_corr, palm_left_corr])

                r_positions_corr = correct_palm_pos_single(palms_raw[:7], pcd_pts)[:3]
                l_positions_corr = correct_palm_pos_single(palms_raw[7:], pcd_pts)[:3]
                palm_right_corr = np.hstack([r_positions_corr, palms_raw[3:7]])
                palm_left_corr = np.hstack([l_positions_corr, palms_raw[7+3:]])
                self.palms_corrected = np.hstack([palm_right_corr, palm_left_corr])
            else:
                r_positions_corr = correct_palm_pos_single(palms_raw[:7], pcd_pts)[:3]
                # l_positions_corr = correct_palm_pos_single(palms_raw[:7], pcd_pts)[:3]
                # palms_positions_corr = {}
                # palms_positions_corr['right'] = r_positions_corr
                # palms_positions_corr['left'] = l_positions_corr
                # self.palms_corrected = np.hstack([palms_positions_corr['right'], palms_raw[3:7]])
                self.palms_corrected = np.hstack([r_positions_corr, palms_raw[3:7]])

            self.palms_raw = palms_raw
            self.palms = self.palms_corrected
        else:
            self.palms = palms
            self.palms_raw = palms

        # check if hands got flipped like a dummy by checking y coordinate in world frame
        if self.palms.shape[0] > 7:
            if self.palms[1] > self.palms[1+7]:
                tmp_l = copy.deepcopy(self.palms[7:])
                self.palms[7:] = copy.deepcopy(self.palms[:7])
                self.palms[:7] = tmp_l
    
    def init_surface(self, surface_name):
        """
        Setter function for internal surface attribute

        Args:
            surface_name (str): Name of surface that object is on at this node
        """
        self.surface = surface_name

    def init_skill(self, skill_name):
        """
        Setter function for internal surface attribute

        Args:
            skill_name (str): Name of skill that is used to reach this node
        """
        self.skill = skill_name  
    
    def _make_full_skill_name(self, keep_separate=False):
        """
        Internal function to create the full name of the skill that was used to transition
        to this node (combining the skill type and the surface name)
        """
        if self.skill is not None and self.surface is not None and \
                isinstance(self.skill, str) and isinstance(self.surface, str) and not keep_separate:
            self._full_skill = self.skill + '_' + self.surface
        else:
            self._full_skill = self.skill 

    def get_skill_name(self):
        return self.skill

    def get_surface_name(self):
        self._make_full_skill_name()
        return self.surface

    def get_full_skill_name(self, keep_separate=False):
        self._make_full_skill_name(keep_separate=keep_separate)
        return self._full_skill


class PointCloudPlaneSegmentation(object):
    def __init__(self):
        self.ksearch = 50
        self.coeff_optimize = True
        self.normal_distance_weight = 0.05
        self.max_iterations = 100
        self.distance_threshold = 0.005

    def segment_pointcloud(self, pointcloud):
        p = pcl.PointCloud(np.asarray(pointcloud, dtype=np.float32))

        seg = p.make_segmenter_normals(ksearch=self.ksearch)
        seg.set_optimize_coefficients(self.coeff_optimize)
        seg.set_model_type(pcl.SACMODEL_PLANE)
        seg.set_normal_distance_weight(self.normal_distance_weight)
        seg.set_method_type(pcl.SAC_RANSAC)
        seg.set_max_iterations(self.max_iterations)
        seg.set_distance_threshold(self.distance_threshold)
        inliers, _ = seg.segment()

        # plane_pts = p.to_array()[inliers]
        # return plane_pts
        return inliers  

    def get_pointcloud_planes(self, pointcloud, visualize=False):
        planes = []

        original_pointcloud = copy.deepcopy(pointcloud)
        com_z = np.mean(original_pointcloud, axis=0)[2]
        for _ in range(5):
            inliers = self.segment_pointcloud(pointcloud)
            masked_pts = pointcloud[inliers]
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(masked_pts)
            pcd.estimate_normals()

            masked_pts_z_mean = np.mean(masked_pts, axis=0)[2]
            above_com = masked_pts_z_mean > com_z

            parallel_z = 0
            if masked_pts.shape[0] == 0:
                print('No points found in segmentation, skipping')
                continue
            for _ in range(100):
                pt_ind = np.random.randint(masked_pts.shape[0])
                pt_sampled = masked_pts[pt_ind, :]
                normal_sampled = np.asarray(pcd.normals)[pt_ind, :]

                dot_x = np.abs(np.dot(normal_sampled, [1, 0, 0]))
                dot_y = np.abs(np.dot(normal_sampled, [0, 1, 0]))
                if np.abs(dot_x) < 0.01 and np.abs(dot_y) < 0.01:
                    parallel_z += 1

            # print(parallel_z)
            if not (above_com and parallel_z > 30):
                # don't consider planes that are above the CoM
                plane_dict = {}
                plane_dict['mask'] = inliers
                plane_dict['points'] = masked_pts
                planes.append(plane_dict)

            if visualize:
                from rpo_planning.utils.visualize import PalmVis
                from rpo_planning.config.multistep_eval_cfg import get_multistep_cfg_defaults
                cfg = get_multistep_cfg_defaults()
                # prep visualization tools
                palm_mesh_file = osp.join(os.environ['CODE_BASE'],
                                            cfg.PALM_MESH_FILE)
                table_mesh_file = osp.join(os.environ['CODE_BASE'],
                                            cfg.TABLE_MESH_FILE)
                viz_palms = PalmVis(palm_mesh_file, table_mesh_file, cfg)
                viz_data = {}
                viz_data['contact_world_frame_right'] = util.pose_stamped2np(util.unit_pose())
                viz_data['contact_world_frame_left'] = util.pose_stamped2np(util.unit_pose())
                viz_data['transformation'] = util.pose_stamped2np(util.unit_pose())
                viz_data['object_pointcloud'] = masked_pts
                viz_data['start'] = masked_pts

                scene_pcd = viz_palms.vis_palms_pcd(viz_data, world=True, corr=False, full_path=True, show_mask=False, goal_number=1)
                scene_pcd.show()

            pointcloud = np.delete(pointcloud, inliers, axis=0)
        return planes


class PointCloudCollisionChecker(object):
    """Class for collision checking between different segmented pointclouds,
    based on convex hull inclusion. 

    Args:
        collision_pcds (list): Each element in list is N X 3 np.ndarray, representing
            the pointcloud of the objects in the scene that should be considered as
            collision obstacles.
    """    
    def __init__(self, collision_pcds):
        # all the pointclouds that we consider to be "collision objects"
        self.collision_pcds = collision_pcds

    def open3d_pcd_init(self, points):
        """Helper function to initialize an open3d pointcloud object from a numpy array

        Args:
            points (np.ndarray): N X 3 pointcloud array

        Returns:
            open3d.geometry.PointCloud: open3d Pointcloud
        """
        pcd3d = open3d.geometry.PointCloud()
        pcd3d.points = open3d.utility.Vector3dVector(points)
        return pcd3d

    def check_2d(self, obj_pcd):
        """Check if obj_pcd is inside the 2D projection of our collision geometry

        Args:
            obj_pcd (np.ndarray): N X 3 pointcloud array, to be checked for collisions
                with respect to other collision object pointclouds
        """
        valid = True
        if self.collision_pcds is None:
            return True
        com_2d = np.mean(obj_pcd, axis=0)[:-1]
        for pcd in self.collision_pcds:
            # hull_2d = ConvexHull(pcd[:, :-1])
            # hull_poly_2d = geometry.Polygon(hull_2d.points[hull_2d.vertices])
            com = np.mean(pcd, axis=0)
            pcd = pcd - com
            pcd = pcd * 1.1
            pcd = pcd + com

            res = self.in_poly_hull_multi(pcd, obj_pcd)
            valid = valid and (np.asarray(res) == False).all()
            if not valid:
                return False

        return valid

    def in_poly_hull_multi(self, poly, points):
        """Function to check if any points in a 3D pointcloud fall inside of
        the convex hull of a set of 3D points. A convex hull for the collision
        geometry is computed, and each point in the manipulated object is checked
        to see if it is inside the convex hull or not. This is done by adding the
        point to be checked to the set of points, recomputing the convex hull,
        and checking to see if the convex hull has changed (if it has not changed, then
        the point is inside the convex hull).

        Args:
            poly (np.ndarray): N X 3 pointcloud array of points in collision object
            points (np.ndarray): N X 3 pointcloud array of points in manipulated object

        Returns:
            list: Each element is a boolean, for each of the points that were checked.
                The value is True if the point is in collision, and False if not. Check
                if all values are False to see if the full object is collision-free
        """
        hull = ConvexHull(poly)
        res = []
        for p in points:
            new_hull = ConvexHull(np.concatenate((poly, [p])))
            res.append(np.array_equal(new_hull.vertices, hull.vertices))
        return res

    def in_poly_hull_single(self, poly, point):
        """Function to check if a 3D point falls inside of
        the convex hull of a set of 3D points. A convex hull for the collision
        geometry is computed, and the point is checked
        to see if it is inside the convex hull or not. This is done by adding the
        point to be checked to the set of points, recomputing the convex hull,
        and checking to see if the convex hull has changed (if it has not changed, then
        the point is inside the convex hull).

        Args:
            poly (np.ndarray): N X 3 pointcloud array of points in collision object
            point (np.ndarray): 1 X 3 point to be checked

        Returns:
            bool: True if point is in collision, False if collision-free
        """        
        hull = ConvexHull(poly)
        new_hull = ConvexHull(np.concatenate((poly, [point])))
        return np.array_equal(new_hull.vertices, hull.vertices)

    def show_collision_pcds(self, extra_pcds=None):
        """Visualize the pointcloud collision geometry that is being used.

        Args:
            extra_pcds (np.ndarray, optional): Other pointclouds to include in the
                visualization that are not used as collision objects. Defaults to None.
        """
        if self.collision_pcds is None:
            raise ValueError('no collision pointclouds found!')
        pcds = []
        for pcd in self.collision_pcds:
            # pcd3d = self.open3d_pcd_init(pcd)
            pcd3d = open3d.geometry.PointCloud()
            pcd3d.points = open3d.utility.Vector3dVector(pcd)
            pcds.append(pcd3d)
        if extra_pcds is not None:
            for pcd in extra_pcds:
                # pcd3d = self.open3d_pcd_init(pcd)
                pcd3d = open3d.geometry.PointCloud()
                pcd3d.points = open3d.utility.Vector3dVector(pcd)
                pcds.append(pcd3d)
        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.6, origin=[0, 0, 0])
        pcds.append(mesh_frame)
        open3d.visualization.draw_geometries(pcds)

    def show_collision_check_sample(self, obj_pcd):
        """Visualize the collision geometry with a queried object that is being
        checked for collision

        Args:
            obj_pcd (np.ndarray): Manipulated object pointcloud, to be checked for collisions
        """
        print('Valid: ' + str(self.check_2d(obj_pcd)))
        self.show_collision_pcds(extra_pcds=[obj_pcd])

