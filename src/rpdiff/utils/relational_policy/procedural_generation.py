import os.path as osp
import numpy as np
import trimesh
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import KMeans
from scipy.cluster.vq import kmeans2
# from scipy.cluster.vq import vq, kmeans, whiten

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

from rpdiff.utils import util
from rpdiff.utils.mesh_util.inside_mesh import check_mesh_contains
from rpdiff.utils.mesh_util import three_util

from typing import List, Union, Tuple
from meshcat import Visualizer

class ProcGenRelations:
    def __init__(self, task_name: str, parent_class: str, child_class: str, upright_dict: dict, mc_vis: Visualizer):
        self.valid_task_names = [
            'mug_on_rack',
            'mug_on_rack_multi',
            'bowl_on_mug',
            'bottle_in_container',
            'book_in_bookshelf',
            'stack_can_in_cabinet'
        ]
        self.task_name = task_name
        assert self.task_name in self.valid_task_names, f'Task name: {self.task_name} invalid! Must be in {", ".join(self.valid_task_names)}'

        self.mc_vis = mc_vis
        self.parent_class = parent_class
        self.child_class = child_class
        self.upright_dict = upright_dict

        self.infer_methods = [
            self.infer_mug_on_rack,
            self.infer_mug_on_rack_multi,
            self.infer_bowl_on_mug,
            self.infer_bottle_in_container,
            self.infer_book_in_bookshelf,
            self.infer_can_in_cabinet
        ]

        # register all available functions for procedural demo generation
        self.infer_rel_method_dict = dict(zip(self.valid_task_names, self.infer_methods))

    def infer_relation_task(self, parent_pcd: np.ndarray, child_pcd: np.ndarray, 
                            parent_mesh: Union[trimesh.Trimesh, List[trimesh.Trimesh]], 
                            child_mesh: Union[trimesh.Trimesh, List[trimesh.Trimesh]], 
                            parent_pose: Union[np.ndarray, List[np.ndarray]], 
                            child_pose: Union[np.ndarray, List[np.ndarray]],
                            parent_scale: Union[float, np.ndarray, List[Union[float, np.ndarray]]], 
                            child_scale: Union[float, np.ndarray, List[Union[float, np.ndarray]]], 
                            viz: bool, *args, **kwargs) -> dict:
        """
        Uses the task name to call the respective method for generating a pose
        that's likely to satisfy the relation
        """
        inf_fn = self.infer_rel_method_dict[self.task_name]
        log_debug(f'Using method: {inf_fn} for procedural task generation')
        out = inf_fn(parent_pcd, child_pcd, parent_mesh, child_mesh, parent_pose, child_pose, parent_scale, child_scale, viz=viz, *args, **kwargs) 
        return out
         
    def infer_mug_on_rack(self, parent_pcd: np.ndarray, child_pcd: np.ndarray, 
                          parent_mesh: trimesh.Trimesh, child_mesh: trimesh.Trimesh, 
                          parent_pose: np.ndarray, child_pose: np.ndarray, 
                          parent_scale: Union[float, np.ndarray], child_scale: Union[float, np.ndarray], 
                          viz: bool, *args, **kwargs) -> dict:
        """
        Generate relative transformation of mug (child object) that is likely to hang on rack
        (parent object). Uses prior knowledge and heuristics about upright pose and where the 
        peg on the rack is likely to be
        """
        parent_pose_mat = util.matrix_from_pose(util.list2pose_stamped(parent_pose))
        child_pose_mat = util.matrix_from_pose(util.list2pose_stamped(child_pose))
        parent_tmesh_origin = trimesh.load(parent_mesh).apply_scale(parent_scale)
        child_tmesh_origin = trimesh.load(child_mesh).apply_scale(child_scale)

        parent_tmesh = parent_tmesh_origin.copy().apply_transform(parent_pose_mat)
        child_tmesh = child_tmesh_origin.copy().apply_transform(child_pose_mat)

        # handle on the mug is in the 
        handle_pos_noise = np.random.random() * 0.01
        approx_handle_z_pos = child_tmesh.scale * -0.3 + handle_pos_noise
        approx_handle_pose_obj_frame = util.list2pose_stamped([0, 0, approx_handle_z_pos, 0, 0, 0, 1])
        approx_obj_pose_handle_frame = util.list2pose_stamped([0, 0, -1.0*approx_handle_z_pos, 0, 0, 0, 1])
        approx_handle_pose = util.convert_reference_frame(
            pose_source=approx_handle_pose_obj_frame,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.list2pose_stamped(child_pose)
        )
        approx_handle_pose_mat = util.matrix_from_pose(approx_handle_pose)

        # don't know where the peg on the rack is by the pose, but at least know the upright. let's also use the oriented bounding box
        rack_obb = parent_tmesh.bounding_box_oriented.to_mesh()
        # peg_xy_vec = rack_obb.principal_inertia_vectors[1]  # should be the middle principal component 
        peg_xy_vec = -1.0*parent_pose_mat[:-1, 1]
        peg_xy_orth_vec = rack_obb.principal_inertia_vectors[2]
        peg_angle = np.random.random() * (55 - 35) + 35
        # peg_pitch = np.deg2rad(55)
        peg_pitch = np.deg2rad(peg_angle)

        # estimate the position about half way along the peg
        peg_side_val = np.random.random() * (1.8 - 1.4) + 1.4
        peg_side_ref_pt = parent_tmesh.centroid + (peg_xy_vec * parent_tmesh_origin.extents[1] / peg_side_val)
        rack_pts = parent_tmesh.sample(1000)
        rack_kdtree = KDTree(rack_pts)
        peg_knn_idx = rack_kdtree.query(peg_side_ref_pt, 100)[1]
        peg_knn_pts = rack_pts[peg_knn_idx]

        peg_cent_noise = np.random.random(3) * 0.01
        peg_centroid = trimesh.PointCloud(peg_knn_pts).centroid + peg_cent_noise
        peg_cent_sph = trimesh.creation.uv_sphere(0.03).apply_translation(peg_centroid)

        # build a coordinate frame with axes that match our mug coordinate system
        peg_x_vec = peg_xy_vec  # axis pointing through handle opening
        # peg_x_vec = -1.0*parent_pose_mat[:-1, 1]
        peg_y_vec = np.array([0, 0, 1])
        peg_z_vec = np.cross(peg_x_vec, peg_y_vec)
        # peg_x_vec = peg_xy_vec  # axis pointing through handle opening
        # peg_z_vec = -1.0 * peg_xy_orth_vec  # axis pointing from handle to centroid
        # peg_y_vec = -1.0 * np.cross(peg_x_vec, peg_z_vec)  # axis pointing along cylindrical body toward top
        peg_pose_mat = util.matrix_from_pose(util.pose_from_vectors(peg_x_vec, peg_y_vec, peg_z_vec, peg_centroid))

        # rotate and transform to peg
        handle_yaw_tf = np.eye(4)
        handle_yaw_rot_mat = R.from_euler('xyz', [0, 0, peg_pitch]).as_matrix()
        handle_yaw_tf[:-1, :-1] = handle_yaw_rot_mat

        # handle_peg_pose = np.matmul(peg_pose_mat, np.matmul(handle_yaw_tf, np.linalg.inv(peg_pose_mat)))
        handle_peg_pose = np.matmul(peg_pose_mat, handle_yaw_tf)

        final_mug_pose = util.convert_reference_frame(
            pose_source=approx_obj_pose_handle_frame,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.pose_from_matrix(handle_peg_pose)
        )
        final_mug_pose_mat = util.matrix_from_pose(final_mug_pose)

        child_tmesh_final = child_tmesh_origin.copy().apply_transform(final_mug_pose_mat)

        if viz:
            util.meshcat_trimesh_show(self.mc_vis, 'scene/rack_origin', parent_tmesh_origin)
            util.meshcat_frame_show(self.mc_vis, 'scene/parent_frame', util.matrix_from_pose(util.list2pose_stamped(parent_pose)))
            util.meshcat_frame_show(self.mc_vis, 'scene/child_frame', util.matrix_from_pose(util.list2pose_stamped(child_pose)))

            # util.meshcat_trimesh_show(self.mc_vis, 'scene/rack_obb', rack_obb, opacity=0.7)

            util.meshcat_pcd_show(self.mc_vis, peg_knn_pts, (255, 0, 0), 'scene/peg_pts')
            util.meshcat_trimesh_show(self.mc_vis, 'scene/peg_cent', peg_cent_sph, color=(0, 0, 255), opacity=0.8)

            util.meshcat_frame_show(self.mc_vis, 'scene/peg_base_frame', peg_pose_mat)
            util.meshcat_frame_show(self.mc_vis, 'scene/handle_peg_frame', handle_peg_pose)

            util.meshcat_frame_show(self.mc_vis, 'scene/final_mug_pose', final_mug_pose_mat)
            util.meshcat_trimesh_show(self.mc_vis, 'scene/final_mug', child_tmesh_final)

        rel_trans = np.matmul(final_mug_pose_mat, np.linalg.inv(child_pose_mat))
        
        ### post-process
        # check if the final objects are intersecting
        # child_mesh_final = child_tmesh_final
        parent_mesh_final = parent_tmesh
        n_coll_pts = 5000
        parent_sample_points = parent_mesh_final.sample(n_coll_pts)
        child_sample_points_original = child_tmesh_origin.sample(n_coll_pts)
        # parent_sample_points_original = parent_mesh_final.sample(n_coll_pts)

        # child_sample_points = copy.deepcopy(child_sample_points_original)
        # parent_sample_points = copy.deepcopy(parent_sample_points_original)

        coll_check_iter = 0
        # delta_t = np.zeros(3)
        delta_xt = 0.0
        delta_yt = 0.0
        delta_zt = 0.0
        while True:
            coll_check_iter += 1
            if coll_check_iter > 50:
                log_info('Sampled for collision-free too many times, returning identity to force failure')
                final_mug_pose_mat = child_pose_mat
                break
            
            pah, pal = 55, 35
            # pah, pal = 60, 30
            peg_angle = np.random.random() * (pah - pal) + pal
            peg_pitch = np.deg2rad(peg_angle)
            handle_yaw_tf = np.eye(4)
            handle_yaw_rot_mat = R.from_euler('xyz', [0, 0, peg_pitch]).as_matrix()
            handle_yaw_tf[:-1, :-1] = handle_yaw_rot_mat
            handle_peg_pose = np.matmul(peg_pose_mat, handle_yaw_tf)

            final_mug_pose = util.convert_reference_frame(
                pose_source=approx_obj_pose_handle_frame,
                pose_frame_target=util.unit_pose(),
                pose_frame_source=util.pose_from_matrix(handle_peg_pose)
            )
            final_mug_pose_mat = util.matrix_from_pose(final_mug_pose)
            final_mug_pose_mat[:-1, -1] += (delta_xt * handle_peg_pose[:-1, 0])
            final_mug_pose_mat[:-1, -1] += (delta_yt * handle_peg_pose[:-1, 1])
            final_mug_pose_mat[:-1, -1] += (delta_zt * handle_peg_pose[:-1, 2])

            child_tmesh_final = child_tmesh_origin.copy().apply_transform(final_mug_pose_mat)
            
            if viz:
                util.meshcat_trimesh_show(self.mc_vis, 'scene/final_mug', child_tmesh_final)

            child_mesh_final = child_tmesh_final
            # child_sample_points = child_mesh_final.sample(n_coll_pts)
            child_sample_points = util.transform_pcd(child_sample_points_original, final_mug_pose_mat)

            # child_sample_points = copy.deepcopy(child_sample_points_original)
            # parent_sample_points = copy.deepcopy(parent_sample_points_original)

            c_in_pts_idx = check_mesh_contains(parent_mesh_final, child_sample_points)
            p_in_pts_idx = check_mesh_contains(child_mesh_final, parent_sample_points)

            c_idx, p_idx = np.where(c_in_pts_idx)[0], np.where(p_in_pts_idx)[0]
            nc_pts, np_pts = c_idx.shape[0], p_idx.shape[0]
            c_pts, p_pts = child_sample_points[c_idx], parent_sample_points[p_idx]
            log_debug(f'Number of intersecting child points: {nc_pts} and parent points: {np_pts}')

            if viz:
                util.meshcat_pcd_show(self.mc_vis, c_pts, (0, 255, 0), 'scene/child_pts_in_coll')
                util.meshcat_pcd_show(self.mc_vis, p_pts, (0, 255, 255), 'scene/parent_pts_in_coll')
            
                if nc_pts > 0:
                    c_cent = np.mean(c_pts, axis=0)
                    c_sph = trimesh.creation.uv_sphere(0.005).apply_translation(c_cent)
                    util.meshcat_trimesh_show(self.mc_vis, 'scene/child_coll_cent', c_sph, color=(0, 255, 0))
                if np_pts > 0:
                    p_cent = np.mean(p_pts, axis=0)
                    p_sph = trimesh.creation.uv_sphere(0.005).apply_translation(p_cent)
                    util.meshcat_trimesh_show(self.mc_vis, 'scene/child_coll_cent', p_sph, color=(0, 255, 255))

            # print('here in main, after getting relative_trans and checking collision')
            # from IPython import embed; embed()
            
            if nc_pts > 1 or np_pts > 1:
                cd_max, pd_max = 0.0, 0.0
                if nc_pts > 1:
                    c_dist = c_pts[:, None, :] - c_pts[None, :, :]
                    cd_max = np.max(np.linalg.norm(c_dist, axis=-1))
                if np_pts > 1:
                    p_dist = p_pts[:, None, :] - p_pts[None, :, :]
                    pd_max = np.max(np.linalg.norm(p_dist, axis=-1))
                d_max = max(cd_max, pd_max)
                log_debug(f'Max Dist: {d_max}')
                if d_max > 0.01:
                    # delta_t = 0.035 * (np.random.random(3) - 0.5)
                    # delta_t = 0.04 * (np.random.random() - 0.5)
                    delta_xt = 0.05 * (np.random.random())
                    # delta_yt = 0.05 * (np.random.random() - 0.5)
                    delta_yt = -0.05 * (np.random.random())
                    delta_zt = 0.05 * (np.random.random() - 0.5)
            else:
                log_debug('No points in collision')
                break
        
        rel_trans = np.matmul(final_mug_pose_mat, np.linalg.inv(child_pose_mat))

        out_dict = {}
        out_dict['rel_trans'] = rel_trans

        if 'return_part_poses' in kwargs.keys():
            if kwargs['return_part_poses']:
                part_poses = dict(
                    parent_part_world=handle_peg_pose,
                    child_part_world=np.matmul(np.linalg.inv(rel_trans), handle_peg_pose),
                    child_part_parent=np.linalg.inv(rel_trans)
                )
                # return rel_trans, part_poses
                out_dict['part_poses'] = part_poses

        # return rel_trans
        return out_dict

    def infer_mug_on_rack_multi(self, parent_pcd: np.ndarray, child_pcd: np.ndarray, 
                                parent_mesh_list: List[trimesh.Trimesh], 
                                child_mesh_list: List[trimesh.Trimesh], 
                                parent_pose_list: List[np.ndarray], 
                                child_pose_list: List[np.ndarray],
                                parent_scale_list: List[Union[np.ndarray, float]], 
                                child_scale_list: List[Union[np.ndarray, float]], 
                                viz: bool, return_parent_idx: bool=True, *args, **kwargs) -> dict:

        try:
            # to start, suppose we only have a single child object
            child_idx = np.random.randint(len(child_mesh_list))  # 0
            child_pose = child_pose_list[child_idx]
            child_scale = child_scale_list[child_idx]
            child_mesh = child_mesh_list[child_idx]

            # sample an index from the multiple racks we have
            parent_idx = np.random.randint(len(parent_mesh_list))
            parent_pose = parent_pose_list[parent_idx]
            parent_scale = parent_scale_list[parent_idx]
            parent_mesh = parent_mesh_list[parent_idx]
        except IndexError as e:
            print(f'IndexError: {e}')
            print("here in infer mug_on_rack_multi")
            from IPython import embed; embed()

        parent_mesh_all = three_util.trimesh_combine(parent_mesh_list, [util.matrix_from_list(pose) for pose in parent_pose_list], parent_scale_list)
    
        # from IPython import embed; embed()
        parent_pose_mat = util.matrix_from_pose(util.list2pose_stamped(parent_pose))
        child_pose_mat = util.matrix_from_pose(util.list2pose_stamped(child_pose))
        parent_tmesh_origin = trimesh.load(parent_mesh).apply_scale(parent_scale)
        child_tmesh_origin = trimesh.load(child_mesh).apply_scale(child_scale)

        parent_tmesh = parent_tmesh_origin.copy().apply_transform(parent_pose_mat)
        child_tmesh = child_tmesh_origin.copy().apply_transform(child_pose_mat)

        # handle on the mug is in the 
        handle_pos_noise = (np.random.random() - 0.5) * 0.01
        approx_handle_z_pos = child_tmesh.scale * -0.3 + handle_pos_noise
        approx_handle_pose_obj_frame = util.list2pose_stamped([0, 0, approx_handle_z_pos, 0, 0, 0, 1])
        approx_obj_pose_handle_frame = util.list2pose_stamped([0, 0, -1.0*approx_handle_z_pos, 0, 0, 0, 1])
        approx_handle_pose = util.convert_reference_frame(
            pose_source=approx_handle_pose_obj_frame,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.list2pose_stamped(child_pose)
        )
        approx_handle_pose_mat = util.matrix_from_pose(approx_handle_pose)

        # let's find all the pegs on this rack
        rack_z_vec = parent_pose_mat[:-1, 2]
        rack_verts = parent_tmesh.sample(5000)
        rack_bottom_z = np.min(rack_verts[:, 2])
        rack_top_z = np.max(rack_verts[:, 2])

        # use the base points to re-estimate the position
        rack_base_xy_verts = rack_verts[np.where(rack_verts[:, 2] < rack_bottom_z*1.02)[0]]
        # rack_base_xy_verts = rack_verts[np.where(rack_bottom_z*1.03 < rack_verts[:, 2] < rack_bottom_z*1.04)[0]]
        rack_base_xy_verts[:, -1] = 0.0
        util.meshcat_pcd_show(self.mc_vis, rack_base_xy_verts, (255, 255, 0), 'scene/rack_base_xy_verts')
        base_xy_cent = np.mean(rack_base_xy_verts, axis=0)

        util.meshcat_frame_show(self.mc_vis, 'scene/parent_frame_pre_adjust', util.matrix_from_pose(util.list2pose_stamped(parent_pose)))
        parent_pose_mat[:2, -1] = base_xy_cent[:2]
        parent_pose = util.pose_stamped2list(util.pose_from_matrix(parent_pose_mat))
        util.meshcat_trimesh_show(self.mc_vis, 'scene/rack_world', parent_tmesh)
        util.meshcat_trimesh_show(self.mc_vis, 'scene/mug_world', child_tmesh)
        util.meshcat_frame_show(self.mc_vis, 'scene/parent_frame', util.matrix_from_pose(util.list2pose_stamped(parent_pose)))

        rack_xy_verts = rack_verts[np.where(rack_verts[:, 2] > rack_bottom_z*1.05)[0]]
        rack_xy_verts[:, -1] = 0.0
        # util.meshcat_pcd_show(self.mc_vis, rack_xy_verts, (0, 255, 0), 'scene/rack_xy_verts')

        # get the xy verts that are all away from the middle
        xy_cent = parent_pose_mat[:-1, -1]; xy_cent[-1] = 0.0
        xy_dist_from_cent = np.linalg.norm(rack_xy_verts - xy_cent, axis=1) 
        far_inds = np.where(xy_dist_from_cent > 0.05)[0]
        rack_xy_far_verts = rack_xy_verts[far_inds]
        # util.meshcat_pcd_show(self.mc_vis, rack_xy_far_verts, (0, 255, 255), 'scene/rack_xy_vert_far')

        try:
            # Find 2 clusters in the data (for now, assume we know that there are two...)
            # km = KMeans(n_clusters=2).fit(rack_xy_far_verts)
            # whitened = whiten(rack_xy_far_verts)
            # codebook, distortion = kmeans(whitened, 2)
            # peg_xy_tip_pos = codebook
            n_pegs = 2
            peg_tip_xy_pos, label = kmeans2(rack_xy_far_verts, n_pegs, minit='points')
            tip_sph1 = trimesh.creation.uv_sphere(0.003).apply_translation(peg_tip_xy_pos[0])
            tip_sph2 = trimesh.creation.uv_sphere(0.003).apply_translation(peg_tip_xy_pos[1])
            # util.meshcat_trimesh_show(self.mc_vis, 'scene/tip_sph1', tip_sph1, (255, 0, 0))
            # util.meshcat_trimesh_show(self.mc_vis, 'scene/tip_sph2', tip_sph2, (0, 0, 255))
        except Exception as e:
            print(f'Exception after failing to run kmeans: {e}')
            # print("here after failing to run kmeans")
            # from IPython import embed; embed()

            rel_trans = np.eye(4)

            part_poses = dict(
                parent_part_world=np.eye(4),
                child_part_world=np.eye(4),
                child_part_parent=np.eye(4)
            )

            out_dict = {}
            out_dict['rel_trans'] = rel_trans

            if 'return_part_poses' in kwargs.keys():
                if kwargs['return_part_poses']:
                    out_dict['part_poses'] = part_poses

            # return rel_trans
            return out_dict

        nz_search = 500
        peg_points = {}
        peg_xy_vecs = {}
        peg_3d_vecs = {}
        peg_z_angles = {}
        rack_kdtree = KDTree(rack_verts)
        for peg_idx in range(peg_tip_xy_pos.shape[0]):
            # get vector from center to tip
            tip_xy_pos = peg_tip_xy_pos[peg_idx]
            tip_vec = tip_xy_pos - xy_cent

            half_xy_pos = xy_cent + tip_vec*0.8

            z_linspace = np.linspace(rack_bottom_z, rack_top_z, nz_search)

            z_tip_pts_search = np.tile(tip_xy_pos, (nz_search, 1))
            z_tip_pts_search[:, 2] = z_linspace

            z_mid_pts_search = np.tile(half_xy_pos, (nz_search, 1))
            z_mid_pts_search[:, 2] = z_linspace

            # util.meshcat_pcd_show(self.mc_vis, z_mid_pts_search, (255, 0, 255), f'scene/z_mid_search_peg_{peg_idx}')
            # util.meshcat_pcd_show(self.mc_vis, z_tip_pts_search, (255, 0, 255), f'scene/z_tip_search_peg_{peg_idx}')

            z_mid_occ = check_mesh_contains(parent_tmesh, z_mid_pts_search)
            z_mid_in = z_mid_pts_search[np.where(z_mid_occ)[0]]

            z_tip_occ = check_mesh_contains(parent_tmesh, z_tip_pts_search)
            z_tip_in = z_tip_pts_search[np.where(z_tip_occ)[0]]

            # peg_mid_pos = tip_half_pos.copy(); peg_mid_pos[2] = np.mean(z_pts_in, axis=0)
            if z_mid_in.shape[0] > 0:
                peg_mid_pos = np.mean(z_mid_in, axis=0)
            else:
                # look up the closest point 
                dists = rack_kdtree.query(z_mid_pts_search)[0]
                min_idx = np.argmin(dists)
                peg_mid_pos = z_mid_pts_search[min_idx]

            # mid_sph = trimesh.creation.uv_sphere(0.003).apply_translation(peg_mid_pos)
            # util.meshcat_trimesh_show(self.mc_vis, f'scene/mid_sph_{peg_idx}', mid_sph, (0, 255, 0))

            # peg_tip_pos = tip_half_pos.copy(); peg_tip_pos[2] = np.mean(z_pts_in, axis=0)
            if z_tip_in.shape[0] > 0:
                peg_tip_pos = np.mean(z_tip_in, axis=0)
            else:
                # look up the closest point
                dists = rack_kdtree.query(z_tip_pts_search)[0]
                min_idx = np.argmin(dists)
                peg_tip_pos = z_tip_pts_search[min_idx]

            # tip_sph = trimesh.creation.uv_sphere(0.003).apply_translation(peg_tip_pos)
            # util.meshcat_trimesh_show(self.mc_vis, f'scene/tip_sph_{peg_idx}', tip_sph, (0, 255, 255))

            # save the information about this peg
            # points on the rack corresponding to this peg

            try:
                peg_knn_idx = rack_kdtree.query(peg_mid_pos, 100)[1]
                peg_knn_pts = rack_verts[peg_knn_idx]
                peg_points[peg_idx] = peg_knn_pts
            except Exception as e:
                print(e)
                print('here with error in mug_on_rack_multi proc gen (knn)')
                from IPython import embed; embed()

            # vector in the xy plane that it points
            peg_xy_vecs[peg_idx] = tip_vec / np.linalg.norm(tip_vec)

            # angle relative to the positive z axis
            peg_3d_vector = peg_tip_pos - peg_mid_pos
            peg_3d_vector = peg_3d_vector / np.linalg.norm(peg_3d_vector)
            peg_3d_vecs[peg_idx] = peg_3d_vector
            peg_3d_angle = util.angle_from_3d_vectors(peg_3d_vector, [0, 0, 1])
            peg_z_angles[peg_idx] = np.pi/2 - peg_3d_angle

    
        # randomly sample a peg
        peg_idx = np.random.randint(n_pegs)
        peg_xy_vec = peg_xy_vecs[peg_idx]
        peg_angle = peg_z_angles[peg_idx]
        peg_pitch = peg_angle
        peg_xy_orth_vec = np.cross(peg_xy_vec, [0, 0, 1])
        peg_knn_pts = peg_points[peg_idx]

        peg_cent_noise = np.random.random(3) * 0.01
        peg_centroid = trimesh.PointCloud(peg_knn_pts).centroid + peg_cent_noise
        peg_cent_sph = trimesh.creation.uv_sphere(0.03).apply_translation(peg_centroid)

        # build a coordinate frame with axes that match our mug coordinate system
        peg_x_vec = peg_xy_vec  # axis pointing through handle opening
        peg_y_vec = np.array([0, 0, 1])
        peg_z_vec = np.cross(peg_x_vec, peg_y_vec)
        peg_pose_mat = util.matrix_from_pose(util.pose_from_vectors(peg_x_vec, peg_y_vec, peg_z_vec, peg_centroid))

        # rotate and transform to peg
        handle_yaw_tf = np.eye(4)
        handle_yaw_rot_mat = R.from_euler('xyz', [0, 0, peg_pitch]).as_matrix()
        handle_yaw_tf[:-1, :-1] = handle_yaw_rot_mat

        # handle_peg_pose = np.matmul(peg_pose_mat, np.matmul(handle_yaw_tf, np.linalg.inv(peg_pose_mat)))
        # handle_peg_pose = np.matmul(peg_pose_mat, handle_yaw_tf)  # BASE VERSION

        if util.exists_and_true(kwargs, 'vary_about_peg'):        
        # if 'vary_about_peg' in kwargs:
        #     if kwargs['vary_about_peg']:
            x_rot_val = np.random.random() * 2 * np.pi
            flip_y = np.random.random() > 0.5
            y_rot_val = np.pi if flip_y else 0.0
            # x_rot = R.from_euler('xyz', [np.pi/4, 0, 0]).as_matrix()
            # y_rot = R.from_euler('xyz', [0, np.pi, 0]).as_matrix()
            x_rot = R.from_euler('xyz', [x_rot_val, 0, 0]).as_matrix()
            y_rot = R.from_euler('xyz', [0, y_rot_val, 0]).as_matrix()
            x_rot_mat = np.eye(4); x_rot_mat[:-1, :-1] = x_rot
            y_rot_mat = np.eye(4); y_rot_mat[:-1, :-1] = y_rot

            # handle_peg_pose = np.matmul(peg_pose_mat, handle_yaw_tf @ y_rot_mat)
            # handle_peg_pose = np.matmul(peg_pose_mat, handle_yaw_tf @ x_rot_mat)
            handle_peg_pose = np.matmul(peg_pose_mat, handle_yaw_tf @ (y_rot_mat @ x_rot_mat))
        else:
            handle_peg_pose = np.matmul(peg_pose_mat, handle_yaw_tf)

        try:
            final_mug_pose = util.convert_reference_frame(
                pose_source=approx_obj_pose_handle_frame,
                pose_frame_target=util.unit_pose(),
                pose_frame_source=util.pose_from_matrix(handle_peg_pose)
            )
        except Exception as e:
            print(e)
            print("here in multi mug on rack, failed to build final_mug_pose")
            from IPython import embed; embed()

        final_mug_pose_mat = util.matrix_from_pose(final_mug_pose)

        child_tmesh_final = child_tmesh_origin.copy().apply_transform(final_mug_pose_mat)

        if viz:
            util.meshcat_trimesh_show(self.mc_vis, 'scene/rack_origin', parent_tmesh_origin)
            util.meshcat_frame_show(self.mc_vis, 'scene/parent_frame', util.matrix_from_pose(util.list2pose_stamped(parent_pose)))
            util.meshcat_frame_show(self.mc_vis, 'scene/child_frame', util.matrix_from_pose(util.list2pose_stamped(child_pose)))

            # util.meshcat_trimesh_show(self.mc_vis, 'scene/rack_obb', rack_obb, opacity=0.7)

            util.meshcat_pcd_show(self.mc_vis, peg_knn_pts, (255, 0, 0), 'scene/peg_pts')
            util.meshcat_trimesh_show(self.mc_vis, 'scene/peg_cent', peg_cent_sph, color=(0, 0, 255), opacity=0.8)

            util.meshcat_frame_show(self.mc_vis, 'scene/peg_base_frame', peg_pose_mat)
            util.meshcat_frame_show(self.mc_vis, 'scene/handle_peg_frame', handle_peg_pose)

            util.meshcat_frame_show(self.mc_vis, 'scene/final_mug_pose', final_mug_pose_mat)
            util.meshcat_trimesh_show(self.mc_vis, 'scene/final_mug', child_tmesh_final)

        rel_trans = np.matmul(final_mug_pose_mat, np.linalg.inv(child_pose_mat))
        
        ### post-process
        # check if the final objects are intersecting
        # child_mesh_final = child_tmesh_final
        parent_mesh_final = parent_tmesh
        n_coll_pts = 5000
        parent_sample_points = parent_mesh_final.sample(n_coll_pts)
        child_sample_points_original = child_tmesh_origin.sample(n_coll_pts)
        # parent_sample_points_original = parent_mesh_final.sample(n_coll_pts)

        # child_sample_points = copy.deepcopy(child_sample_points_original)
        # parent_sample_points = copy.deepcopy(parent_sample_points_original)

        coll_check_iter = 0
        # delta_t = np.zeros(3)
        delta_xt = 0.0
        delta_yt = 0.0
        delta_zt = 0.0

        # if coll_checking:
        skip_coll_checking = False
        if skip_coll_checking:
            pass
        else:
            peg_angle_original = peg_angle
            while True:
                coll_check_iter += 1
                if coll_check_iter > 50:
                    log_info('Sampled for collision-free too many times, returning identity to force failure')
                    final_mug_pose_mat = child_pose_mat
                    break
                
                # pah, pal = 55, 35
                # pah, pal = 60, 30
                # peg_angle = np.random.random() * (pah - pal) + pal
                # peg_pitch = np.deg2rad(peg_angle)
                peg_angle = peg_angle_original + np.deg2rad(30*(np.random.random() - 0.5))
                peg_pitch = peg_angle

                handle_yaw_tf = np.eye(4)
                handle_yaw_rot_mat = R.from_euler('xyz', [0, 0, peg_pitch]).as_matrix()
                handle_yaw_tf[:-1, :-1] = handle_yaw_rot_mat
                # handle_peg_pose = np.matmul(peg_pose_mat, handle_yaw_tf)
                if 'vary_about_peg' in kwargs:
                    if kwargs['vary_about_peg']:
                        x_rot_val = np.random.random() * 2 * np.pi
                        x_rot = R.from_euler('xyz', [x_rot_val, 0, 0]).as_matrix()
                        x_rot_mat = np.eye(4); x_rot_mat[:-1, :-1] = x_rot
                        handle_peg_pose = np.matmul(peg_pose_mat, handle_yaw_tf @ (y_rot_mat @ x_rot_mat))
                        if flip_y:
                            delta_xt = -1.0 * delta_xt if (delta_xt > 0.0) else delta_xt
                else:
                    handle_peg_pose = np.matmul(peg_pose_mat, handle_yaw_tf)

                final_mug_pose = util.convert_reference_frame(
                    pose_source=approx_obj_pose_handle_frame,
                    pose_frame_target=util.unit_pose(),
                    pose_frame_source=util.pose_from_matrix(handle_peg_pose)
                )
                final_mug_pose_mat = util.matrix_from_pose(final_mug_pose)
                final_mug_pose_mat[:-1, -1] += (delta_xt * handle_peg_pose[:-1, 0])
                final_mug_pose_mat[:-1, -1] += (delta_yt * handle_peg_pose[:-1, 1])
                final_mug_pose_mat[:-1, -1] += (delta_zt * handle_peg_pose[:-1, 2])

                child_tmesh_final = child_tmesh_origin.copy().apply_transform(final_mug_pose_mat)
                
                if viz:
                    util.meshcat_trimesh_show(self.mc_vis, 'scene/final_mug', child_tmesh_final)

                child_mesh_final = child_tmesh_final
                # child_sample_points = child_mesh_final.sample(n_coll_pts)
                child_sample_points = util.transform_pcd(child_sample_points_original, final_mug_pose_mat)

                # child_sample_points = copy.deepcopy(child_sample_points_original)
                # parent_sample_points = copy.deepcopy(parent_sample_points_original)

                c_in_pts_idx = check_mesh_contains(parent_mesh_all, child_sample_points)
                # c_in_pts_idx = check_mesh_contains(parent_mesh_final, child_sample_points)
                p_in_pts_idx = check_mesh_contains(child_mesh_final, parent_sample_points)

                c_idx, p_idx = np.where(c_in_pts_idx)[0], np.where(p_in_pts_idx)[0]
                nc_pts, np_pts = c_idx.shape[0], p_idx.shape[0]
                c_pts, p_pts = child_sample_points[c_idx], parent_sample_points[p_idx]
                log_debug(f'Number of intersecting child points: {nc_pts} and parent points: {np_pts}')

                if viz:
                    util.meshcat_pcd_show(self.mc_vis, c_pts, (0, 255, 0), 'scene/child_pts_in_coll')
                    util.meshcat_pcd_show(self.mc_vis, p_pts, (0, 255, 255), 'scene/parent_pts_in_coll')
                
                    if nc_pts > 0:
                        c_cent = np.mean(c_pts, axis=0)
                        c_sph = trimesh.creation.uv_sphere(0.005).apply_translation(c_cent)
                        util.meshcat_trimesh_show(self.mc_vis, 'scene/child_coll_cent', c_sph, color=(0, 255, 0))
                    if np_pts > 0:
                        p_cent = np.mean(p_pts, axis=0)
                        p_sph = trimesh.creation.uv_sphere(0.005).apply_translation(p_cent)
                        util.meshcat_trimesh_show(self.mc_vis, 'scene/child_coll_cent', p_sph, color=(0, 255, 255))

                # print('here in main, after getting relative_trans and checking collision')
                # from IPython import embed; embed()
                
                if nc_pts > 1 or np_pts > 1:
                    cd_max, pd_max = 0.0, 0.0
                    if nc_pts > 1:
                        c_dist = c_pts[:, None, :] - c_pts[None, :, :]
                        cd_max = np.max(np.linalg.norm(c_dist, axis=-1))
                    if np_pts > 1:
                        p_dist = p_pts[:, None, :] - p_pts[None, :, :]
                        pd_max = np.max(np.linalg.norm(p_dist, axis=-1))
                    d_max = max(cd_max, pd_max)
                    log_debug(f'Max Dist: {d_max}')
                    if d_max > 0.01:
                        delta_xt = 0.025 * (np.random.random())  # x is along the peg
                        delta_yt = -0.025 * (np.random.random())
                        delta_zt = 0.025 * (np.random.random() - 0.5)
                        if False:
                            # delta_t = 0.035 * (np.random.random(3) - 0.5)
                            # delta_t = 0.04 * (np.random.random() - 0.5)
                            delta_xt = 0.05 * (np.random.random())
                            # delta_yt = 0.05 * (np.random.random() - 0.5)
                            delta_yt = -0.05 * (np.random.random())
                            delta_zt = 0.05 * (np.random.random() - 0.5)
                else:
                    log_debug('No points in collision')
                    break
        
        rel_trans = np.matmul(final_mug_pose_mat, np.linalg.inv(child_pose_mat))

        out_dict = {}
        out_dict['rel_trans'] = rel_trans
        out_dict['parent_idx'] = parent_idx

        part_poses = dict(
            parent_part_world=handle_peg_pose,
            child_part_world=np.matmul(np.linalg.inv(rel_trans), handle_peg_pose),
            child_part_parent=np.linalg.inv(rel_trans)
        )

        if 'return_part_poses' in kwargs.keys():
            if kwargs['return_part_poses']:
                # return rel_trans, parent_idx, part_poses
                out_dict['part_poses'] = part_poses

        # return rel_trans, parent_idx
        return out_dict

    def infer_bowl_on_mug(self, parent_pcd: np.ndarray, child_pcd: np.ndarray, 
                          parent_mesh: trimesh.Trimesh, child_mesh: trimesh.Trimesh, 
                          parent_pose: np.ndarray, child_pose: np.ndarray, 
                          parent_scale: Union[float, np.ndarray], child_scale: Union[float, np.ndarray], 
                          viz: bool, *args, **kwargs) -> dict:
        """
        Generate relative transformation of bowl (child object) that is likely to be stacked on
        mug (parent object). Uses prior knowledge of the upright poses and finding the top/bottom
        of each object
        """
        parent_pose_mat = util.matrix_from_pose(util.list2pose_stamped(parent_pose))
        child_pose_mat = util.matrix_from_pose(util.list2pose_stamped(child_pose))
        parent_tmesh_origin = trimesh.load(parent_mesh).apply_scale(parent_scale)
        child_tmesh_origin = trimesh.load(child_mesh).apply_scale(child_scale)

        parent_tmesh = parent_tmesh_origin.copy().apply_transform(parent_pose_mat)
        child_tmesh = child_tmesh_origin.copy().apply_transform(child_pose_mat)

        # find the bottom of the bowl and the top of the mug
        upright_bowl_quat = self.upright_dict['bowl']
        upright_mug_quat = self.upright_dict['mug']

        # bottom/top means lowest/highest y-coordinate in the body frame
        top_mug_idx = np.argmax(parent_tmesh_origin.vertices[:, 1])
        bottom_bowl_idx = np.argmin(child_tmesh_origin.vertices[:, 1])

        delta_z_top = 0.0075
        top_mug_pt_body = parent_tmesh_origin.vertices[top_mug_idx]; top_mug_pt_body[1] += delta_z_top
        bottom_bowl_pt_body = child_tmesh_origin.vertices[bottom_bowl_idx]

        mug_top_body_frame = np.eye(4); mug_top_body_frame[1, -1] = top_mug_pt_body[1] 
        bowl_bottom_body_frame = np.eye(4); bowl_bottom_body_frame[1, -1] = bottom_bowl_pt_body[1] 

        # transform to the world frame
        mug_top_frame = util.matrix_from_pose(util.convert_reference_frame(
            pose_source=util.pose_from_matrix(mug_top_body_frame),
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.pose_from_matrix(parent_pose_mat)
        ))
        bowl_bottom_frame = util.matrix_from_pose(util.convert_reference_frame(
            pose_source=util.pose_from_matrix(bowl_bottom_body_frame),
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.pose_from_matrix(child_pose_mat)
        ))

        # transformation to align the y-axes, and make the bottom bowl point align with the top mug point
        rel_trans = np.matmul(mug_top_frame, np.linalg.inv(bowl_bottom_frame))
        final_bowl_mesh = child_tmesh.copy().apply_transform(rel_trans)

        if viz:
            util.meshcat_trimesh_show(self.mc_vis, 'scene/mug_origin', parent_tmesh_origin)
            util.meshcat_trimesh_show(self.mc_vis, 'scene/bowl_origin', child_tmesh_origin)
            util.meshcat_frame_show(self.mc_vis, 'scene/parent_frame', util.matrix_from_pose(util.list2pose_stamped(parent_pose)))
            util.meshcat_frame_show(self.mc_vis, 'scene/child_frame', util.matrix_from_pose(util.list2pose_stamped(child_pose)))

            # mug_sph = trimesh.creation.uv_sphere(0.005).apply_translation(mug_top_pt)
            # bowl_sph = trimesh.creation.uv_sphere(0.005).apply_translation(bowl_bottom_pt)
            # util.meshcat_trimesh_show(self.mc_vis, 'scene/mug_top', mug_sph, color=(255, 0, 0), opacity=0.8)
            # util.meshcat_trimesh_show(self.mc_vis, 'scene/bowl_bottom', bowl_sph, color=(0, 0, 255), opacity=0.8)
            # util.meshcat_trimesh_show(self.mc_vis, 'scene/rack_obb', rack_obb, opacity=0.7)

            util.meshcat_frame_show(self.mc_vis, 'scene/mug_top_frame', mug_top_frame)
            util.meshcat_frame_show(self.mc_vis, 'scene/bowl_bottom_frame', bowl_bottom_frame)

            util.meshcat_trimesh_show(self.mc_vis, 'scene/final_bowl', final_bowl_mesh)

            # from IPython import embed; embed()

        out_dict = {}
        out_dict['rel_trans'] = rel_trans

        if 'return_part_poses' in kwargs.keys():
            if kwargs['return_part_poses']:
                part_poses = dict(
                    parent_part_world=mug_top_frame,
                    child_part_world=bowl_bottom_frame,
                    child_part_parent=np.linalg.inv(rel_trans)
                )
                # return rel_trans, part_poses
                out_dict['part_poses'] = part_poses

        # return rel_trans
        return out_dict

    def infer_bottle_in_container(self, parent_pcd: np.ndarray, child_pcd: np.ndarray, 
                                  parent_mesh: trimesh.Trimesh, child_mesh: trimesh.Trimesh, 
                                  parent_pose: np.ndarray, child_pose: np.ndarray, 
                                  parent_scale: Union[float, np.ndarray], child_scale: Union[float, np.ndarray], 
                                  viz: bool, *args, **kwargs) -> dict:
        """
        Generate relative transformation of bottle (child object) that is likely to be placed upright
        in a container (parent object), ideally near the corners of the container. 
        """
        parent_pose_mat = util.matrix_from_pose(util.list2pose_stamped(parent_pose))
        child_pose_mat = util.matrix_from_pose(util.list2pose_stamped(child_pose))
        parent_tmesh_origin = trimesh.load(parent_mesh).apply_scale(parent_scale)
        child_tmesh_origin = trimesh.load(child_mesh).apply_scale(child_scale)

        parent_tmesh = parent_tmesh_origin.copy().apply_transform(parent_pose_mat)
        child_tmesh = child_tmesh_origin.copy().apply_transform(child_pose_mat)

        # bottom/top means lowest/highest y-coordinate in the body frame
        bottom_container_idx = np.argmin(parent_tmesh_origin.vertices[:, 2])
        bottom_bottle_idx = np.argmin(child_tmesh_origin.vertices[:, 1])

        delta_z_top = 0.015
        bottom_container_pt_body = parent_tmesh_origin.vertices[bottom_container_idx]; bottom_container_pt_body[1] += delta_z_top
        bottom_bottle_pt_body = child_tmesh_origin.vertices[bottom_bottle_idx]

        # get an estimate of the corner of the container bottom
        # container_bottom_pts_idx = np.argsort(parent_tmesh_origin.vertices[:, 2])
        # n_bottom_pts = 100
        container_bottom_pts_idx = np.where(parent_tmesh_origin.vertices[:, 2] < 0.95 * bottom_container_pt_body[2])[0]
        container_bottom_pts = parent_tmesh_origin.vertices[container_bottom_pts_idx]
        container_bottom_bb = trimesh.PointCloud(container_bottom_pts).bounding_box.to_mesh()
        container_corner_pt = container_bottom_bb.vertices[np.random.randint(container_bottom_bb.vertices.shape[0])]

        # x,y offset from corner to get a location for the bottle
        bottle_xz_delta = 0.005
        bottle_max_xz = np.max(child_tmesh_origin.vertices[:, 0]) + bottle_xz_delta
        container_offset_x = bottle_max_xz; container_offset_y = bottle_max_xz
        if container_corner_pt[0] > 0:
            container_place_pt_x = container_corner_pt[0] - bottle_max_xz
        else:
            container_place_pt_x = container_corner_pt[0] + bottle_max_xz

        if container_corner_pt[1] > 0:
            container_place_pt_y = container_corner_pt[1] - bottle_max_xz
        else:
            container_place_pt_y = container_corner_pt[1] + bottle_max_xz

        container_place_pt = np.array([container_place_pt_x, container_place_pt_y, container_corner_pt[2]])

        # container_bottom_body_frame = np.eye(4); container_bottom_body_frame[2, -1] = bottom_container_pt_body[2] 
        container_bottom_body_frame = np.eye(4)
        container_bottom_body_frame[:2, -1] = container_place_pt[:2]
        container_bottom_body_frame[2, -1] = bottom_container_pt_body[2] 
        bottle_bottom_body_frame = np.eye(4); bottle_bottom_body_frame[1, -1] = bottom_bottle_pt_body[1] 

        upright_bottle_quat = self.upright_dict['bottle']
        upright_bottle_mat = util.matrix_from_list([0, 0, 0] + upright_bottle_quat)
        bottle_bottom_body_frame = np.matmul(bottle_bottom_body_frame, np.linalg.inv(upright_bottle_mat))
        # bottle_bottom_body_frame = np.matmul(bottle_bottom_body_frame, upright_bottle_mat)
        # bottle_bottom_body_frame = np.matmul(upright_bottle_mat, bottle_bottom_body_frame)

        # transform to the world frame
        container_bottom_corner_frame = util.matrix_from_pose(util.convert_reference_frame(
            pose_source=util.pose_from_matrix(container_bottom_body_frame),
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.pose_from_matrix(parent_pose_mat)
        ))
        bottle_bottom_frame = util.matrix_from_pose(util.convert_reference_frame(
            pose_source=util.pose_from_matrix(bottle_bottom_body_frame),
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.pose_from_matrix(child_pose_mat)
        ))

        # container_bottom_corner_frame = np.eye(4)
        # bottle_bottom_frame = np.eye(4)

        rel_trans = np.matmul(container_bottom_corner_frame, np.linalg.inv(bottle_bottom_frame))
        final_bottle_mesh = child_tmesh.copy().apply_transform(rel_trans)

        if viz:
            util.meshcat_trimesh_show(self.mc_vis, 'scene/container_origin', parent_tmesh_origin)
            util.meshcat_trimesh_show(self.mc_vis, 'scene/bottle_origin', child_tmesh_origin)
            util.meshcat_frame_show(self.mc_vis, 'scene/parent_frame', util.matrix_from_pose(util.list2pose_stamped(parent_pose)))
            util.meshcat_frame_show(self.mc_vis, 'scene/child_frame', util.matrix_from_pose(util.list2pose_stamped(child_pose)))

            util.meshcat_frame_show(self.mc_vis, 'scene/container_bottom_corner_frame', container_bottom_corner_frame)
            util.meshcat_frame_show(self.mc_vis, 'scene/bottle_bottom_frame', bottle_bottom_frame)

            corner_sph = trimesh.creation.uv_sphere(0.005).apply_translation(container_corner_pt)
            util.meshcat_trimesh_show(self.mc_vis, 'scene/corner_pt', corner_sph, color=(255, 0, 0), opacity=0.8)

            place_sph = trimesh.creation.uv_sphere(0.005).apply_translation(container_place_pt)
            util.meshcat_trimesh_show(self.mc_vis, 'scene/corner_place_pt', place_sph, color=(255, 0, 255), opacity=0.8)
            # util.meshcat_trimesh_show(self.mc_vis, 'scene/final_bottle', final_bottle_mesh)

            # from IPython import embed; embed()

        out_dict = {}
        out_dict['rel_trans'] = rel_trans

        if 'return_part_poses' in kwargs.keys():
            if kwargs['return_part_poses']:
                part_poses = dict(
                    parent_part_world=container_bottom_corner_frame,
                    child_part_world=bottle_bottom_frame,
                    child_part_parent=np.linalg.inv(rel_trans)
                )
                # return rel_trans, part_poses
                out_dict['part_poses'] = part_poses

        # return rel_trans
        # raise NotImplementedError
        return out_dict

    def infer_book_in_bookshelf(self, parent_pcd: np.ndarray, child_pcd: np.ndarray, 
                                parent_mesh_list: List[trimesh.Trimesh], 
                                child_mesh_list: List[trimesh.Trimesh], 
                                parent_pose_list: List[np.ndarray], 
                                child_pose_list: List[np.ndarray],
                                parent_scale_list: List[Union[np.ndarray, float]], 
                                child_scale_list: List[Union[np.ndarray, float]], 
                                viz: bool, return_parent_idx: bool=True, *args, **kwargs) -> dict:
        """
        Generate relative transformation of bottle (child object) that is likely to be placed upright
        in a container (parent object), ideally near the corners of the container. 
        """
        
        # Full generality, parent and child pose/scale/meshes will come as lists (we might have scenes composed of multiple objects)
        # Here, we internally assume there is just a single object for each one (legacy compatibility)

        # to start, suppose we only have a single child object
        child_idx = np.random.randint(len(child_mesh_list))  # 0
        child_pose = child_pose_list[child_idx]
        child_scale = child_scale_list[child_idx]
        child_mesh = child_mesh_list[child_idx]

        # sample an index from the multiple racks we have
        parent_idx = 0
        parent_pose = parent_pose_list[parent_idx]
        parent_scale = parent_scale_list[parent_idx]
        parent_mesh = parent_mesh_list[parent_idx]

        parent_mesh_all = three_util.trimesh_combine(parent_mesh_list, [util.matrix_from_list(pose) for pose in parent_pose_list], parent_scale_list)

        parent_pose_mat = util.matrix_from_pose(util.list2pose_stamped(parent_pose))
        child_pose_mat = util.matrix_from_pose(util.list2pose_stamped(child_pose))
        parent_tmesh_origin = trimesh.load(parent_mesh).apply_scale(parent_scale)
        child_tmesh_origin = trimesh.load(child_mesh).apply_scale(child_scale)

        parent_tmesh = parent_tmesh_origin.copy().apply_transform(parent_pose_mat)
        child_tmesh = child_tmesh_origin.copy().apply_transform(child_pose_mat)

        # get the list of available poses
        bookshelf_name = parent_mesh.split('/')[-1].replace('.obj', '').replace('_dec', '')
        saved_available_poses_fname = osp.join(parent_mesh.split(bookshelf_name)[0], 'open_slot_poses', bookshelf_name + '_open_slot_poses.txt')
        loaded_poses = np.loadtxt(saved_available_poses_fname)
        if loaded_poses.shape[0] == 0:
            rel_trans = np.eye(4)

            out_dict = {}
            out_dict['rel_trans'] = rel_trans

            if 'return_part_poses' in kwargs.keys():
                if kwargs['return_part_poses']:
                    part_poses = dict(
                        parent_part_world=np.eye(4),
                        child_part_world=np.eye(4),
                        child_part_parent=np.eye(4)
                    )
                    # return rel_trans, part_poses
                    out_dict['part_poses'] = part_poses

            # return rel_trans
            return out_dict

        util.meshcat_trimesh_show(self.mc_vis, 'scene/bookshelf', parent_tmesh_origin)
        util.meshcat_trimesh_show(self.mc_vis, 'scene/book', child_tmesh_origin)
        child_pose_nom = loaded_poses[np.random.randint(loaded_poses.shape[0])]
        child_pose_nom_mat = util.matrix_from_list(child_pose_nom)

        child_mesh_posed = child_tmesh_origin.copy().apply_transform(child_pose_nom_mat)
        util.meshcat_trimesh_show(self.mc_vis, 'scene/book_posed', child_mesh_posed, (255, 0, 255))

        n_coll_pts = 5000
        parent_mesh_final = parent_tmesh
        # util.meshcat_trimesh_show(self.mc_vis, 'scene/bookshelf_world', parent_mesh_final)

        child_sample_points_original = child_tmesh_origin.sample(n_coll_pts)
        i_iter = 0
        while True:
            i_iter += 1
            # sample a pose
            
            if i_iter > loaded_poses.shape[0]*4:
                child_pose_nom_world_mat = child_pose_mat
                break

            child_pose_nom = loaded_poses[np.random.randint(loaded_poses.shape[0])]
            child_pose_nom[0] += 0.01
            child_pose_nom_mat = util.matrix_from_list(child_pose_nom)
            child_pose_nom_world = util.convert_reference_frame(
                pose_source=util.list2pose_stamped(child_pose_nom),
                pose_frame_target=util.unit_pose(),
                pose_frame_source=util.pose_from_matrix(parent_pose_mat)
            )
            child_pose_nom_world_mat = util.matrix_from_pose(child_pose_nom_world)
            child_pose_nom_world_mat[2, -1] += 0.01

            j_iter = 0
            feasible_book = False
            while True:
                if j_iter > 5:
                    break

                j_iter += 1
                # sample small variations of this pose to see if it's collision free

                child_mesh_final = child_tmesh_origin.copy().apply_transform(child_pose_nom_world_mat)
                util.meshcat_trimesh_show(self.mc_vis, 'scene/final_child', child_mesh_final)
                # child_sample_points = child_mesh_final.sample(n_coll_pts)
                child_sample_points = util.transform_pcd(child_sample_points_original, child_pose_nom_world_mat)

                c_in_pts_idx = check_mesh_contains(parent_mesh_final, child_sample_points)

                c_idx = np.where(c_in_pts_idx)[0]
                nc_pts = c_idx.shape[0]
                c_pts = child_sample_points[c_idx]
                log_debug(f'Number of intersecting child points: {nc_pts}')

                c_cent = np.mean(c_pts, axis=0)
                if viz:
                    util.meshcat_pcd_show(self.mc_vis, c_pts, (0, 255, 0), 'scene/child_pts_in_coll')
                
                    if nc_pts > 0:
                        c_sph = trimesh.creation.uv_sphere(0.005).apply_translation(c_cent)
                        util.meshcat_trimesh_show(self.mc_vis, 'scene/child_coll_cent', c_sph, color=(0, 255, 0))
                        # print('pts in coll')
                        # from IPython import embed; embed()
                
                if nc_pts == 0:
                    feasible_book = True
                    log_debug('No points in collision')
                    break
                else:
                    # try making a little adustment, away from the center of where the collisions are
                    book_cent_xy = child_pose_nom_world_mat[:2, -1]
                    coll_cent_xy = c_cent[:2]
                    xy_vec = book_cent_xy - coll_cent_xy
                    xy_vec = xy_vec / np.linalg.norm(xy_vec)
                    child_pose_nom_world_mat[0, -1] += 0.005*xy_vec[0]
                    child_pose_nom_world_mat[1, -1] += 0.005*xy_vec[1]

                    # add a general bit of random xy noise
                    xy_rnd = (np.random.random(2) - 0.5) * 0.003
                    child_pose_nom_world_mat[0, -1] += xy_rnd[0]
                    child_pose_nom_world_mat[1, -1] += xy_rnd[1]
                    child_pose_nom_world_mat[2, -1] += np.abs(xy_rnd[1])

            if feasible_book:
                break

        rel_trans = np.matmul(child_pose_nom_world_mat, np.linalg.inv(child_pose_mat))

        out_dict = {}
        out_dict['rel_trans'] = rel_trans

        if 'return_part_poses' in kwargs.keys():
            if kwargs['return_part_poses']:
                part_poses = dict(
                    parent_part_world=np.eye(4),
                    child_part_world=np.eye(4),
                    child_part_parent=np.eye(4)
                )
                # return rel_trans, part_poses
                out_dict['part_poses'] = part_poses

        # return rel_trans
        return out_dict

    def infer_can_in_cabinet(self, parent_pcd: np.ndarray, child_pcd: np.ndarray, 
                             parent_mesh_list: List[trimesh.Trimesh], 
                             child_mesh_list: List[trimesh.Trimesh], 
                             parent_pose_list: List[np.ndarray], 
                             child_pose_list: List[np.ndarray],
                             parent_scale_list: List[Union[np.ndarray, float]], 
                             child_scale_list: List[Union[np.ndarray, float]], 
                             viz: bool, return_parent_idx: bool=True, *args, **kwargs) -> dict:
        """
        Generate relative transformation of can (child object) that is likely to be placed upright
        in a cabinet (parent object) in either an open base slot or on top of a stack of existing cans
        """

        # Full generality, parent and child pose/scale/meshes will come as lists (we might have scenes composed of multiple objects)
        # Here, we internally assume there is just a single object for each one (legacy compatibility)

        # to start, suppose we only have a single child object
        child_idx = np.random.randint(len(child_mesh_list))  # 0
        child_pose = child_pose_list[child_idx]
        child_scale = child_scale_list[child_idx]
        child_mesh = child_mesh_list[child_idx]

        # sample an index from the multiple racks we have
        parent_idx = 0
        parent_pose = parent_pose_list[parent_idx]
        parent_scale = parent_scale_list[parent_idx]
        parent_mesh = parent_mesh_list[parent_idx]

        parent_pose_mat = util.matrix_from_pose(util.list2pose_stamped(parent_pose))
        child_pose_mat = util.matrix_from_pose(util.list2pose_stamped(child_pose))
        parent_tmesh_origin = trimesh.load(parent_mesh).apply_scale(parent_scale)
        child_tmesh_origin = trimesh.load(child_mesh).apply_scale(child_scale)

        child_h = child_tmesh_origin.extents[-1]
        child_r = child_tmesh_origin.extents[0]/2

        parent_tmesh = parent_tmesh_origin.copy().apply_transform(parent_pose_mat)
        child_tmesh = child_tmesh_origin.copy().apply_transform(child_pose_mat)

        # get the list of available poses
        cabinet_name = parent_mesh.split('/')[-1].replace('.obj', '').replace('_dec', '')

        # get the list of available poses
        cabinet_name = parent_mesh.split('/')[-1].replace('.obj', '').replace('_dec', '')
        saved_available_poses_fname = osp.join(parent_mesh.split(cabinet_name)[0], 'open_slot_poses', cabinet_name + '_open_slot_poses.npz')
        loaded_poses = np.load(saved_available_poses_fname, allow_pickle=True)
        avail_pose_info_all = loaded_poses['avail_top_poses']

        # cl = trimesh.creation.cylinder(radius=avail_pose_dims['r'], height=avail_pose_dims['h'])
        # body_pose = avail_pose_top_pose; body_pose[2, -1] += avail_pose_dims['h']/2
        # cl.apply_transform(body_pose)

        if avail_pose_info_all.shape[0] == 0:
            rel_trans = np.eye(4)

            out_dict = {}
            out_dict['rel_trans'] = rel_trans

            if 'return_part_poses' in kwargs.keys():
                if kwargs['return_part_poses']:
                    part_poses = dict(
                        parent_part_world=np.eye(4),
                        child_part_world=np.eye(4),
                        child_part_parent=np.eye(4)
                    )
                    # return rel_trans, part_poses
                    out_dict['part_poses'] = part_poses

            # return rel_trans
            return out_dict

        for li in range(avail_pose_info_all.shape[0]):
            pi = avail_pose_info_all[li]
            top_pose = pi['pose']
            util.meshcat_frame_show(self.mc_vis, f'scene/avail_poses/{li}', top_pose)

        util.meshcat_trimesh_show(self.mc_vis, 'scene/cabinet', parent_tmesh_origin)
        util.meshcat_trimesh_show(self.mc_vis, 'scene/can', child_tmesh_origin)

        n_coll_pts = 5000
        parent_mesh_final = parent_tmesh
        # util.meshcat_trimesh_show(self.mc_vis, 'scene/cabinet_world', parent_mesh_final)

        child_sample_points_original = child_tmesh_origin.sample(n_coll_pts)
        i_iter = 0
        while True:
            p_idx = np.random.randint(0, avail_pose_info_all.shape[0])
            avail_pose_info = avail_pose_info_all[p_idx]
            avail_pose_top_pose = avail_pose_info['pose']
            avail_pose_dims = avail_pose_info['dims']
            child_pose_nom_mat = avail_pose_top_pose; child_pose_nom_mat[2, -1] += child_h/2
            # child_pose_nom_mat = avail_pose_top_pose; child_pose_nom_mat[2, -1] += avail_pose_dims['h']/2
            child_pose_nom_mat[2, -1] += 0.001

            i_iter += 1
            # sample a pose

            if i_iter > avail_pose_info_all.shape[0]*4:
                child_pose_nom_world_mat = child_pose_mat
                break

            if child_h > avail_pose_dims['h']:
                log_debug(f'[Stack can on cabinet proc gen] Child height {child_h} too large for available poses ({avail_pose_dims["h"]})')
                continue
            if child_r > avail_pose_dims['r']:
                log_debug(f'[Stack can on cabinet proc gen] Child radius {child_r} too large for available poses ({avail_pose_dims["r"]})')
                continue

            # child_pose_nom = loaded_poses[np.random.randint(loaded_poses.shape[0])]
            # child_pose_nom_mat = util.matrix_from_list(child_pose_nom)

            child_mesh_posed = child_tmesh_origin.copy().apply_transform(child_pose_nom_mat)
            util.meshcat_trimesh_show(self.mc_vis, 'scene/can_posed', child_mesh_posed, (255, 0, 255))

            # child_pose_nom = loaded_poses[np.random.randint(loaded_poses.shape[0])]
            # child_pose_nom[0] += 0.01
            # child_pose_nom_mat = util.matrix_from_list(child_pose_nom)
            child_pose_nom_world = util.convert_reference_frame(
                pose_source=util.pose_from_matrix(child_pose_nom_mat),
                pose_frame_target=util.unit_pose(),
                pose_frame_source=util.pose_from_matrix(parent_pose_mat)
            )
            child_pose_nom_world_mat = util.matrix_from_pose(child_pose_nom_world)
            child_pose_nom_world_mat[2, -1] += 0.01

            j_iter = 0
            feasible_can = False
            while True:
                if j_iter > 5:
                    break

                j_iter += 1
                # sample small variations of this pose to see if it's collision free

                child_mesh_final = child_tmesh_origin.copy().apply_transform(child_pose_nom_world_mat)
                util.meshcat_trimesh_show(self.mc_vis, 'scene/final_child', child_mesh_final)
                # child_sample_points = child_mesh_final.sample(n_coll_pts)
                child_sample_points = util.transform_pcd(child_sample_points_original, child_pose_nom_world_mat)

                c_in_pts_idx = check_mesh_contains(parent_mesh_final, child_sample_points)

                c_idx = np.where(c_in_pts_idx)[0]
                nc_pts = c_idx.shape[0]
                c_pts = child_sample_points[c_idx]
                log_debug(f'Number of intersecting child points: {nc_pts}')

                c_cent = np.mean(c_pts, axis=0)
                if viz:
                    util.meshcat_pcd_show(self.mc_vis, c_pts, (0, 255, 0), 'scene/child_pts_in_coll')
                
                    if nc_pts > 0:
                        c_sph = trimesh.creation.uv_sphere(0.005).apply_translation(c_cent)
                        util.meshcat_trimesh_show(self.mc_vis, 'scene/child_coll_cent', c_sph, color=(0, 255, 0))
                        util.meshcat_trimesh_show(self.mc_vis, 'scene/parenet_mesh_final', parent_mesh_final, color=(255, 0, 0))
                        # print('pts in coll')
                        # from IPython import embed; embed()
                
                if nc_pts == 0:
                    feasible_can = True
                    log_debug('No points in collision')
                    break
                else:
                    # try making a little adustment, away from the center of where the collisions are
                    can_cent_xy = child_pose_nom_world_mat[:2, -1]
                    coll_cent_xy = c_cent[:2]
                    xy_vec = can_cent_xy - coll_cent_xy
                    xy_vec = xy_vec / np.linalg.norm(xy_vec)
                    child_pose_nom_world_mat[0, -1] += 0.005*xy_vec[0]
                    child_pose_nom_world_mat[1, -1] += 0.005*xy_vec[1]

                    # add a general bit of random xy noise
                    xy_rnd = (np.random.random(2) - 0.5) * 0.003
                    child_pose_nom_world_mat[0, -1] += xy_rnd[0]
                    child_pose_nom_world_mat[1, -1] += xy_rnd[1]
                    child_pose_nom_world_mat[2, -1] += np.abs(xy_rnd[1])

            if feasible_can:
                break

        rel_trans = np.matmul(child_pose_nom_world_mat, np.linalg.inv(child_pose_mat))

        out_dict = {}
        out_dict['rel_trans'] = rel_trans

        if 'return_part_poses' in kwargs.keys():
            if kwargs['return_part_poses']:
                part_poses = dict(
                    parent_part_world=np.eye(4),
                    child_part_world=np.eye(4),
                    child_part_parent=np.eye(4)
                )
                # return rel_trans, part_poses
                out_dict['part_poses'] = part_poses

        # return rel_trans
        return out_dict

