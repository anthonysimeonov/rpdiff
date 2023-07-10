import os, os.path as osp
import random
import numpy as np
import time
from imageio import imwrite
from scipy.spatial.transform import Rotation as R

import meshcat
import meshcat.geometry as mcg
import trimesh
import matplotlib.pyplot as plt

from airobot.sensor.camera.rgbdcam import RGBDCamera

from rpdiff.utils import util, trimesh_util, geometry_np
from rpdiff.utils.util import np2img
from rpdiff.utils.seg_aug_util import SegmentationAugmentation

# import visdom
# vis_vis = visdom.Visdom()
# mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
default_img_size = (480, 640)
seg_aug_helper = SegmentationAugmentation(default_img_size, circle_radius_hl=[0.1, 0.025], rectangle_side_hl=[0.2, 0.04])


def rhlb(bounds):
    val = np.random.random() * (max(bounds) - min(bounds)) + min(bounds)
    return val


def npz2dict(npz):
    out_dict = {}
    for k in npz.files:
        out_dict[k] = npz[k]
    return out_dict


def camera_cfgs(height=720, width=1280, fov=60):
    """
    Returns a set of camera config parameters

    Returns:
    YACS CfgNode: Cam config params
    """
    _C = CN()
    _C.ZNEAR = 0.01
    _C.ZFAR = 10
    _C.WIDTH = width
    _C.HEIGHT = height
    _C.FOV = 60
    _ROOT_C = CN()
    _ROOT_C.CAM = CN()
    _ROOT_C.CAM.SIM = _C
    _ROOT_C.CAM.REAL = _C
    return _ROOT_C.clone()


def build_default_int_mat(height=480, width=640, fov_deg=60):
    sensor_half_width = width/2
    sensor_half_height = height/2

    vert_fov = fov_deg * np.pi / 180

    vert_f = sensor_half_height / np.tan(vert_fov / 2)
    hor_f = sensor_half_width / (np.tan(vert_fov / 2) * 320 / 240)

    intrinsics = np.array(
        [[hor_f, 0., sensor_half_width, 0.],
        [0., vert_f, sensor_half_height, 0.],
        [0., 0., 1., 0.]]
    )
    return intrinsics


def sample_cam_pose_mat(focus_pt=None):
    # use euler angles in top hemisphere to generate random camera poses
    rhlb = lambda bounds: np.random.random() * (max(bounds) - min(bounds)) + min(bounds)

    roll = rhlb((-np.pi, np.pi))
    pitch = rhlb((-np.pi, np.pi))
    distance = rhlb((1.2, 0.45))
        
    if focus_pt is None:
        focus_pt = np.array([0.35, 0.0, 1.15])

    cam_pose_mat = np.eye(4)
    cam_pose_mat[:-1, :-1] = R.from_euler('xyz', [roll, -1.0*pitch, 0]).as_matrix()

    dist_vec_norm = cam_pose_mat[:-1, 2] / np.linalg.norm(cam_pose_mat[:-1, 2])
    cam_pose_mat[:-1, -1] = focus_pt - (dist_vec_norm * distance)

    return cam_pose_mat


def simulate_camera(cam, cam_int_mat, cam_pose_mat):
    cam.set_cam_ext(cam_ext=cam_pose_mat)
    cam.cam_int_mat = cam_int_mat[:3, :3]
    cam._init_pers_mat()

    return cam


def simulate_random_occlusions(pcd, cam_poses, cam_int_mat, img_size=(480, 640), select_prob=1.0, min_pts=1500):
    n_cams = len(cam_poses)
    select_cam = np.random.random((n_cams,)) > (1.0 - select_prob)
    if not select_cam.any():
        idx = np.random.randint(n_cams)
        select_cam[idx] = True

    pcds_aug = []
    depths = []
    for cam_idx in range(n_cams):
        if not select_cam[cam_idx]:
            continue
        cam_pose_mat = cam_poses[cam_idx]
        pcd_cam = util.transform_pcd(pcd, np.linalg.inv(cam_pose_mat))

        # util.meshcat_pcd_show(mc_vis, pcd, [255, 0, 0], f'scene/pcd_world_{cam_idx}')
        # util.meshcat_pcd_show(mc_vis, pcd_cam, [255, 128, 128], f'scene/pcd_cam_{cam_idx}')
        # util.meshcat_frame_show(mc_vis, f'scene/cam_pose_frame_{cam_idx}', cam_pose_mat)

        depth = geometry_np.project(pcd_cam[:, 0], pcd_cam[:, 1], pcd_cam[:, 2], cam_int_mat) 
        # depth_int = np.hstack([np.round(depth[:, :2]), depth[:, 2].reshape(-1, 1)])
        depth_int = np.hstack([np.floor(depth[:, :2]), depth[:, 2].reshape(-1, 1)])

        # print('depth int', depth_int)
        # from IPython import embed; embed()
        final_depth = np.zeros((img_size[0], img_size[1]))
        
        yy, xx = np.meshgrid(np.arange(img_size[0]), np.arange(img_size[1]), indexing='ij')
        pixel_coords = np.vstack([yy.ravel(), xx.ravel()]).T
        
        # get combined (1-D) pixel indices
        # uv_vec = depth_int[:, :-1].astype(int)
        vu_vec = depth_int[:, :-1].astype(int)
        vu_vec_merged = (vu_vec[:, 1]*max(default_img_size) + vu_vec[:, 0]).astype(int)
        depth_int_merged = np.hstack([vu_vec_merged.reshape(-1, 1), depth_int[:, -1].reshape(-1, 1)])

        # get indices corresponding to sorted depth (smallest depth first)
        sorted_depth_int_merged = depth_int_merged[np.argsort(depth_int_merged[:, 1])]

        # get the unique 1-D pixel indices of the DEPTH-SORTED array (first index now corresponds to smallest depth)
        # unique_inds = np.unique(sorted_depth_int_merged[:, 0].astype(int), return_index=True)[1]
        unique_inds = sorted(np.unique(sorted_depth_int_merged[:, 0].astype(int), return_index=True)[1])

        # get the corresponding sorting of the (u, v) index array, and then use unique inds to index into this
        sorted_depth_int = depth_int[np.argsort(depth_int_merged[:, 1])]
        unique_sorted_depth_int = sorted_depth_int[unique_inds]
        unique_sorted_depth_int_merged = sorted_depth_int_merged[unique_inds]
        unique_sorted_depth_int_merged[:, 0] = np.clip(unique_sorted_depth_int_merged[:, 0], 0, default_img_size[0]*default_img_size[1]-1)

        final_depth_flat = final_depth.reshape(-1)
        final_depth_flat[unique_sorted_depth_int_merged[:, 0].astype(int)] = unique_sorted_depth_int_merged[:, 1]

        final_depth = final_depth_flat.reshape(img_size[0], img_size[1])

        # vis_vis.heatmap(viz_depth_imgs[0])
        depths.append(final_depth)
        # fig = plt.figure()
        # plt.imshow(final_depth)
        # fig.savefig('depth2.png')
        # from IPython import embed; embed()
        
        # add holes
        obj_mask_inds = np.where(final_depth.reshape(-1))[0]
        obj_binary_mask = np.zeros(img_size, dtype=np.uint8).flatten()
        obj_visible_mask = np.zeros(img_size, dtype=np.uint8).flatten()
        obj_binary_mask[obj_mask_inds] = 1
        obj_binary_mask = obj_binary_mask.reshape(img_size)
        obj_visible_mask[obj_mask_inds] = 255
        obj_visible_mask = obj_visible_mask.reshape(img_size)

        # get the augmented segmentations and apply to depth image
        circle_inside = np.random.random() < 0.9
        # circle_inside = np.random.random() > 0.9
        # aug_mask = seg_aug_helper.sample_halfspace(obj_binary_mask)
        # aug_mask = seg_aug_helper.sample_circle(obj_binary_mask, inside=circle_inside)
        aug_mask = seg_aug_helper.sample_rectangle(obj_binary_mask)
        # aug_num = np.random.random()
        # circle_inside = np.random.random() > 0.9
        # if aug_num < 0.3:
        #     aug_mask = seg_aug_helper.sample_halfspace(obj_binary_mask)
        # elif 0.3 < aug_num < 0.6:
        #     aug_mask = seg_aug_helper.sample_circle(obj_binary_mask, inside=circle_inside)
        # else:
        #     aug_mask = seg_aug_helper.sample_halfspace(obj_binary_mask)
        #     aug_mask = aug_mask & seg_aug_helper.sample_circle(obj_binary_mask, inside=circle_inside)

        full_aug_mask = aug_mask
        # full_aug_mask = aug_mask & obj_binary_mask
        aug_seg_inds = np.where(full_aug_mask.reshape(-1))[0]

        aug_binary_mask = np.zeros(img_size, dtype=np.uint8).flatten()
        aug_visible_mask = np.zeros(img_size, dtype=np.uint8).flatten()
        aug_binary_mask[aug_seg_inds] = 1
        aug_binary_mask = aug_binary_mask.reshape(img_size)
        aug_visible_mask[aug_seg_inds] = 255
        aug_visible_mask = aug_visible_mask.reshape(img_size)

        final_depth = final_depth * full_aug_mask

        # # get the ground truth depth for comparision 
        # depth_gt = np.zeros((img_size[0], img_size[1])).flatten()
        # depth_gt[grasp_data['seg'][cam_idx][0]] = grasp_data['depth'][cam_idx]
        # depth_gt = depth_gt.reshape(img_size)

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(final_depth)
        # ax[1].imshow(depth_gt)
        # fig.savefig('depth2.png')
        # imwrite('seg_obj_fname_vis.png', obj_visible_mask)
        # imwrite('seg_aug_fname_vis.png', aug_visible_mask)
        # print('created depth from point cloud')
        # from IPython import embed; embed() 

        # go back to 3D from this augmented depth map
        pcd_aug_cam = geometry_np.lift(pixel_coords[:, 1], pixel_coords[:, 0], final_depth.flatten(), cam_int_mat)
        pcd_aug_cam = pcd_aug_cam[np.where(np.linalg.norm(pcd_aug_cam, axis=-1) > 0.0)[0]]
        pcd_aug_world = util.transform_pcd(pcd_aug_cam, cam_pose_mat)

        # util.meshcat_pcd_show(mc_vis, pcd_aug_world, [0, 255, 0], f'scene/pcd_aug_world_{cam_idx}')
        # from IPython import embed; embed()

        pcds_aug.append(pcd_aug_world)
    
    pcd_aug = np.concatenate(pcds_aug, axis=0)

    # # from IPython import embed; embed()
    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(depths[0])
    # ax[0, 1].imshow(depths[1])
    # ax[1, 0].imshow(depths[2])
    # ax[1, 1].imshow(depths[3])
    # fig.savefig('depth_all.png')
    # # from IPython import embed; embed()

    # if pcd_aug.shape[0] < 1500:
    if pcd_aug.shape[0] < min_pts:
        return pcd
    else:
        return pcd_aug


def simulate_cut_plane(pcd):
    bounding_box = trimesh.PointCloud(pcd).bounding_box
    # bounding_box = bounding_box.to_mesh()
    random_pts = bounding_box.sample_volume(2)

    # first point will be the point, second point will be used for the normal
    pt = random_pts[0]
    vec = random_pts[1] - pt
    normal = vec / np.linalg.norm(vec)
    normal = normal.reshape(3, 1)

    # get points that are on one side of this plane
    shifted_pcd = pcd - pt
    dots = np.matmul(shifted_pcd, normal)
    pos_inds = np.where(dots > 0)[0]
    neg_inds = np.where(dots < 0)[0]

    keep_pts = pcd[pos_inds]
    remove_pts = pcd[neg_inds]
    # mc_vis['scene/keep_pts'].delete()
    # util.meshcat_pcd_show(mc_vis, keep_pts, name='scene/keep_pts')
    # util.meshcat_pcd_show(mc_vis, remove_pts, name='scene/remove_pts')
    # from IPython import embed; embed()

    if keep_pts.shape[0] > 1500:
        pcd = keep_pts
    else:
        pcd = remove_pts
    return pcd


def get_parent_child_contact_point(parent_pcd, child_pcd):
    pbb = trimesh.PointCloud(parent_pcd).bounding_box_oriented
    cbb = trimesh.PointCloud(child_pcd).bounding_box_oriented

    pbb_pts = pbb.sample_volume(1000)
    cbb_pts = cbb.sample_volume(1000)

    pbb_mask = pbb.contains(cbb_pts)
    cbb_mask = cbb.contains(pbb_pts)

    bb_pts = np.concatenate([pbb_pts[cbb_mask], cbb_pts[pbb_mask]], axis=0)
    # util.meshcat_pcd_show(mc_vis, bb_pts, color=[0, 255, 0], name=f'scene/{i}/final_pcds_bb_pts_{i}')

    # if i in [0, 1]:
    #     util.meshcat_pcd_show(mc_vis, pf_pcd, color=[255, 0, 0], name=f'scene/{i}/final_pcds_parent_{i}')
    #     # util.meshcat_pcd_show(mc_vis, cf_pcd, color=[0, 0, 255], name=f'scene/{i}/final_pcds_child_{i}')
    #     # util.meshcat_pcd_show(mc_vis, bb_pts, color=[0, 255, 0], name=f'scene/{i}/final_pcds_bb_pts_{i}')
    #     # util.meshcat_trimesh_show(mc_vis, f'scene/{i}/final_bb_pts_bb_{i}', fine_grained_bb)
    #     # util.meshcat_pcd_show(mc_vis, demo_parent_qp, color=[255, 255, 0], name=f'scene/{i}/translate_qp_{i}')

    if bb_pts.shape[0] > 4:
        fine_grained_bb = trimesh.PointCloud(bb_pts).bounding_box_oriented.to_mesh()

        # intersection_trans = fine_grained_bb.sample_volume(1)
        intersection_trans = fine_grained_bb.centroid
        return intersection_trans
    else:
        return None



def simulate_deform_contact_point(pcd, deform_about_point, rot_grid, uniform=False):
    # high, low = 1.5, 0.5
    high, low = 2.0, 0.4
    
    # scale up the points about this specific location
    # scale_x, scale_y, scale_z = rhlb((1.5, 0.5)), rhlb((1.5, 0.5)), rhlb((1.5, 0.5))
    scale_x, scale_y, scale_z = rhlb((high, low)), rhlb((high, low)), rhlb((high, low))

    # apply the scaling to the place pcd
    pcd_contact_cent = pcd - deform_about_point
    if uniform:
        pcd_contact_cent = pcd_contact_cent * scale_x
        pcd_aug = pcd_contact_cent + deform_about_point
    else:
        # apply a random rotation, scale, and then unrotate
        rot_idx = np.random.randint(rot_grid.shape[0], size=1)
        rnd_rot = rot_grid[rot_idx]
        rnd_rot_T = np.eye(4); rnd_rot_T[:-1, :-1] = rnd_rot

        pcd_contact_cent = util.transform_pcd(pcd_contact_cent, rnd_rot_T) 

        pcd_contact_cent[:, 0] *= scale_x
        pcd_contact_cent[:, 1] *= scale_y
        pcd_contact_cent[:, 2] *= scale_z 

        pcd_contact_cent = util.transform_pcd(pcd_contact_cent, np.linalg.inv(rnd_rot_T))

        pcd_aug = pcd_contact_cent + deform_about_point

    return pcd_aug


def pcd_aug_full(pcd, rot_grid, deform_about_point=None, rnd_occlusion=True, cut_plane=True, 
                 per_point_noise=True, apply_deformation=True, uniform_scaling=False,
                 per_point_noise_std=0.01, n_cams=4, min_pts=None):
    assert not (deform_about_point is None and apply_deformation), '"deform_about_point" cannot be None while "apply_deformation" is True!'

    # build cameras that match what was used in the demos
    cam_int_mat = build_default_int_mat(height=default_img_size[0], width=default_img_size[1], fov_deg=60)

    # cam_cfg = camera_cfgs(width=int(cam_int_mat[0, 2]*2), height=int(cam_int_mat[1, 2]*2))
    # cam = RGBDCamera(cam_cfg)
    # # cam.depth_scale = 0.001
    # depth_scale_true = 0.001
    # cam.depth_scale = 1
    # cam.img_height = cam_cfg.CAM.SIM.HEIGHT
    # cam.img_width = cam_cfg.CAM.SIM.WIDTH
    # cam.depth_min = cam_cfg.CAM.SIM.ZNEAR
    # cam.depth_max = cam_cfg.CAM.SIM.ZFAR
    
    pcd_aug = pcd.copy()
    st = time.time()
    cam_poses = []
    for i in range(n_cams):
        cam_pose_mat = sample_cam_pose_mat(focus_pt=np.mean(pcd, axis=0))
        cam_poses.append(cam_pose_mat)

    # # get the initial point cloud from this demo
    # pcd_mean = np.mean(pcd, axis=0)
    # inliers = np.where(np.linalg.norm(pcd - pcd_mean, 2, 1) < 0.2)[0]
    # pcd = pcd[inliers]
             
    # augmentation: project to depth and mask out shapes
    if rnd_occlusion:
        if min_pts is None:
            min_pts = 1500
        pcd_aug = simulate_random_occlusions(pcd, cam_poses, cam_int_mat, select_prob=0.5, min_pts=min_pts)

    # convert 3D points in the world frame to be in the camera frame
    # augmentation: cut plane in 3D
    if cut_plane and (np.random.random() > 0.7):
        pcd_aug = simulate_cut_plane(pcd_aug)

    if per_point_noise:
        # augmentation: per-point noise
        per_point_noise_std = rhlb((per_point_noise_std, 0.0))
        # pt_noise = (np.random.random(size=pcd_aug.shape) - 0.5) * per_point_noise_std
        pt_noise = np.random.randn(*pcd_aug.shape) * per_point_noise_std
        pcd_aug = pcd_aug + pt_noise

    if apply_deformation:
        # augment based on the placing object "scaling point"

        # sph = trimesh.creation.uv_sphere(0.01) 
        # sph.apply_translation(deform_about_point)
        # util.meshcat_trimesh_show(mc_vis, 'scene/deform_about_point', sph)

        # print('pcd, deform_about_point', pcd_aug, deform_about_point)
        pcd_aug = simulate_deform_contact_point(pcd_aug, deform_about_point, rot_grid, uniform=False)

    if uniform_scaling and not apply_deformation:
        pcd_aug = simulate_deform_contact_point(pcd_aug, deform_about_point, rot_grid, uniform=True)

    if pcd_aug.shape[0] < 1500:
        while True:
            n_pts_left = 1500 - pcd_aug.shape[0]
            new_pts = pcd_aug[np.random.permutation(pcd_aug.shape[0])[:n_pts_left]]
            pcd_aug = np.vstack([pcd_aug, new_pts])

            if pcd_aug.shape[0] >= 1500:
                break

    return pcd_aug

