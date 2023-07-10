import os, os.path as osp
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import sys
import random
import numpy as np
import time
import signal
import torch, torch.nn as nn
import argparse
import shutil
import threading
import copy
import json
import trimesh

import pybullet as p
import meshcat

from airobot import Robot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils import common
from airobot.utils.pb_util import create_pybullet_client
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet

import rpdiff_robot.model.vnn_occupancy_net_pointnet_dgcnn as vnn_occupancy_network
from rpdiff_robot.utils import util, trimesh_util, torch3d_util, torch_util
from rpdiff_robot.utils.util import np2img
from rpdiff_robot.utils.plotly_save import plot3d

from rpdiff_robot.opt.optimizer import OccNetOptimizer
from rpdiff_robot.robot.multicam import MultiCams
from rpdiff_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rpdiff_robot.config.default_obj_cfg import get_obj_cfg_defaults
from rpdiff_robot.utils import path_util
from rpdiff_robot.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from rpdiff_robot.utils.pb2mc.pybullet_recorder import PyBulletRecorder
from rpdiff_robot.utils.eval_gen_utils import (
    soft_grasp_close, constraint_grasp_close, constraint_obj_world, constraint_grasp_open, 
    safeCollisionFilterPair, object_is_still_grasped, get_ee_offset, post_process_grasp_point,
    process_demo_data_rack, process_demo_data_shelf, process_xq_data, process_xq_rs_data, safeRemoveConstraint,
)

from rpdiff_robot.eval.evaluate_relations import infer_relation as infer_relation_ebm
from rpdiff_robot.eval.relations_baseline.multi_rpdiff_intersections import infer_relation_intersection, create_target_descriptors, filter_parent_child_pcd

from rpdiff_robot.utils.rpdiff_util import DescriptorField
from rpdiff_robot.model.policy import EBM

# sys.path.append(os.environ['REL_EBM_SRC_DIR'])
# from train_mug_bowl import (
#     EBM, DescriptorField, load_data, load_real_data, load_data_start, se3_equiv_encode, rot_grid,
#     expmap2quat, expmap2rotmat, quat2expmap,
#     quaternion_to_angle_axis, angle_axis_to_rotation_matrix)

cam_frame_scene_dict = {}
cam_up_vec = [0, 0, 1]
plotly_camera = {
    'up': {'x': cam_up_vec[0], 'y': cam_up_vec[1],'z': cam_up_vec[2]},
    'center': {'x': 0, 'y': 0, 'z': 0},
    'eye': {'x': 0, 'y': 0, 'z': 0},
}
plotly_scene = {
    'xaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
    'yaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
    'zaxis': {'nticks': 16, 'range': [-0.5, 0.5]},
}
cam_frame_scene_dict['camera'] = plotly_camera
cam_frame_scene_dict['scene'] = plotly_scene

NOISE_VALUE_LIST = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.16, 0.24, 0.32, 0.4]

def hide_link(obj_id, link_id): 
    if link_id is not None:
        p.changeVisualShape(obj_id, link_id, rgbaColor=[0, 0, 0, 0])


def show_link(obj_id, link_id, color):
    if link_id is not None:
        p.changeVisualShape(obj_id, link_id, rgbaColor=color)


def pb2mc_update(recorder, mc_vis, stop_event, run_event):
    iters = 0
    # while True:
    while not stop_event.is_set():
        run_event.wait()
        iters += 1
        recorder.add_keyframe()
        recorder.update_meshcat_current_state(mc_vis)
        time.sleep(1/230.0)


def create_target_desc_subdir(demo_path, parent_model_path, child_model_path):
    parent_model_name_full = parent_model_path.split('rpdiff_vnn/')[-1]
    child_model_name_full = child_model_path.split('rpdiff_vnn/')[-1]

    parent_model_name_specific = parent_model_name_full.split('.pth')[0].replace('/', '--')
    child_model_name_specific = child_model_name_full.split('.pth')[0].replace('/', '--')
    
    subdir_name = f'parent_model--{parent_model_name_specific}_child--{child_model_name_specific}'
    dirname = osp.join(demo_path, subdir_name)
    util.safe_makedirs(dirname)
    return dirname


def main(args):

    #####################################################################################
    # set up all generic experiment info
    assert args.relation_method in ['intersection', 'ebm'], 'Invalid argument for --relation_method'

    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    signal.signal(signal.SIGINT, util.signal_handler)

    demo_path  = osp.join(path_util.get_rpdiff_data(), 'relation_demos', args.rel_demo_exp)
    demo_files = [fn for fn in sorted(os.listdir(demo_path)) if fn.endswith('.npz')]
    demos = []
    for f in demo_files:
        demo = np.load(demo_path+'/'+f, allow_pickle=True)
        demos.append(demo)

    parent_model_name_full = args.parent_model_path.split('rpdiff_vnn/')[-1]
    child_model_name_full = args.child_model_path.split('rpdiff_vnn/')[-1]

    parent_model_save_path = parent_model_name_full.split('/')[0]
    child_model_save_path = child_model_name_full.split('/')[0]

    parent_model_name_specific = parent_model_name_full.split('.pth')[0].replace('/', '--')
    child_model_name_specific = child_model_name_full.split('.pth')[0].replace('/', '--')
    if args.rel_model_path is None:
        ebm_model_name_specific = 'ebm_model_None'
    else:
        ebm_model_name_specific = args.rel_model_path.split('.pth')[0].replace('/', '--')
    # print(f'Parent model name specific: {parent_model_name_specific}, Child model name specific: {child_model_name_specific}')
    print(f'Parent model name specific: {parent_model_name_specific}, Child model name specific: {child_model_name_specific}, EBM name specific: {ebm_model_name_specific}')
    
    expstr = f'exp--{args.exp}_demo-exp--{args.rel_demo_exp}'
    # modelstr = f'parent_model--{args.parent_model_path}_child_model--{args.child_model_path}'
    modelstr = f'parent_model--{parent_model_save_path}_child_model--{child_model_save_path}'
    seedstr = 'seed--' + str(args.seed)
    experiment_name = '_'.join([expstr, modelstr, seedstr])
    # experiment_name_spec_model = f'parent--{parent_model_name_specific}_child--{child_model_name_specific}'
    experiment_name_spec_model = f'parent--{parent_model_name_specific}_child--{child_model_name_specific}_ebm--{ebm_model_name_specific}'

    eval_save_dir_root = osp.join(path_util.get_rpdiff_eval_data(), args.eval_data_dir, experiment_name)
    demo_aug_dir_root = osp.join(path_util.get_rpdiff_data(), args.demo_aug_dir, experiment_name)
    eval_save_dir = osp.join(eval_save_dir_root, experiment_name_spec_model)
    demo_aug_dir = osp.join(demo_aug_dir_root, experiment_name_spec_model)
    util.safe_makedirs(eval_save_dir_root)
    util.safe_makedirs(demo_aug_dir_root)
    util.safe_makedirs(eval_save_dir)
    util.safe_makedirs(demo_aug_dir)

    zmq_url = 'tcp://127.0.0.1:6000'
    # zmq_url = 'tcp://127.0.0.1:6001'
    log_warn(f'Starting meshcat at zmq_url: {zmq_url}')
    # mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
    mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis['scene'].delete()

    pb_client = create_pybullet_client(gui=args.pybullet_viz, opengl_render=True, realtime=True, server=args.pybullet_server)
    recorder = PyBulletRecorder(pb_client=pb_client)
    recorder.clear()
    print('here')

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # general experiment + environment setup/scene generation configs
    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_rpdiff_config(), 'eval_cfgs', args.config + '.yaml')
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info('Config file %s does not exist, using defaults' % config_fname)
    # cfg.freeze()

    eval_teleport_imgs_dir = osp.join(eval_save_dir, 'teleport_imgs')
    util.safe_makedirs(eval_teleport_imgs_dir)

    save_dir = osp.join(path_util.get_rpdiff_eval_data(), 'multi_class', args.exp)
    util.safe_makedirs(save_dir)
    
    #####################################################################################
    # load in the parent/child model and the cameras

    parent_model_path = osp.join(path_util.get_rpdiff_model_weights(), args.parent_model_path)
    child_model_path = osp.join(path_util.get_rpdiff_model_weights(), args.child_model_path)

    if args.parent_model_path_ebm is not None:
        parent_model_path_ebm = osp.join(path_util.get_rpdiff_model_weights(), args.parent_model_path_ebm)
    else:
        parent_model_path_ebm = parent_model_path
    if args.child_model_path_ebm is not None:
        child_model_path_ebm = osp.join(path_util.get_rpdiff_model_weights(), args.child_model_path_ebm)
    else:
        child_model_path_ebm = child_model_path

    parent_model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()
    child_model = vnn_occupancy_network.VNNOccNet(latent_dim=256, model_type='pointnet', return_features=True, sigmoid=True).cuda()

    parent_model.load_state_dict(torch.load(parent_model_path))
    child_model.load_state_dict(torch.load(child_model_path))

    cams = MultiCams(cfg.CAMERA, pb_client, n_cams=cfg.N_CAMERAS)
    cam_info = {}
    cam_info['pose_world'] = []
    for cam in cams.cams:
        cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))

    #####################################################################################
    # load all the multi class mesh info

    mesh_data_dirs = {
        'rack': 'rack_centered_obj', 
        'mug': 'mug_centered_obj_normalized', 
        'bottle': 'bottle_centered_obj_normalized', 
        'bowl': 'bowl_centered_obj_normalized',
        'syn_rack_easy': 'syn_racks_easy_obj',
        'syn_rack_hard': 'syn_racks_hard_obj',
        'cuboid': 'cuboids_centered_obj',
        'box_container': 'box_containers_unnormalized'
    }
    mesh_data_dirs = {k: osp.join(path_util.get_rpdiff_obj_descriptions(), v) for k, v in mesh_data_dirs.items()}

    bad_ids = {
        'rack': [],
        'syn_rack_easy': [],
        'syn_rack_hard': [],
        'cuboid': [],
        'bowl': bad_shapenet_bowls_ids_list,
        'mug': bad_shapenet_mug_ids_list,
        'bottle': bad_shapenet_bottles_ids_list,
        'box_container': []
    }

    upright_orientation_dict = {
        'rack': common.euler2quat([0, 0, 0]).tolist(),
        'mug': common.euler2quat([np.pi/2, 0, 0]).tolist(), 
        'bottle': common.euler2quat([np.pi/2, 0, 0]).tolist(), 
        'bowl': common.euler2quat([np.pi/2, 0, 0]).tolist(),
        'syn_rack_easy': common.euler2quat([0, 0, 0]).tolist(),
        'syn_rack_hard': common.euler2quat([0, 0, 0]).tolist(),
        'cuboid': common.euler2quat([0, 0, 0]).tolist(),
        'box_container': common.euler2quat([0, 0, 0]).tolist(),
    }

    mesh_names = {}
    for k, v in mesh_data_dirs.items():
        # get train samples
        objects_raw = os.listdir(v)
        objects_filtered = [fn for fn in objects_raw if fn.split('/')[-1] not in bad_ids[k]]
        # objects_filtered = objects_raw
        total_filtered = len(objects_filtered)
        train_n = int(total_filtered * 0.9); test_n = total_filtered - train_n

        train_objects = sorted(objects_filtered)[:train_n]
        test_objects = sorted(objects_filtered)[train_n:]

        log_info('\n\n\nTest objects: ')
        log_info(test_objects)
        # log_info('\n\n\n')

        mesh_names[k] = objects_filtered

    obj_classes = list(mesh_names.keys())

    scale_high, scale_low = cfg.MESH_SCALE_HIGH, cfg.MESH_SCALE_LOW
    scale_default = cfg.MESH_SCALE_DEFAULT

    # cfg.OBJ_SAMPLE_Y_HIGH_LOW = [0.3, -0.3]
    cfg.OBJ_SAMPLE_Y_HIGH_LOW = [-0.35, 0.175]
    x_low, x_high = cfg.OBJ_SAMPLE_X_HIGH_LOW
    y_low, y_high = cfg.OBJ_SAMPLE_Y_HIGH_LOW
    table_z = cfg.TABLE_Z

    #####################################################################################
    # load all the parent/child info

    parent_class = args.parent_class
    child_class = args.child_class
    is_parent_shapenet_obj = args.is_parent_shapenet_obj
    is_child_shapenet_obj = args.is_child_shapenet_obj

    pcl = ['parent', 'child']
    pc_master_dict = dict(parent={}, child={})
    pc_master_dict['parent']['class'] = parent_class
    pc_master_dict['child']['class'] = child_class
    
    valid_load_pose_types = ['any_pose', 'demo_pose', 'random_upright']
    assert args.parent_load_pose_type in valid_load_pose_types, f'Invalid string value for args.parent_load_pose_type! Must be in {", ".join(valid_load_pose_types)}'
    assert args.child_load_pose_type in valid_load_pose_types, f'Invalid string value for args.child_load_pose_type! Must be in {", ".join(valid_load_pose_types)}'

    pc_master_dict['parent']['load_pose_type'] = args.parent_load_pose_type
    pc_master_dict['child']['load_pose_type'] = args.child_load_pose_type

    # load in ids for objects that can be used for testing
    pc_master_dict['parent']['test_ids'] = np.loadtxt(osp.join(path_util.get_rpdiff_share(), '%s_test_object_split.txt' % parent_class), dtype=str).tolist()
    pc_master_dict['child']['test_ids'] = np.loadtxt(osp.join(path_util.get_rpdiff_share(), '%s_test_object_split.txt' % child_class), dtype=str).tolist()

    # process these to remove the file type
    pc_master_dict['parent']['test_ids'] = [val.split('.')[0] for val in pc_master_dict['parent']['test_ids']]
    pc_master_dict['child']['test_ids'] = [val.split('.')[0] for val in pc_master_dict['child']['test_ids']]

    log_info(f'Test ids (parent): {", ".join(pc_master_dict["parent"]["test_ids"])}')
    log_info(f'Test ids (child): {", ".join(pc_master_dict["child"]["test_ids"])}')

    for pc in pcl:
        object_class = pc_master_dict[pc]['class']
        if object_class == 'mug':
            avoid_ids = bad_shapenet_mug_ids_list + cfg.MUG.AVOID_SHAPENET_IDS
        elif object_class == 'bowl':
            avoid_ids = bad_shapenet_bowls_ids_list + cfg.BOWL.AVOID_SHAPENET_IDS
        elif object_class == 'bottle':
            avoid_ids = bad_shapenet_bottles_ids_list + cfg.BOTTLE.AVOID_SHAPENET_IDS 
        else:
            avoid_ids = []

        pc_master_dict[pc]['avoid_ids'] = avoid_ids

    pc_master_dict['parent']['xhl'] = [x_high, x_low]
    pc_master_dict['parent']['yhl'] = [y_high, 0.075]
    # pc_master_dict['parent']['yhl'] = [y_high, 0.05]

    pc_master_dict['child']['xhl'] = [x_high, x_low]
    pc_master_dict['child']['yhl'] = [-0.2, y_low]
    # pc_master_dict['child']['yhl'] = [-0.075, y_low]
    # pc_master_dict['child']['yhl'] = [-0.05, y_low]
    
    # get the class specific ranges for scaling the objects
    for pc in pcl:
        if pc_master_dict[pc]['class'] == 'mug':
            pc_master_dict[pc]['scale_hl'] = [0.35, 0.25] 
            pc_master_dict[pc]['scale_default'] = 0.3
        if pc_master_dict[pc]['class'] == 'bowl':
            pc_master_dict[pc]['scale_hl'] = [0.325, 0.15] 
            pc_master_dict[pc]['scale_default'] = 0.3
        if pc_master_dict[pc]['class'] == 'bottle':
            pc_master_dict[pc]['scale_hl'] = [0.35, 0.2] 
            pc_master_dict[pc]['scale_default'] = 0.3
        if pc_master_dict[pc]['class'] == 'syn_rack_easy':
            # pc_master_dict[pc]['scale_hl'] = [1.1, 0.9]
            # pc_master_dict[pc]['scale_default'] = 1.0
            pc_master_dict[pc]['scale_hl'] = [0.35, 0.25]
            pc_master_dict[pc]['scale_default'] = 0.3
        if pc_master_dict[pc]['class'] == 'box_container':
            pc_master_dict[pc]['scale_hl'] = [1.1, 0.9]
            pc_master_dict[pc]['scale_default'] = 1.0

    # pc_master_dict['parent']['scale_hl'] = [0.45, 0.25]
    # pc_master_dict['parent']['scale_default'] = 0.3
    # pc_master_dict['child']['scale_hl'] = [1.1, 0.9]
    # pc_master_dict['child']['scale_default'] = 1.0

    # pc_master_dict['parent']['scale_hl'] = [1.1, 0.9]
    # pc_master_dict['parent']['scale_default'] = 1.0
    # pc_master_dict['child']['scale_hl'] = [0.45, 0.25]
    # pc_master_dict['child']['scale_default'] = 0.3

    pc_master_dict['parent']['model_path'] = parent_model_path
    pc_master_dict['child']['model_path'] = child_model_path

    pc_master_dict['parent']['model'] = parent_model
    pc_master_dict['child']['model'] = child_model

    parent_model.load_state_dict(torch.load(parent_model_path))
    child_model.load_state_dict(torch.load(child_model_path))

    # put the data in our pc master
    pc_master_dict['parent']['demo_start_pcds'] = []
    pc_master_dict['parent']['demo_final_pcds'] = []
    pc_master_dict['child']['demo_start_pcds'] = []
    pc_master_dict['child']['demo_final_pcds'] = []
    for pc in pcl:
        for idx, demo in enumerate(demos):
            s_pcd = demo['multi_obj_start_pcd'].item()[pc]
            f_pcd = demo['multi_obj_final_pcd'].item()[pc]

            pc_master_dict[pc]['demo_start_pcds'].append(s_pcd)
            pc_master_dict[pc]['demo_final_pcds'].append(f_pcd)

    # load data from demos in case we want to test on the shapes we trained on
    for pc in pcl:
        pc_master_dict[pc]['demo_ids'] = [dat['multi_object_ids'].item()[pc] for dat in demos] 
        pc_master_dict[pc]['demo_start_poses'] = [dat['multi_obj_start_obj_pose'].item()[pc] for dat in demos] 

    #####################################################################################
    # prepare the target descriptors 

    target_desc_subdir = create_target_desc_subdir(demo_path, parent_model_path, child_model_path)
    target_desc_fname = osp.join(demo_path, target_desc_subdir, args.target_desc_name)
    if args.relation_method == 'intersection':
        if not osp.exists(target_desc_fname) or args.new_descriptors:
            print(f'\n\n\nCreating target descriptors for this parent model + child model, and these demos\nSaving to {target_desc_fname}\n\n\n')
            n_demos = 'all' if args.n_demos < 1 else args.n_demos
            if args.add_noise:
                add_noise = True
                noise_value = NOISE_VALUE_LIST[args.noise_idx]
            else:
                add_noise = False
                noise_value = 0.0001
            create_target_descriptors(parent_model, child_model, pc_master_dict, target_desc_fname, 
                                      cfg, query_scale=0.035, scale_pcds=False, target_rounds=3, pc_reference=args.pc_reference,
                                      skip_alignment=args.skip_alignment, n_demos=n_demos, manual_target_idx=args.target_idx, add_noise=add_noise, interaction_pt_noise_std=noise_value,
                                      visualize=True, mc_vis=mc_vis)

    if osp.exists(target_desc_fname):
        log_info(f'Loading target descriptors from file:\n{target_desc_fname}')
        target_descriptors_data = np.load(target_desc_fname)
        parent_overall_target_desc = target_descriptors_data['parent_overall_target_desc']
        child_overall_target_desc = target_descriptors_data['child_overall_target_desc']
        parent_overall_target_desc = torch.from_numpy(parent_overall_target_desc).float().cuda()
        child_overall_target_desc = torch.from_numpy(child_overall_target_desc).float().cuda()
        parent_query_points = target_descriptors_data['parent_query_points']
        child_query_points = copy.deepcopy(parent_query_points)

        log_info(f'Making a copy of the target descriptors in eval folder')
        shutil.copy(target_desc_fname, eval_save_dir)

        parent_optimizer = OccNetOptimizer(
            parent_model,
            query_pts=parent_query_points,
            query_pts_real_shape=parent_query_points,
            opt_iterations=args.opt_iterations,
            cfg=cfg.OPTIMIZER)

        child_optimizer = OccNetOptimizer(
            child_model,
            query_pts=parent_query_points,
            query_pts_real_shape=parent_query_points,
            opt_iterations=args.opt_iterations,
            cfg=cfg.OPTIMIZER)

        parent_optimizer.setup_meshcat(mc_vis)
        child_optimizer.setup_meshcat(mc_vis)

    #########################################################################
    # Set up the relational energy model

    if args.relation_method == 'ebm' or args.refine_with_ebm:
        rel_model = EBM().cuda()
        rel_model_path = osp.join(path_util.get_rpdiff_model_weights(), 'relation_energy/cachedir', args.rel_model_path)
        assert osp.exists(rel_model_path), f'Path to relation energy model: {rel_model_path} does not exist!'
        rel_checkpoint = torch.load(rel_model_path, map_location=torch.device('cpu'))
        rel_model.load_state_dict(rel_checkpoint['model_state_dict'])

    #####################################################################################
    # prepare the simuation environment

    # put table at right spot
    table_ori = common.euler2quat([0, 0, np.pi / 2])

    # table_urdf_fname = osp.join(path_util.get_rpdiff_descriptions(), 'hanging/table/table_manual.urdf')
    table_urdf_fname = osp.join(path_util.get_rpdiff_descriptions(), 'hanging/table/table_rack_manual.urdf')
    # table_urdf_fname = osp.join(path_util.get_rpdiff_descriptions(), 'hanging/table/table_rack.urdf')
    table_id = pb_client.load_urdf(table_urdf_fname,
                            cfg.TABLE_POS,
                            cfg.TABLE_ORI,
                            scaling=1.0)
    recorder.register_object(table_id, table_urdf_fname)

    rec_stop_event = threading.Event()
    rec_run_event = threading.Event()
    rec_th = threading.Thread(target=pb2mc_update, args=(recorder, mc_vis, rec_stop_event, rec_run_event))# , mc_vis))
    rec_th.daemon = True
    rec_th.start()

    pause_mc_thread = lambda pause_bool : rec_run_event.clear() if pause_bool else rec_run_event.set()
    pause_mc_thread(False)

    rack_link_id = 0

    hide_link(table_id, rack_link_id)

    eval_imgs_dir = osp.join(eval_save_dir, 'eval_imgs')
    util.safe_makedirs(eval_imgs_dir)
    eval_cam = RGBDCameraPybullet(cams._camera_cfgs(), pb_client)
    eval_cam.setup_camera(
        focus_pt=[0.4, 0.0, table_z],
        dist=0.9,
        yaw=45,
        pitch=-25,
        roll=0)

    #####################################################################################
    # start experiment: sample parent and child object on each iteration and infer the relation
    place_success_list = []
    full_cfg_dict = {}
    for k, v in args.__dict__.items():
        full_cfg_dict[k] = v
    for k, v in util.cn2dict(cfg).items():
        full_cfg_dict[k] = v
    full_cfg_fname = osp.join(eval_save_dir, 'full_exp_cfg.txt')
    json.dump(full_cfg_dict, open(full_cfg_fname, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    for iteration in range(args.start_iteration, args.num_iterations):
        #####################################################################################
        # set up the trial
        
        demo_idx = np.random.randint(len(demos))
        demo = demos[demo_idx]
        if args.test_on_train:
            parent_id = pc_master_dict['parent']['demo_ids'][demo_idx]
            child_id = pc_master_dict['child']['demo_ids'][demo_idx]
        else:
            parent_id = random.sample(pc_master_dict['parent']['test_ids'], 1)[0]
            child_id = random.sample(pc_master_dict['child']['test_ids'], 1)[0]

        id_str = f'Parent ID: {parent_id}, Child ID: {child_id}'
        log_info(id_str)

        eval_iter_dir = osp.join(eval_save_dir, 'trial_%d' % iteration)
        util.safe_makedirs(eval_iter_dir)

        #####################################################################################
        # load parent/child objects into the scene -- mesh file, pose, and pybullet object id

        if is_parent_shapenet_obj:
            parent_obj_file = osp.join(mesh_data_dirs[parent_class], parent_id, 'models/model_normalized.obj')
            parent_obj_file_dec = parent_obj_file.split('.obj')[0] + '_dec.obj'
        else:
            parent_obj_file = osp.join(mesh_data_dirs[parent_class], parent_id + '.obj')
            parent_obj_file_dec = parent_obj_file.split('.obj')[0] + '_dec.obj'

        if is_child_shapenet_obj:
            child_obj_file = osp.join(mesh_data_dirs[child_class], child_id, 'models/model_normalized.obj')
            child_obj_file_dec = child_obj_file.split('.obj')[0] + '_dec.obj'
        else:
            child_obj_file = osp.join(mesh_data_dirs[child_class], child_id + '.obj')
            child_obj_file_dec = child_obj_file.split('.obj')[0] + '_dec.obj'

        new_parent_scale = None
        # check if bottle/container are the right sizes
        if parent_class == 'box_container' and child_class == 'bottle':
            # from IPython import embed; embed()
            container_mesh = trimesh.load(parent_obj_file_dec)
            bottle_mesh = trimesh.load(child_obj_file_dec)
            container_mesh.apply_scale(pc_master_dict['parent']['scale_default'])
            bottle_mesh.apply_scale(pc_master_dict['child']['scale_default'])

            # make upright
            container_upright_orientation = upright_orientation_dict['box_container']
            bottle_upright_orientation = upright_orientation_dict['bottle']
            container_upright_mat = np.eye(4); container_upright_mat[:-1, :-1] = common.quat2rot(container_upright_orientation)
            bottle_upright_mat = np.eye(4); bottle_upright_mat[:-1, :-1] = common.quat2rot(bottle_upright_orientation)

            container_mesh.apply_transform(container_upright_mat)
            bottle_mesh.apply_transform(bottle_upright_mat)

            # get the 2D projection of the vertices
            container_2d = np.asarray(container_mesh.vertices)[:, :-1]
            bottle_2d = np.asarray(bottle_mesh.vertices)[:, :-1]
            container_flat = np.hstack([container_2d, np.zeros(container_2d.shape[0]).reshape(-1, 1)])
            bottle_flat = np.hstack([bottle_2d, np.zeros(bottle_2d.shape[0]).reshape(-1, 1)])

            # container_extents = trimesh.PointCloud(container_flat).bounding_box.extents
            # bottle_extents = trimesh.PointCloud(bottle_flat).bounding_box.extents
            container_box = trimesh.PointCloud(container_flat).bounding_box #_oriented
            bottle_box = trimesh.PointCloud(bottle_flat).bounding_box #_oriented

            with recorder.meshcat_scene_lock:
                util.meshcat_trimesh_show(mc_vis, 'scene/container_box', container_box.to_mesh().apply_translation([0.0, 0.2, 0.0]), color=(255, 0, 0))
                util.meshcat_trimesh_show(mc_vis, 'scene/bottle_box', bottle_box.to_mesh().apply_translation([0.0, -0.2, 0.0]), color=(0, 0, 255))

            container_extents = container_box.extents
            bottle_extents = bottle_box.extents

            new_parent_scale = None
            # if bottle_extents[0] > np.max(container_extents):
            # if bottle_extents[0] > (1.25 * np.min(container_extents[:-1])):
            # if bottle_extents[0] > (0.75 * np.min(container_extents[:-1])):
            if np.max(bottle_extents) > (0.75 * np.min(container_extents[:-1])):
                # new_parent_scale = bottle_extents[0] * (np.random.random() * (2.5 - 1.5) + 1.5) / np.max(container_extents)
                # new_parent_scale = bottle_extents[0] * (np.random.random() * (2 - 1.5) + 1.5) / np.min(container_extents[:-1])
                new_parent_scale = np.max(bottle_extents) * (np.random.random() * (2 - 1.5) + 1.5) / np.min(container_extents[:-1])

            ext_str = f'\nContainer extents: {", ".join([str(val) for val in container_extents])}, \nBottle extents: {", ".join([str(val) for val in bottle_extents])}\n'
            log_info(ext_str)

            # check the 2D extents
            # from IPython import embed; embed()

        for pc in pcl:
            pc_master_dict[pc]['mesh_file'] = parent_obj_file if pc == 'parent' else child_obj_file
            pc_master_dict[pc]['mesh_file_dec'] = parent_obj_file_dec if pc == 'parent' else child_obj_file_dec

            scale_high, scale_low = pc_master_dict[pc]['scale_hl']
            if pc == 'parent':
                if new_parent_scale is None:
                    scale_default = pc_master_dict[pc]['scale_default']
                else:
                    log_warn(f'Setting new parent scale to: {new_parent_scale:.3f} to ensure parent is large enough for child')
                    scale_default = new_parent_scale
            else:
                scale_default = pc_master_dict[pc]['scale_default']

            if args.rand_mesh_scale:
                mesh_scale = [np.random.random() * (scale_high - scale_low) + scale_low] * 3
            else:
                mesh_scale=[scale_default] * 3

            pc_master_dict[pc]['mesh_scale'] = mesh_scale

            object_class = pc_master_dict[pc]['class']
            upright_orientation = upright_orientation_dict[object_class]

            load_pose_type = pc_master_dict[pc]['load_pose_type']
            x_high, x_low = pc_master_dict[pc]['xhl']
            y_high, y_low = pc_master_dict[pc]['yhl']
            # load_pose_type = 'test_on_train_pose'  # TODO
            if load_pose_type == 'any_pose':
                if object_class in ['bowl', 'bottle']:
                    rp = np.random.rand(2) * (2 * np.pi / 3) - (np.pi / 3)
                    ori = common.euler2quat([rp[0], rp[1], 0]).tolist()
                else:
                    rpy = np.random.rand(3) * (2 * np.pi / 3) - (np.pi / 3)
                    ori = common.euler2quat([rpy[0], rpy[1], rpy[2]]).tolist()

                pos = [
                    np.random.random() * (x_high - x_low) + x_low,
                    np.random.random() * (y_high - y_low) + y_low,
                    table_z]
                pose = pos + ori
                rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
                pose_w_yaw = util.transform_pose(util.list2pose_stamped(pose), util.pose_from_matrix(rand_yaw_T))
                pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
            else:
                if load_pose_type == 'demo_pose':
                    obj_start_pose_demo = pc_master_dict[pc]['demo_start_poses'][demo_idx]
                    pos, ori = obj_start_pose_demo[:3], obj_start_pose_demo[3:]
                else:
                    pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, table_z]
                    pose = util.list2pose_stamped(pos + upright_orientation)
                    rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
                    pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
                    pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]

            # convert mesh with vhacd
            obj_obj_file, obj_obj_file_dec = pc_master_dict[pc]['mesh_file'], pc_master_dict[pc]['mesh_file_dec']

            if not osp.exists(obj_obj_file_dec):
                p.vhacd(
                    obj_obj_file,
                    obj_obj_file_dec,
                    'log.txt',
                    concavity=0.0025,
                    alpha=0.04,
                    beta=0.05,
                    gamma=0.00125,
                    minVolumePerCH=0.0001,
                    resolution=1000000,
                    depth=20,
                    planeDownsampling=4,
                    convexhullDownsampling=4,
                    pca=0,
                    mode=0,
                    convexhullApproximation=1
                )

            obj_id = pb_client.load_geom(
                'mesh',
                mass=0.01,
                mesh_scale=mesh_scale,
                visualfile=obj_obj_file_dec,
                collifile=obj_obj_file_dec,
                base_pos=pos,
                base_ori=ori)
            recorder.register_object(obj_id, obj_obj_file_dec, scaling=mesh_scale)
            safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=rack_link_id, enableCollision=False)

            p.changeDynamics(obj_id, -1, lateralFriction=0.5)

            o_cid = None
            # if load_pose_type == 'any_pose':
            # if load_pose_type == 'any_pose' and pc == 'child':
            if (object_class in ['syn_rack_easy', 'syn_rack_hard', 'syn_rack_med']) or (load_pose_type == 'any_pose' and pc == 'child'):
            # if False:
                o_cid = constraint_obj_world(obj_id, pos, ori)
                pb_client.set_step_sim(False)
            pc_master_dict[pc]['o_cid'] = o_cid
            safeCollisionFilterPair(obj_id, table_id, -1, -1, enableCollision=True)
            p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
            time.sleep(1.5)

            pc_master_dict[pc]['pb_obj_id'] = obj_id

        # get object point cloud
        rgb_imgs = []
        depth_imgs = []
        cam_poses = []
        cam_intrinsics = []
        seg_depth_imgs = []
        seg_idxs = []
        obj_pcd_pts = []
        table_pcd_pts = []

        parent_pcd_pts = []
        child_pcd_pts = []

        pc_obs_info = {}
        pc_obs_info['pcd'] = {}
        pc_obs_info['pcd_pts'] = {}
        pc_obs_info['pcd_pts']['parent'] = []
        pc_obs_info['pcd_pts']['child'] = [] 

        for i, cam in enumerate(cams.cams): 
            cam_int = cam.cam_int_mat
            cam_ext = cam.cam_ext_mat
            cam_intrinsics.append(cam_int)
            cam_poses.append(cam_ext)

            # get image and raw point cloud
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)

            # flatten and find corresponding pixels in segmentation mask
            flat_seg = seg.flatten()
            flat_depth = depth.flatten()

            for pc in pcl:
                obj_id = pc_master_dict[pc]['pb_obj_id']
                obj_inds = np.where(flat_seg == obj_id)
                table_inds = np.where(flat_seg == table_id)
                seg_depth = flat_depth[obj_inds[0]]  
                
                obj_pts = pts_raw[obj_inds[0], :]
                # obj_pcd_pts.append(util.crop_pcd(obj_pts))
                pc_obs_info['pcd_pts'][pc].append(util.crop_pcd(obj_pts))

            table_pts = pts_raw[table_inds[0], :][::int(table_inds[0].shape[0]/500)]
            table_pcd_pts.append(table_pts)

            depth_imgs.append(depth)
            rgb_imgs.append(rgb)
            seg_depth_imgs.append(seg_depth)
            seg_idxs.append(obj_inds)
        
        for pc, obj_pcd_pts in pc_obs_info['pcd_pts'].items():
            target_obj_pcd_obs = np.concatenate(obj_pcd_pts, axis=0)  # object shape point cloud
            target_pts_mean = np.mean(target_obj_pcd_obs, axis=0)
            inliers = np.where(np.linalg.norm(target_obj_pcd_obs - target_pts_mean, 2, 1) < 0.2)[0]
            target_obj_pcd_obs = target_obj_pcd_obs[inliers]
            
            pc_obs_info['pcd'][pc] = target_obj_pcd_obs

        parent_pcd = pc_obs_info['pcd']['parent']
        child_pcd = pc_obs_info['pcd']['child']

        parent_rpdiff = DescriptorField(parent_model, parent_pcd)
        child_rpdiff = DescriptorField(child_model, child_pcd)

        parent_center = torch.Tensor(parent_pcd.mean(axis=0))[None, :].cuda()
        child_center = torch.Tensor(child_pcd.mean(axis=0))[None, :].cuda()
        FLAGS = args
        logdir = 'pcd_vis'

        if args.relation_method == 'ebm':
            parent_rpdiff = DescriptorField(parent_model, parent_pcd)
            child_rpdiff = DescriptorField(child_model, child_pcd)

            parent_center = torch.Tensor(parent_pcd.mean(axis=0))[None, :].cuda()
            child_center = torch.Tensor(child_pcd.mean(axis=0))[None, :].cuda()
            FLAGS = args
            logdir = 'pcd_vis'
            
            # while True:
            log_info(f'[RELATION], Loading model weights for relation inference')
            parent_model.load_state_dict(torch.load(parent_model_path_ebm))
            child_model.load_state_dict(torch.load(child_model_path_ebm))
            
            relative_trans_list = []
            energy_list = []
            for i in range(10):
                relative_trans, energy = infer_relation_ebm(mc_vis, rel_model, parent_rpdiff, child_rpdiff, parent_center, child_center, FLAGS, logdir, step=0, return_energy=True)
                relative_trans_list.append(relative_trans)
                energy_list.append(energy)

            best_energy_idx = np.argmin(energy_list)
            log_info(f'[RELATION], Best energy value: {energy_list[best_energy_idx]:.5f}')
            relative_trans = relative_trans_list[best_energy_idx]
        else:  # intersection
            log_info(f'[INTERSECTION], Loading model weights for multi RRP inference')
            parent_model.load_state_dict(torch.load(parent_model_path))
            child_model.load_state_dict(torch.load(child_model_path))
            pause_mc_thread(True)
            relative_trans, out_best_desc = infer_relation_intersection(
                mc_vis, parent_optimizer, child_optimizer, 
                parent_overall_target_desc, child_overall_target_desc, 
                parent_pcd, child_pcd, parent_query_points, child_query_points, opt_visualize=args.opt_visualize,
                return_desc=True)
            pause_mc_thread(False)

            # refine
            if args.refine_with_ebm:
                # pause_mc_thread(True)
                child_pcd_refine = util.transform_pcd(child_pcd, relative_trans)
                parent_rpdiff = DescriptorField(parent_model, parent_pcd)
                child_rpdiff = DescriptorField(child_model, child_pcd_refine)

                parent_center = torch.Tensor(parent_pcd.mean(axis=0))[None, :].cuda()
                child_center = torch.Tensor(child_pcd_refine.mean(axis=0))[None, :].cuda()
                FLAGS = args
                logdir = 'pcd_vis'
                
                # while True:
                log_info(f'[RELATION], Loading model weights for relation inference')
                parent_model.load_state_dict(torch.load(parent_model_path_ebm))
                child_model.load_state_dict(torch.load(child_model_path_ebm))
                
                relative_trans_list = []
                energy_list = []
                for i in range(10):
                    relative_trans_i, energy = infer_relation_ebm(mc_vis, rel_model, parent_rpdiff, child_rpdiff, parent_center, child_center, FLAGS, logdir, step=0, return_energy=True)
                    relative_trans_list.append(relative_trans_i)
                    energy_list.append(energy)

                best_energy_idx = np.argmin(energy_list)
                log_info(f'[RELATION], Best energy value: {energy_list[best_energy_idx]:.5f}')
                relative_trans_refine = relative_trans_list[best_energy_idx]

                transformed_child1 = util.transform_pcd(child_pcd_refine, relative_trans_refine)

                pause_mc_thread(True)
                with recorder.meshcat_scene_lock:
                    util.meshcat_pcd_show(mc_vis, child_pcd_refine, color=[255, 0, 0], name='scene/child_pcd_refine')
                    util.meshcat_pcd_show(mc_vis, transformed_child1, color=[255, 255, 0], name='scene/child_pcd_refine_1')
                pause_mc_thread(False)
                # from IPython import embed; embed()
                relative_trans = np.matmul(relative_trans_refine, relative_trans)
        
        # from IPython import embed; embed()
        time.sleep(1.0)

        parent_obj_id = pc_master_dict['parent']['pb_obj_id']
        child_obj_id = pc_master_dict['child']['pb_obj_id']
        start_child_pose = np.concatenate(pb_client.get_body_state(child_obj_id)[:2]).tolist()
        start_child_pose_mat = util.matrix_from_pose(util.list2pose_stamped(start_child_pose))
        final_child_pose_mat = np.matmul(relative_trans, start_child_pose_mat)

        start_parent_pose = np.concatenate(pb_client.get_body_state(parent_obj_id)[:2]).tolist()
        start_parent_pose_mat = util.matrix_from_pose(util.list2pose_stamped(start_parent_pose))
        upright_orientation = upright_orientation_dict[pc_master_dict['parent']['class']]
        upright_parent_ori_mat = common.quat2rot(upright_orientation)

        pb_client.set_step_sim(True)
        if pc_master_dict['parent']['load_pose_type'] == 'any_pose':
            # get the relative transformation to make it upright
            # start_parent_pose = np.concatenate(pb_client.get_body_state(parent_obj_id)[:2]).tolist()
            # start_parent_pose_mat = util.matrix_from_pose(util.list2pose_stamped(start_parent_pose))
            # upright_parent_ori_mat = common.quat2rot(upright_orientation)
            upright_parent_pose_mat = copy.deepcopy(start_parent_pose_mat); upright_parent_pose_mat[:-1, :-1] = upright_parent_ori_mat
            relative_upright_pose_mat = np.matmul(upright_parent_pose_mat, np.linalg.inv(start_parent_pose_mat))

            upright_parent_pos, upright_parent_ori = start_parent_pose[:3], common.rot2quat(upright_parent_ori_mat)
            pb_client.reset_body(parent_obj_id, upright_parent_pos, upright_parent_ori)

            final_child_pose_mat = np.matmul(relative_upright_pose_mat, final_child_pose_mat)

        final_parent_pose = np.concatenate(pb_client.get_body_state(parent_obj_id)[:2]).tolist()
        final_child_pose = util.pose_stamped2list(util.pose_from_matrix(final_child_pose_mat))
        final_child_pos, final_child_ori = final_child_pose[:3], final_child_pose[3:]

        pb_client.reset_body(child_obj_id, final_child_pos, final_child_ori)
        if pc_master_dict['parent']['class'] not in ['syn_rack_easy', 'syn_rack_med']:
            safeRemoveConstraint(pc_master_dict['parent']['o_cid'])
        if pc_master_dict['child']['class'] not in ['syn_rack_easy', 'syn_rack_med']:
            safeRemoveConstraint(pc_master_dict['child']['o_cid'])

        final_parent_pcd = copy.deepcopy(pc_obs_info['pcd']['parent'])
        final_child_pcd = util.transform_pcd(pc_obs_info['pcd']['child'], relative_trans)
        with recorder.meshcat_scene_lock:
            util.meshcat_pcd_show(mc_vis, final_child_pcd, color=[255, 0, 255], name='scene/final_child_pcd')
        safeCollisionFilterPair(pc_master_dict['child']['pb_obj_id'], table_id, -1, -1, enableCollision=False)

        time.sleep(3.0)

        pb_client.set_step_sim(False)

        # evaluation criteria
        time.sleep(2.0)
        
        success_crit_dict = {}
        kvs = {}

        obj_surf_contacts = p.getContactPoints(pc_master_dict['child']['pb_obj_id'], pc_master_dict['parent']['pb_obj_id'], -1, -1)
        touching_surf = len(obj_surf_contacts) > 0
        success_crit_dict['touching_surf'] = touching_surf
        # from IPython import embed; embed()
        if parent_class == 'box_container' and child_class == 'bottle':
            bottle_final_pose = np.concatenate(p.getBasePositionAndOrientation(pc_master_dict['child']['pb_obj_id'])[:2]).tolist()

            # get the y-axis in the body frame
            bottle_body_y = common.quat2rot(bottle_final_pose[3:])[:, 1]
            bottle_body_y = bottle_body_y / np.linalg.norm(bottle_body_y)

            # get the angle deviation from the vertical
            angle_from_upright = util.angle_from_3d_vectors(bottle_body_y, np.array([0, 0, 1]))
            bottle_upright = angle_from_upright < args.upright_ori_diff_thresh
            success_crit_dict['bottle_upright'] = bottle_upright

            # bottle_final_ori = np.asarray(p.getBasePositionAndOrientation(pc_master_dict['child']['pb_obj_id'])[1])
            # upright_ori = upright_orientation_dict['bottle']
            # bottle_upright_pose = np.concatenate([bottle_final_pose[:3], upright_ori])
            # upright_ori_diff = util.pose_difference_np(bottle_final_pose, bottle_upright_pose)[1].item()

            # bottle_upright = upright_ori_diff < args.upright_ori_diff_thresh
            
            # from IPython import embed; embed()

        # take an image to make sure it's good
        eval_rgb = eval_cam.get_images(get_rgb=True)[0]
        eval_img_fname = osp.join(eval_imgs_dir, f'{iteration}.png')
        util.np2img(eval_rgb.astype(np.uint8), eval_img_fname)

        ##########################################################################
        # upside down check for too much inter-penetration
        pb_client.set_step_sim(True)

        # remove constraints, if there are any
        safeRemoveConstraint(pc_master_dict['parent']['o_cid'])
        safeRemoveConstraint(pc_master_dict['child']['o_cid'])
        
        # first, reset everything
        pb_client.reset_body(parent_obj_id, start_parent_pose[:3], start_parent_pose[3:])
        pb_client.reset_body(child_obj_id, start_child_pose[:3], start_child_pose[3:])

        # then, compute a new position + orientation for the parent object, that is upside down
        upside_down_ori_mat = np.matmul(common.euler2rot([np.pi, 0, 0]), upright_parent_ori_mat)
        upside_down_pose_mat = np.eye(4); upside_down_pose_mat[:-1, :-1] = upside_down_ori_mat; upside_down_pose_mat[:-1, -1] = start_parent_pose[:3]
        upside_down_pose_mat[2, -1] += 0.15  # move up in z a bit
        parent_upside_down_pose_list = util.pose_stamped2list(util.pose_from_matrix(upside_down_pose_mat))

        # reset parent to this state and constrain to world
        pb_client.reset_body(parent_obj_id, parent_upside_down_pose_list[:3], parent_upside_down_pose_list[3:]) 
        ud_cid = constraint_obj_world(parent_obj_id, parent_upside_down_pose_list[:3], parent_upside_down_pose_list[3:]) 

        # get the final relative pose of the child object
        final_child_pose_parent = util.convert_reference_frame(
            pose_source=util.pose_from_matrix(final_child_pose_mat),
            pose_frame_target=util.pose_from_matrix(start_parent_pose_mat),
            pose_frame_source=util.unit_pose()
        )
        # get the final world frame pose of the child object in upside down pose
        final_child_pose_upside_down = util.convert_reference_frame(
            pose_source=final_child_pose_parent,
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.pose_from_matrix(upside_down_pose_mat)
        )
        final_child_pose_upside_down_list = util.pose_stamped2list(final_child_pose_upside_down)
        final_child_pose_upside_down_mat = util.matrix_from_pose(final_child_pose_upside_down)

        # reset child to this state
        pb_client.reset_body(child_obj_id, final_child_pose_upside_down_list[:3], final_child_pose_upside_down_list[3:]) 

        # turn on the simulation and wait for a couple seconds
        pb_client.set_step_sim(False)
        time.sleep(2.0)

        # check if they are still in contact (they shouldn't be)
        ud_obj_surf_contacts = p.getContactPoints(parent_obj_id, child_obj_id, -1, -1)
        ud_touching_surf = len(ud_obj_surf_contacts) > 0
        success_crit_dict['fell_off_upside_down'] = not ud_touching_surf

        #########################################################################

        place_success = np.all(np.asarray(list(success_crit_dict.values())))
        # place_success = touching_surf and child_obj_is_upright
        
        # place_success = False  # TODO, per class
        place_success_list.append(place_success)
        log_str = 'Iteration: %d, ' % iteration

        kvs['Place Success'] = sum(place_success_list) / float(len(place_success_list))

        if parent_class == 'box_container' and child_class == 'bottle':
            kvs['Angle From Upright'] = angle_from_upright

        for k, v in kvs.items():
            log_str += '%s: %.3f, ' % (k, v)
        for k, v in success_crit_dict.items():
            log_str += '%s: %s, ' % (k, v)

        id_str = f', parent_id: {parent_id}, child_id: {child_id}'
        log_info(log_str + id_str)

        eval_iter_dir = osp.join(eval_save_dir, f'trial_{iteration}')
        util.safe_makedirs(eval_iter_dir)
        sample_fname = osp.join(eval_iter_dir, 'success_rate_relation.npz')
        full_cfg_fname = osp.join(eval_iter_dir, 'full_config.json')
        results_txt_fname = osp.join(eval_iter_dir, 'results.txt')
        np.savez(
            sample_fname,
            parent_id=parent_id,
            child_id=child_id,
            is_parent_shapenet_obj=is_parent_shapenet_obj,
            is_child_shapenet_obj=is_child_shapenet_obj,
            success_criteria_dict=success_crit_dict,
            place_success=place_success,
            place_success_list=place_success_list,
            mesh_file=obj_obj_file,
            args=args.__dict__,
            cfg=util.cn2dict(cfg),
        )
        json.dump(full_cfg_dict, open(full_cfg_fname, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

        results_txt_dict = {}
        results_txt_dict['place_success'] = place_success
        results_txt_dict['place_success_list'] = place_success_list
        results_txt_dict['current_success_rate'] = sum(place_success_list) / float(len(place_success_list))
        results_txt_dict['success_criteria_dict'] = success_crit_dict
        open(results_txt_fname, 'w').write(str(results_txt_dict))

        eval_img_fname2 = osp.join(eval_iter_dir, f'{iteration}.png')
        util.np2img(eval_rgb.astype(np.uint8), eval_img_fname2)

        # save it as a new demo and label the success
        demo_aug_fname = osp.join(demo_aug_dir, f'demo_aug_{iteration}.npz')
        aug_save_dict = {
            'success': place_success,
            'multi_obj_names': dict(parent=pc_master_dict['parent']['class'], child=pc_master_dict['child']['class']),
            'multi_obj_start_pcd': dict(parent=pc_obs_info['pcd']['parent'], child=pc_obs_info['pcd']['child']),
            'multi_obj_final_pcd': dict(parent=final_parent_pcd, child=final_child_pcd),
            'grasp_pose_world': dict(parent=None, child=None),
            'place_pose_world': dict(parent=None, child=None),
            'grasp_joints': dict(parent=None, child=None),
            'place_joints': dict(parent=None, child=None),
            'ee_link': None,
            'gripper_type': None,
            'pcd_pts': None,
            'processed_pcd': None,
            'rgb_imgs': None,
            'depth_imgs': None,
            'cam_intrinsics': cam_intrinsics,
            'cam_poses': cam_poses,
            'multi_object_ids': dict(parent=parent_id, child=child_id),
            'real_sim': 'sim', # real or sim
            'multi_obj_start_obj_pose': dict(parent=start_parent_pose, child=start_child_pose),
            'multi_obj_final_obj_pose': dict(parent=final_parent_pose, child=final_child_pose),
            'multi_obj_mesh_file': dict(parent=pc_master_dict['parent']['mesh_file'], child=pc_master_dict['child']['mesh_file']),
            'multi_obj_mesh_file_dec': dict(parent=pc_master_dict['parent']['mesh_file_dec'], child=pc_master_dict['child']['mesh_file_dec']),
            'multi_obj_best_descriptors': out_best_desc,
        }

        np.savez(demo_aug_fname, **aug_save_dict)

        pause_mc_thread(True)
        for pc in pcl:
            obj_id = pc_master_dict[pc]['pb_obj_id']
            pb_client.remove_body(obj_id)
            recorder.remove_object(obj_id, mc_vis)
        mc_vis['scene/child_pcd_refine'].delete()
        mc_vis['scene/child_pcd_refine_1'].delete()
        mc_vis['scene/final_child_pcd'].delete()
        pause_mc_thread(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--parent_class', type=str, required=True)
    parser.add_argument('--child_class', type=str, required=True)
    parser.add_argument('--rel_demo_exp', type=str, required=True)

    parser.add_argument('--parent_model_path', type=str, required=True)
    parser.add_argument('--child_model_path', type=str, required=True)
    parser.add_argument('--parent_model_path_ebm', type=str, default=None)
    parser.add_argument('--child_model_path_ebm', type=str, default=None)
    parser.add_argument('--rel_model_path', type=str, default=None)

    parser.add_argument('--config', type=str, default='base_cfg')
    parser.add_argument('--exp', type=str, default='debug_eval')
    parser.add_argument('--eval_data_dir', type=str, default='eval_data')
    parser.add_argument('--demo_aug_dir', type=str, default='demo_aug')

    parser.add_argument('--opt_visualize', action='store_true')
    parser.add_argument('--opt_iterations', type=int, default=100)
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--save_all_opt_results', action='store_true', help='If True, then we will save point clouds for all optimization runs, otherwise just save the best one (which we execute)')
    parser.add_argument('--start_iteration', type=int, default=0)

    parser.add_argument('--single_instance', action='store_true')
    parser.add_argument('--rand_mesh_scale', action='store_true')
    parser.add_argument('--only_test_ids', action='store_true')
    parser.add_argument('--parent_load_pose_type', type=str, default='demo_pose', help='Must be in [any_pose, demo_pose, random_upright]')
    parser.add_argument('--child_load_pose_type', type=str, default='demo_pose', help='Must be in [any_pose, demo_pose, random_upright]')

    # rel ebm flags
    parser.add_argument('--no_trans', action='store_true', help='whether or not to include translation opt')
    parser.add_argument('--load_start', action='store_true', help='if we should load the start point clouds from demos')
    parser.add_argument('--rand_pose', action='store_true')
    parser.add_argument('--rand_rot', action='store_true')
    parser.add_argument('--test_idx', default=0, type=int)
    parser.add_argument('--real', action='store_true')

    parser.add_argument('--pybullet_viz', action='store_true')
    parser.add_argument('--pybullet_server', action='store_true')
    parser.add_argument('--is_parent_shapenet_obj', action='store_true')
    parser.add_argument('--is_child_shapenet_obj', action='store_true')
    parser.add_argument('--test_on_train', action='store_true')

    parser.add_argument('--relation_method', type=str, default='intersection', help='either "intersection", "ebm"')
    parser.add_argument('--create_target_desc', action='store_true', help='If True and --relation_method="intersection", then create the target descriptors if a file does not already exist containing them')
    parser.add_argument('--target_desc_name', type=str, default='target_descriptors.npz')
    parser.add_argument('--refine_with_ebm', action='store_true')
    parser.add_argument('--no_grasp', action='store_true')
    parser.add_argument('--pc_reference', type=str, default='parent', help='either "parent" or "child"')
    parser.add_argument('--skip_alignment', action='store_true')
    parser.add_argument('--new_descriptors', action='store_true')
    parser.add_argument('--n_demos', type=int, default=0)
    parser.add_argument('--target_idx', type=int, default=-1)

    # some threshold
    parser.add_argument('--upright_ori_diff_thresh', type=float, default=np.deg2rad(15))

    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--noise_idx', type=int, default=0)

    args = parser.parse_args()

    main(args)
