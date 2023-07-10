import os, os.path as osp
import random
import numpy as np
import time
import signal
import torch
import argparse
import threading
import datetime
import copy
import json
import trimesh
from scipy.spatial.transform import Rotation as R

import pybullet as p
import meshcat

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from airobot.utils.pb_util import create_pybullet_client
from airobot.sensor.camera.rgbdcam_pybullet import RGBDCameraPybullet

from rpdiff.utils import util, config_util

from rpdiff.robot.multicam import MultiCams
from rpdiff.robot.floating_sphere_gripper import FloatingSphereGripper
from rpdiff.utils import path_util
from rpdiff.share.globals import bad_shapenet_mug_ids_list, bad_shapenet_bowls_ids_list, bad_shapenet_bottles_ids_list
from rpdiff.utils.pb2mc.pybullet_meshcat import PyBulletMeshcat
from rpdiff.utils.mesh_util import inside_mesh, three_util
from rpdiff.utils.eval_gen_utils import constraint_obj_world, safeCollisionFilterPair, safeRemoveConstraint

from rpdiff.utils.relational_policy.multistep_pose_regression import policy_inference_methods_dict
from rpdiff.model.coarse_affordance import CoarseAffordanceVoxelRot
from rpdiff.model.transformer.policy import (
    NSMTransformerSingleTransformationRegression, 
    NSMTransformerSingleTransformationRegressionCVAE,
    NSMTransformerSingleSuccessClassifier)


def check_ckpt_load_latest(ckpt_path: str) -> str:
    if ckpt_path is None or ckpt_path == 'None':
        return None
    if not ckpt_path.endswith('.pth'):
        # assume this is the directory containing
        model_names = [val.replace('.pth', '') for val in os.listdir(ckpt_path) if (('latest' not in val) and (val.endswith('.pth')))]
        model_nums = [int(val.split('_')[-1]) for val in model_names]
        latest_num = max(model_nums)
        ckpt_path_full = osp.join(ckpt_path, f'model_{latest_num}.pth')
        return ckpt_path_full
    else:
        return ckpt_path


def pb2mc_update(
        recorder: PyBulletMeshcat, 
        mc_vis: meshcat.Visualizer, 
        stop_event: threading.Event, 
        run_event: threading.Event) -> None:
    iters = 0
    # while True:
    while not stop_event.is_set():
        run_event.wait()
        iters += 1
        try:
            recorder.add_keyframe()
            recorder.update_meshcat_current_state(mc_vis)
        except KeyError as e:
            print(f'PyBullet to Meshcat thread Exception: {e}')
            time.sleep(0.1)
        time.sleep(1/230.0)


def main(args: config_util.AttrDict) -> None:

    #####################################################################################
    # set up all generic experiment info
    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    signal.signal(signal.SIGINT, util.signal_handler)

    expstr = args.experiment.experiment_name

    seedstr = f'seed_{str(args.seed)}'

    experiment_name = expstr
    experiment_name_spec = seedstr

    eval_save_dir_root = osp.join(path_util.get_rpdiff_eval_data(), args.experiment.eval_data_dir, experiment_name)
    eval_save_dir = osp.join(eval_save_dir_root, experiment_name_spec)
    util.safe_makedirs(eval_save_dir_root)
    util.safe_makedirs(eval_save_dir)
    
    if args.new_meshcat:
        mc_vis = meshcat.Visualizer()
    else:
        zmq_url = f'tcp://127.0.0.1:{args.port_vis}'
        mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis['scene'].delete()

    pb_client = create_pybullet_client(
        gui=args.experiment.pybullet_viz, 
        opengl_render=True, 
        realtime=True, 
        server=args.experiment.pybullet_server)
    recorder = PyBulletMeshcat(pb_client=pb_client)
    recorder.clear()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    eval_teleport_imgs_dir = osp.join(eval_save_dir, 'teleport_imgs')
    util.safe_makedirs(eval_teleport_imgs_dir)

    infer_kwargs = {}

    #####################################################################################
    # load all the multi class mesh info

    mesh_data_dirs = {}
    upright_euler_dict = {}
    bad_ids = {}
    for obj_cat, obj_cat_info in args.objects.categories.items():
        mesh_data_dirs[obj_cat] = obj_cat_info.mesh_dir

        if obj_cat_info.upright_euler is None:
            upright_euler_dict[obj_cat] = args.objects.default_upright_euler
        else:
            upright_euler_dict[obj_cat] = obj_cat_info.upright_euler
        
        if obj_cat == 'mug':
            bad_ids[obj_cat] = bad_shapenet_mug_ids_list
        elif obj_cat == 'bowl':
            bad_ids[obj_cat] = bad_shapenet_bowls_ids_list
        elif obj_cat == 'bottle':
            bad_ids[obj_cat] = bad_shapenet_bottles_ids_list
        else:
            bad_ids[obj_cat] = []

    mesh_data_dirs = {k: osp.join(path_util.get_rpdiff_obj_descriptions(), v) for k, v in mesh_data_dirs.items()}
    upright_orientation_dict = {k: R.from_euler('xyz', v).as_quat().tolist() for k, v in upright_euler_dict.items()}

    mesh_names = {}
    train_n_dict = {}
    for k, v in mesh_data_dirs.items():
        # get train samples
        objects_raw = os.listdir(v)
        objects_filtered = [fn for fn in objects_raw if (fn.split('/')[-1] not in bad_ids[k]) and ('dec' not in fn and fn.endswith('.obj'))]
        # objects_filtered = objects_raw
        total_filtered = len(objects_filtered)
        train_n = int(total_filtered * 0.9); test_n = total_filtered - train_n

        train_objects = sorted(objects_filtered)[:train_n]
        test_objects = sorted(objects_filtered)[train_n:]

        log_info('\n\n\nTest objects: ')
        log_info(test_objects)

        train_n_dict[k] = train_n
        mesh_names[k] = objects_filtered

    obj_classes = list(mesh_names.keys())
    
    env_args = args.environment
    scale_high, scale_low = env_args.mesh_scale_high, env_args.mesh_scale_low
    scale_default = env_args.mesh_scale_default

    x_low, x_high = env_args.obj_sample_x_high_low
    y_low, y_high = env_args.obj_sample_y_high_low
    table_z = env_args.table_z

    #####################################################################################
    # load all the parent/child info

    parent_class = args.experiment.parent_class
    child_class = args.experiment.child_class
    is_parent_shapenet_obj = args.experiment.is_parent_shapenet_obj
    is_child_shapenet_obj = args.experiment.is_child_shapenet_obj

    pcl = ['parent', 'child']
    pc_master_dict = dict(parent={}, child={})
    pc_master_dict['parent']['class'] = parent_class
    pc_master_dict['child']['class'] = child_class
    
    valid_load_pose_types = ['any_pose', 'demo_pose', 'random_upright']
    assert args.experiment.eval.parent_load_pose_type in valid_load_pose_types, f'Invalid string value for args.experiment.eval.parent_load_pose_type! Must be in {", ".join(valid_load_pose_types)}'
    assert args.experiment.eval.child_load_pose_type in valid_load_pose_types, f'Invalid string value for args.experiment.eval.child_load_pose_type! Must be in {", ".join(valid_load_pose_types)}'

    pc_master_dict['parent']['load_pose_type'] = args.experiment.eval.parent_load_pose_type
    pc_master_dict['child']['load_pose_type'] = args.experiment.eval.child_load_pose_type

    # load in ids for objects that can be used for testing
    parent_test_split_fname = osp.join(path_util.get_rpdiff_share(), '%s_test_object_split.txt' % parent_class)
    child_test_split_fname = osp.join(path_util.get_rpdiff_share(), '%s_test_object_split.txt' % child_class)
    train_np = train_n_dict[parent_class]
    train_nc = train_n_dict[child_class]
    if osp.exists(parent_test_split_fname):
        pc_master_dict['parent']['test_ids'] = np.loadtxt(parent_test_split_fname, dtype=str).tolist()
    else:
        pc_master_dict['parent']['test_ids'] = [val for val in sorted(mesh_names[parent_class])[train_np:] if ('_dec' not in val) and (val.endswith('.obj'))]
    if osp.exists(child_test_split_fname):
        pc_master_dict['child']['test_ids'] = np.loadtxt(child_test_split_fname, dtype=str).tolist()
    else:
        pc_master_dict['child']['test_ids'] = [val for val in sorted(mesh_names[child_class])[train_nc:] if ('_dec' not in val) and (val.endswith('.obj'))]

    # process these to remove the file type
    pc_master_dict['parent']['test_ids'] = [val.split('.')[0] for val in pc_master_dict['parent']['test_ids']]
    pc_master_dict['child']['test_ids'] = [val.split('.')[0] for val in pc_master_dict['child']['test_ids']]

    log_info(f'Test ids (parent): {", ".join(pc_master_dict["parent"]["test_ids"])}')
    log_info(f'Test ids (child): {", ".join(pc_master_dict["child"]["test_ids"])}')

    # setup bounds for sampling random object positions in the environment
    pc_master_dict['parent']['xhl'] = [0.65, 0.2]
    pc_master_dict['parent']['yhl'] = [0.6, -0.6]
    pc_master_dict['child']['xhl'] = [0.65, 0.2]
    pc_master_dict['child']['yhl'] = [0.6, -0.6]
    
    pc_object_class = dict(parent=pc_master_dict['parent']['class'], child=pc_master_dict['child']['class'])
    for pc in pcl:
        pc_master_dict[pc]['scale_hl'] = args.objects.categories[pc_object_class[pc]].scale_hl 
        pc_master_dict[pc]['scale_default'] = args.objects.categories[pc_object_class[pc]].scale_default 

    #########################################################################
    # Set up the models (feat encoder, voxel affordance, pose refinement, success)

    model_ckpt_logdir = osp.join(path_util.get_rpdiff_model_weights(), args.experiment.logdir) 

    reso_grid = args.data.voxel_grid.reso_grid  # args.reso_grid
    padding_grid = args.data.voxel_grid.padding

    raster_pts = three_util.get_raster_points(reso_grid, padding=padding_grid)
    raster_pts = raster_pts.reshape(reso_grid, reso_grid, reso_grid, 3)
    raster_pts = raster_pts.transpose(2, 1, 0, 3)
    raster_pts = raster_pts.reshape(-1, 3)

    rot_grid_samples = args.data.rot_grid_samples
    rot_grid = util.generate_healpix_grid(size=rot_grid_samples) 
    args.data.rot_grid_bins = rot_grid.shape[0]

    exp_args = args.experiment

    ############################################################################
    # Setup each model

    ###
    # Load pose refinement model, loss, and optimizer
    ###

    pose_refine_model_path = None
    pr_model = None
    if exp_args.load_pose_regression:
        
        # assumes model path exists
        pose_refine_model_path = osp.join(model_ckpt_logdir, args.experiment.eval.pose_refine_model_name)
        pose_refine_model_path = check_ckpt_load_latest(pose_refine_model_path)
        pose_refine_ckpt = torch.load(pose_refine_model_path, map_location=torch.device('cpu'))

        # update config with config used during training
        config_util.update_recursive(args.model.refine_pose, config_util.recursive_attr_dict(pose_refine_ckpt['args']['model']['refine_pose']))

        # model (pr = pose refine)
        pr_type = args.model.refine_pose.type
        pr_args = config_util.copy_attr_dict(args.model[pr_type])
        if args.model.refine_pose.get('model_kwargs') is not None:
            custom_pr_args = args.model.refine_pose.model_kwargs[pr_type]
            config_util.update_recursive(pr_args, custom_pr_args)
        
        if pr_type == 'nsm_transformer':
            pr_model_cls = NSMTransformerSingleTransformationRegression
        elif pr_type == 'nsm_transformer_cvae':
            pr_model_cls = NSMTransformerSingleTransformationRegressionCVAE
        else:
            raise ValueError(f'Unrecognized: {pr_type}')

        pr_model = pr_model_cls(
            mc_vis=mc_vis, 
            feat_dim=args.model.refine_pose.feat_dim, 
            **pr_args).cuda()

        pr_model.load_state_dict(pose_refine_ckpt['refine_pose_model_state_dict'])

    ###
    # Load success classifier and optimizer
    ###

    success_model_path = None
    success_model = None
    if exp_args.load_success_classifier:

        # assumes model path exists
        success_model_path = osp.join(model_ckpt_logdir, args.experiment.eval.success_model_name)
        success_model_path = check_ckpt_load_latest(success_model_path)
        success_ckpt = torch.load(success_model_path, map_location=torch.device('cpu'))

        # update config with config used during training
        config_util.update_recursive(args.model.success, config_util.recursive_attr_dict(success_ckpt['args']['model']['success']))

        sc_type = args.model.success.type
        sc_args = config_util.copy_attr_dict(args.model[sc_type])
        if args.model.success.get('model_kwargs') is not None:
            custom_sc_args = args.model.success.model_kwargs[sc_type]
            config_util.update_recursive(sc_args, custom_sc_args)

        # model
        if sc_type == 'nsm_transformer':
            success_model_cls = NSMTransformerSingleSuccessClassifier
        else:
            raise ValueError(f'Unrecognized success model type: {sc_type}')

        sc_args.sigmoid = True
        
        success_model = success_model_cls(
            mc_vis=mc_vis,
            feat_dim=args.model.success.feat_dim,
            **sc_args).cuda()

        success_model.load_state_dict(success_ckpt['success_model_state_dict'])

    ###
    # Load coarse affordance model, loss, and optimizer
    ###

    voxel_aff_model_path = None
    coarse_aff_model = None
    if exp_args.load_coarse_aff:

        # assumes model path exists
        voxel_aff_model_path = osp.join(model_ckpt_logdir, args.experiment.eval.voxel_aff_model_name)
        voxel_aff_model_path = check_ckpt_load_latest(voxel_aff_model_path)
        voxel_aff_ckpt = torch.load(voxel_aff_model_path, map_location=torch.device('cpu'))

        # update config with config used during training
        config_util.update_recursive(args.model.coarse_aff, config_util.recursive_attr_dict(voxel_aff_ckpt['args']['model']['coarse_aff']))

        # model
        coarse_aff_type = args.model.coarse_aff.type
        coarse_aff_args = config_util.copy_attr_dict(args.model[coarse_aff_type])
        if args.model.coarse_aff.get('model_kwargs') is not None:
            custom_coarse_aff_args = args.model.coarse_aff.model_kwargs[coarse_aff_args]
            config_util.update_recursive(coarse_aff_args, custom_coarse_aff_args)

        coarse_aff_model = CoarseAffordanceVoxelRot(
            mc_vis=mc_vis, 
            feat_dim=args.model.coarse_aff.feat_dim,
            rot_grid_dim=args.data.rot_grid_bins,
            padding=args.data.voxel_grid.padding,
            voxel_reso_grid=args.data.voxel_grid.reso_grid,
            scene_encoder_kwargs=coarse_aff_args).cuda()

        coarse_aff_model.load_state_dict(voxel_aff_ckpt['coarse_aff_model_state_dict'])
        
        if args.model.coarse_aff.get('multi_model') is not None:
            if args.model.coarse_aff.multi_model:

                coarse_aff_args2 = config_util.copy_attr_dict(args.model[coarse_aff_type])
                if args.model.coarse_aff.get('model_kwargs2') is not None:
                    custom_coarse_aff_args2 = args.model.coarse_aff.model_kwargs2[coarse_aff_type]
                    config_util.update_recursive(coarse_aff_args2, custom_coarse_aff_args2)
                
                # hacky thing we shouldn't need?
                args.model.coarse_aff = config_util.recursive_attr_dict(args.model.coarse_aff)
                voxel_reso_grid2 = args.data.voxel_grid.reso_grid
                voxel_reso_grid2 = util.set_if_not_none(voxel_reso_grid2, args.model.coarse_aff.model2.voxel_grid.reso_grid)

                padding2 = args.data.voxel_grid.padding
                padding2 = util.set_if_not_none(padding2, args.data.voxel_grid.padding)

                coarse_aff_model2 = CoarseAffordanceVoxelRot(
                    mc_vis=mc_vis, 
                    feat_dim=args.model.coarse_aff.feat_dim,
                    rot_grid_dim=args.data.rot_grid_bins,
                    padding=padding2,
                    voxel_reso_grid=voxel_reso_grid2,
                    scene_encoder_kwargs=coarse_aff_args2).cuda()

                infer_kwargs['coarse_aff_model2'] = {}
                infer_kwargs['coarse_aff_model2']['model'] = coarse_aff_model2
                infer_kwargs['coarse_aff_model2']['reso_grid'] = voxel_reso_grid2
                infer_kwargs['coarse_aff_model2']['padding'] = padding2

                coarse_aff_model2.load_state_dict(voxel_aff_ckpt['coarse_aff_model_state_dict2'])

    #####################################################################################
    # prepare function used for inference using set of trained models

    infer_relation_policy = policy_inference_methods_dict[args.experiment.eval.inference_method]

    #####################################################################################
    # prepare the simuation environment

    # put table at right spot
    table_urdf_fname = osp.join(path_util.get_rpdiff_descriptions(), 'franka_panda_table/table_manual.urdf')
    table_id = pb_client.load_urdf(table_urdf_fname,
                            env_args.table_pos,
                            env_args.table_ori,
                            scaling=1.0)
    recorder.register_object(table_id, table_urdf_fname)

    rec_stop_event = threading.Event()
    rec_run_event = threading.Event()
    rec_th = threading.Thread(target=pb2mc_update, args=(recorder, mc_vis, rec_stop_event, rec_run_event))
    rec_th.daemon = True
    rec_th.start()

    pause_mc_thread = lambda pause_bool : rec_run_event.clear() if pause_bool else rec_run_event.set()
    pause_mc_thread(False)

    table_base_id = 0

    eval_imgs_dir = osp.join(eval_save_dir, 'eval_imgs')
    util.safe_makedirs(eval_imgs_dir)
    eval_cam = None

    #####################################################################################
    # start experiment: sample parent and child object on each iteration and infer the relation

    # Set up experiment run/config logging
    nowstr = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_logs = osp.join(eval_save_dir, 'run_logs')
    util.safe_makedirs(run_logs)
    run_log_folder = osp.join(run_logs, nowstr)
    util.safe_makedirs(run_log_folder)
    
    # Save full name of model paths used
    if exp_args.load_pose_regression:
        args.experiment.eval.pose_refine_model_name_full = pose_refine_model_path
    if exp_args.load_success_classifier:
        args.experiment.eval.success_model_name_full = success_model_path
    if exp_args.load_coarse_aff:
        args.experiment.eval.voxel_aff_model_name_full = voxel_aff_model_path

    place_success_list = []
    full_cfg_dict = {}
    full_cfg_dict['args'] = config_util.recursive_dict(args)
    full_cfg_fname = osp.join(eval_save_dir, 'full_exp_cfg.txt')
    json.dump(full_cfg_dict, open(full_cfg_fname, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    #####################################################################################
    # load  floating gripper

    floating_hand = FloatingSphereGripper(pb_client=pb_client)
    floating_hand.hide_hand()
    if args.experiment.show_floating_hand_meshcat:
        recorder.register_object(floating_hand.hand_id, floating_hand.urdf_file)

    #####################################################################################
    eval_bookshelf_task = parent_class == 'syn_bookshelf' and child_class == 'syn_book'
    # eval_cabinet_task = parent_class == 'syn_cabinet' and child_class == 'syn_can'
    eval_cabinet_task = 'syn_cabinet' in parent_class and 'syn_can' in child_class
    eval_mug_rack_multi_task = parent_class == 'syn_rack_med' and child_class == 'mug'

    scene_extents = args.data.coarse_aff.scene_extents 
    scene_scale = 1 / np.max(scene_extents)
    args.data.coarse_aff.scene_scale = scene_scale 

    if util.exists_and_true(exp_args.eval, 'multi_aff_rot'):
        infer_kwargs['multi_aff_rot'] = True

    for iteration in range(exp_args.start_iteration, exp_args.num_iterations):
        #####################################################################################
        # set up the trial
        
        pause_mc_thread(True)

        parent_id_list = random.sample(pc_master_dict['parent']['test_ids'], np.random.randint(1, exp_args.n_parent_instances+1))
        child_id_list = [random.sample(pc_master_dict['child']['test_ids'], 1)[0]]

        if eval_bookshelf_task:
            print('Using book that matches the shelf')
            child_id = parent_id_list[0].replace('shelf_', '_')  # assume we have a single bookshelf now
            child_id_list = [child_id]

        if eval_cabinet_task:
            cabinet_name = parent_id_list[0].split('/')[-1].replace('.obj', '').replace('_dec', '')
            parent_mesh_dir = mesh_data_dirs[parent_class]
            saved_available_poses_fname = osp.join(parent_mesh_dir, 'open_slot_poses', cabinet_name + '_open_slot_poses.npz')
            loaded_poses = np.load(saved_available_poses_fname, allow_pickle=True)
            avail_pose_info_all = loaded_poses['avail_top_poses']

            rnd_idx = np.random.randint(avail_pose_info_all.shape[0])
            rnd_avail_pose = avail_pose_info_all[rnd_idx]
            req_dim_r = rnd_avail_pose['dims']['r']
            req_dim_h = rnd_avail_pose['dims']['h']

        for parent_id in parent_id_list:
            if parent_id.endswith('_dec'):
                parent_id = parent_id.replace('_dec', '')
        for child_id in child_id_list:
            if child_id.endswith('_dec'):
                child_id = child_id.replace('_dec', '')

        id_str = f'Parent ID: {parent_id_list}, Child ID: {child_id}'
        log_info(id_str)

        eval_iter_dir = osp.join(eval_save_dir, 'trial_%d' % iteration)
        util.safe_makedirs(eval_iter_dir)

        #####################################################################################
        # load parent/child objects into the scene -- mesh file, pose, and pybullet object id

        if is_parent_shapenet_obj:
            parent_obj_file_list = [osp.join(mesh_data_dirs[parent_class], parent_id, 'models/model_normalized.obj') for parent_id in parent_id_list]
            parent_obj_file_dec_list = [parent_obj_file.split('.obj')[0] + '_dec.obj' for parent_obj_file in parent_obj_file_list]
        else:
            parent_obj_file_list = [osp.join(mesh_data_dirs[parent_class], parent_id + '.obj') for parent_id in parent_id_list]
            parent_obj_file_dec_list = [parent_obj_file.split('.obj')[0] + '_dec.obj' for parent_obj_file in parent_obj_file_list]

        if is_child_shapenet_obj:
            child_obj_file_list = [osp.join(mesh_data_dirs[child_class], child_id, 'models/model_normalized.obj') for child_id in child_id_list]
            child_obj_file_dec_list = [child_obj_file.split('.obj')[0] + '_dec.obj' for child_obj_file in child_obj_file_list]
        else:
            child_obj_file_list = [osp.join(mesh_data_dirs[child_class], child_id + '.obj') for child_id in child_id_list]
            child_obj_file_dec_list = [child_obj_file.split('.obj')[0] + '_dec.obj' for child_obj_file in child_obj_file_list]

        current_scene_concat_mesh = None
        current_parent_scale_list = []
        current_child_scale_list = []
        for pc in pcl:
            obj_file_list = parent_obj_file_list if pc == 'parent' else child_obj_file_list
            pc_master_dict[pc]['o_cid'] = []
            pc_master_dict[pc]['pb_obj_id'] = []

            for obj_ind, obj_file in enumerate(obj_file_list):
                pc_master_dict[pc]['mesh_file'] = obj_file
                pc_master_dict[pc]['mesh_file_dec'] = parent_obj_file_dec_list[obj_ind] if pc == 'parent' else child_obj_file_dec_list[obj_ind]

                scale_high, scale_low = pc_master_dict[pc]['scale_hl']
                scale_default = pc_master_dict[pc]['scale_default']

                if exp_args.eval.rand_mesh_scale:
                    mesh_scale = [np.random.random() * (scale_high - scale_low) + scale_low] * 3
                else:
                    mesh_scale=[scale_default] * 3

                object_class = pc_master_dict[pc]['class']

                if object_class == 'syn_book':
                    custom_book_scale = 0.9
                    mesh_scale = [val*custom_book_scale for val in mesh_scale]
                else:
                    custom_book_scale = 1.0

                if object_class == 'syn_can':
                    downscale_mesh = trimesh.load(pc_master_dict[pc]['mesh_file'])
                    can_h = downscale_mesh.extents[2]
                    can_r = downscale_mesh.extents[0] / 2.0
                    mesh_scale = [1.0]*3
                    if can_h > req_dim_h:
                        mesh_scale[2] = req_dim_h / can_h
                    if can_r > req_dim_r:
                        mesh_scale[0] = req_dim_r / can_r
                        mesh_scale[1] = req_dim_r / can_r

                pc_master_dict[pc]['mesh_scale'] = mesh_scale

                upright_orientation = upright_orientation_dict[object_class]

                load_pose_type = pc_master_dict[pc]['load_pose_type']
                x_high, x_low = pc_master_dict[pc]['xhl']
                y_high, y_low = pc_master_dict[pc]['yhl']

                obj_obj_file, obj_obj_file_dec = pc_master_dict[pc]['mesh_file'], pc_master_dict[pc]['mesh_file_dec']

                feas_check_mesh = trimesh.load(obj_obj_file).apply_scale(mesh_scale)
                feasible_pose = False
                try_feasible_pose = 0
                while True:
                    if object_class in ['syn_bookshelf', 'syn_cabinet', 'syn_cabinet_packed_nonuniform', 'syn_cabinet_packed_uniform']:
                        xy_origin_offset = np.random.random() * (0.55 - 0.45) + 0.45
                        theta = np.random.random() * np.pi - np.pi/2

                        x = xy_origin_offset * np.cos(theta) + 0.3
                        y = xy_origin_offset * np.sin(theta)
                        z = table_z + 0.005
                        pos = [x, y, z]

                        yaw = theta + np.pi
                        bookshelf_yaw = yaw
                        rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=yaw-0.0001, max_theta=yaw+0.0001)
                        pose = util.list2pose_stamped(pos + upright_orientation)
                        pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
                        pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]

                        pos[2] = 1.0
                    else:
                        if load_pose_type == 'any_pose':

                            ori = R.random().as_quat().tolist()

                            pos = [
                                np.random.random() * (x_high - x_low) + x_low,
                                np.random.random() * (y_high - y_low) + y_low,
                                table_z + 0.02]
                            pose = pos + ori
                            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
                            pose_w_yaw = util.transform_pose(util.list2pose_stamped(pose), util.pose_from_matrix(rand_yaw_T))
                            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]
                        else:
                            pos = [np.random.random() * (x_high - x_low) + x_low, np.random.random() * (y_high - y_low) + y_low, table_z]
                            pose = util.list2pose_stamped(pos + upright_orientation)
                            rand_yaw_T = util.rand_body_yaw_transform(pos, min_theta=-np.pi, max_theta=np.pi)
                            pose_w_yaw = util.transform_pose(pose, util.pose_from_matrix(rand_yaw_T))
                            pos, ori = util.pose_stamped2list(pose_w_yaw)[:3], util.pose_stamped2list(pose_w_yaw)[3:]

                    if current_scene_concat_mesh is not None:
                        sample_mesh = feas_check_mesh.copy()
                        sample_pose = util.matrix_from_list(pos + ori)
                        sample_mesh.apply_transform(sample_pose)
                        sample_query_points = sample_mesh.sample(5000)

                        occ_values = inside_mesh.check_mesh_contains(current_scene_concat_mesh, sample_query_points)
                        occ_inds = np.where(occ_values)[0]
                        log_debug(f'Number of infeasible query points: {occ_inds.shape[0]}')
                        
                        if args.debug:
                            pause_mc_thread(True)
                            time.sleep(0.3)
                            util.meshcat_trimesh_show(mc_vis, 'scene/full_mesh', current_scene_concat_mesh)
                            util.meshcat_trimesh_show(mc_vis, 'scene/sample_mesh', sample_mesh)

                            pause_mc_thread(False)

                        if occ_inds.shape[0] == 0:
                            feasible_pose = True
                    else:
                        feasible_pose = True
                    
                    try_feasible_pose += 1
                    if try_feasible_pose > 100:
                        log_warn('Attempted to find feasible pose 100 times, something is wrong')
                        from IPython import embed; embed()

                    if feasible_pose:
                        log_debug('Found feasible pose')
                        break

                if object_class in ['syn_bookshelf', 'syn_cabinet', 'syn_cabinet_packed_nonuniform', 'syn_cabinet_packed_uniform']:
                    pass
                else:
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

                if pc == 'parent':
                    current_parent_scale_list.append(mesh_scale)
                else:
                    current_child_scale_list.append(mesh_scale)

                color = (1.0, 1.0, 1.0, 1.0) if pc == 'parent' else (0.0, 0.0, 1.0, 1.0)

                concave_urdf_classes = [
                    # 'syn_rack_easy', 
                    # 'syn_rack_med', 
                    # 'syn_rack_hard', 
                    'syn_bookshelf', 
                    'syn_cabinet', 
                    'syn_cabinet_packed_nonuniform', 
                    'syn_cabinet_packed_uniform'
                ]

                mass = 0.01 if pc == 'child' else 0.0
                if object_class in concave_urdf_classes:
                    # replace the .obj with .urdf to load our URDF with "concave" flag (better than using convex decomp for shelves/cabinets)
                    dir_to_load = '/'.join(obj_obj_file.split('/')[:-1])
                    fname_to_load = osp.join(dir_to_load, obj_obj_file.split('/')[-1] + '.urdf')
                    assert osp.exists(fname_to_load), f'URDF file: {fname_to_load} does not exist!'
                    obj_id = pb_client.load_urdf(fname_to_load, base_pos=pos, base_ori=ori) 
                    recorder.register_object(obj_id, fname_to_load)
                    log_debug(f'Loaded from URDF: {fname_to_load}')
                else:
                    obj_id = pb_client.load_geom(
                        'mesh',
                        mass=mass,
                        mesh_scale=mesh_scale,
                        visualfile=obj_obj_file_dec,
                        collifile=obj_obj_file_dec,
                        base_pos=pos,
                        base_ori=ori,
                        rgba=color)
                    recorder.register_object(obj_id, obj_obj_file_dec, scaling=mesh_scale)
                safeCollisionFilterPair(bodyUniqueIdA=obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=table_base_id, enableCollision=True)

                p.changeDynamics(obj_id, -1, lateralFriction=0.5)

                o_cid = None
                if (object_class in ['syn_rack_easy', 'syn_rack_hard', 'syn_rack_med']) or (load_pose_type == 'any_pose' and pc == 'child'):
                    o_cid = constraint_obj_world(obj_id, pos, ori)
                    pb_client.set_step_sim(False)
                pc_master_dict[pc]['o_cid'].append(o_cid)
                safeCollisionFilterPair(obj_id, table_id, -1, table_base_id, enableCollision=True)
                p.changeDynamics(obj_id, -1, linearDamping=5, angularDamping=5)
                time.sleep(1.5)

                pc_master_dict[pc]['pb_obj_id'].append(obj_id)

                if (eval_bookshelf_task or eval_cabinet_task) and pc == 'parent':
                    start_parent_pose = np.concatenate(pb_client.get_body_state(obj_id)[:2]).tolist()
                    parent_pose_mat = util.matrix_from_list(start_parent_pose)

                time.sleep(1.0)
                pause_mc_thread(True)
                time.sleep(1.0)

                recorder.add_keyframe()
                time.sleep(0.5)
                recorder.update_meshcat_current_state(mc_vis)
                time.sleep(0.5)

                current_state_info = recorder.get_formatted_current_state()
                obj_meshes = []
                obj_poses = []
                obj_scales = []
                for k, v in current_state_info.items():
                    pos = v['current_state']['position']
                    ori = v['current_state']['orientation']
                    pose = util.matrix_from_list(pos + ori)

                    mesh_path = v['mesh_path']
                    mesh_scale = v['mesh_scale'].tolist()

                    log_debug('Pose, Mesh Path, Mesh Scale: ')
                    log_debug(pose)
                    log_debug(mesh_path)
                    log_debug(mesh_scale)
                    obj_meshes.append(copy.deepcopy(mesh_path.split(path_util.get_rpdiff_data())[-1]))
                    obj_poses.append(copy.deepcopy(pose))
                    obj_scales.append(copy.deepcopy(mesh_scale))

                current_scene_concat_mesh = three_util.trimesh_combine(obj_meshes, obj_poses, obj_scales)
                # pause_mc_thread(False)

                time.sleep(1.0)
                pause_mc_thread(False)
                time.sleep(1.0)

                # turn off collisions betwen hand and everything
                for ii in range(p.getNumJoints(floating_hand.hand.hand_id)):
                    safeCollisionFilterPair(floating_hand.hand_id, obj_id, ii, -1, enableCollision=False)

        time.sleep(1.0)
        pause_mc_thread(True)
        time.sleep(1.0)

        # pause_mc_thread(True)
        time.sleep(0.5)

        # turn off collisions betwen hand and everything
        for ii in range(p.getNumJoints(floating_hand.hand.hand_id)):
            safeCollisionFilterPair(floating_hand.hand_id, table_id, ii, table_base_id, enableCollision=False)

        # get object point cloud
        depth_imgs = []
        seg_idxs = []
        obj_pcd_pts = []

        pc_obs_info = {}
        pc_obs_info['pcd'] = {}
        pc_obs_info['pcd_pts'] = {}
        pc_obs_info['pcd_pts']['parent'] = []
        pc_obs_info['pcd_pts']['child'] = [] 

        obj_pose_world = p.getBasePositionAndOrientation(obj_id)
        obj_pose_world = util.list2pose_stamped(list(obj_pose_world[0]) + list(obj_pose_world[1]))

        cam_cfg = config_util.copy_attr_dict(env_args.cameras)
        if (eval_bookshelf_task or eval_cabinet_task):
            # modify the focus point and yaw angle for the first two cameras
            bookshelf_focus_pt = parent_pose_mat[:-1, -1].tolist(); bookshelf_focus_pt[2] += 0.275 
            cam_cfg.focus_pt_set[0] = bookshelf_focus_pt 
            cam_cfg.focus_pt_set[1] = bookshelf_focus_pt 

            cam_cfg.yaw_angles[0] = np.rad2deg(bookshelf_yaw) + 90 - 25
            cam_cfg.yaw_angles[1] = np.rad2deg(bookshelf_yaw) + 90 + 25

        cams = MultiCams(cam_cfg, pb_client, n_cams=env_args.n_cameras)

        cam_info = {}
        cam_info['pose_world'] = []
        for i, cam in enumerate(cams.cams):
            cam_info['pose_world'].append(util.pose_from_matrix(cam.cam_ext_mat))
            util.meshcat_frame_show(mc_vis, f'scene/cam_pose_{i}', cam.cam_ext_mat) 
        
        if eval_cam is None:
            eval_cam = RGBDCameraPybullet(cams._camera_cfgs(), pb_client)
            eval_cam.setup_camera(
                focus_pt=[0.4, 0.0, table_z],
                dist=0.9,
                yaw=270,
                pitch=-25,
                roll=0)

        for i, cam in enumerate(cams.cams): 
            # get image and raw point cloud
            rgb, depth, seg = cam.get_images(get_rgb=True, get_depth=True, get_seg=True)
            pts_raw, _ = cam.get_pcd(in_world=True, rgb_image=rgb, depth_image=depth, depth_min=0.0, depth_max=np.inf)

            # flatten and find corresponding pixels in segmentation mask
            flat_seg = seg.flatten()
            flat_depth = depth.flatten()

            for pc in pcl:
                for obj_id in pc_master_dict[pc]['pb_obj_id']:
                    obj_inds = np.where(flat_seg == obj_id)
                    seg_depth = flat_depth[obj_inds[0]]  
                    
                    obj_pts = pts_raw[obj_inds[0], :]
                    pc_obs_info['pcd_pts'][pc].append(obj_pts)

            depth_imgs.append(seg_depth)
            seg_idxs.append(obj_inds)
        
        for pc, obj_pcd_pts in pc_obs_info['pcd_pts'].items():
            target_obj_pcd_obs = np.concatenate(obj_pcd_pts, axis=0)  # object shape point cloud
            pc_obs_info['pcd'][pc] = target_obj_pcd_obs

        current_parent_pose_list = []
        current_child_pose_list = []
        final_child_pose_list = []
        for parent_obj_id in pc_master_dict['parent']['pb_obj_id']:
            start_parent_pose = np.concatenate(pb_client.get_body_state(parent_obj_id)[:2]).tolist()
            start_parent_pose_mat = util.matrix_from_pose(util.list2pose_stamped(start_parent_pose))
            current_parent_pose_list.append(start_parent_pose)
        for child_obj_id in pc_master_dict['child']['pb_obj_id']:
            start_child_pose = np.concatenate(pb_client.get_body_state(child_obj_id)[:2]).tolist()
            start_child_pose_mat = util.matrix_from_pose(util.list2pose_stamped(start_child_pose))
            current_child_pose_list.append(start_child_pose)
        child_obj_pose_world = current_child_pose_list[0]

        parent_pcd = pc_obs_info['pcd']['parent']
        # child_pcd = pc_obs_info['pcd']['child']

        if exp_args.eval.load_full_pcd:
            log_warn(f'!!! USING FULL POINT CLOUD FROM OBJECT MESH !!!')
            # obj_obj_file, obj_obj_file_dec = pc_master_dict[pc]['mesh_file'], pc_master_dict[pc]['mesh_file_dec']
            child_pcd_orig = trimesh.load(child_obj_file_list[0], process=False).apply_scale(custom_book_scale).sample(5000)
            child_pcd = util.transform_pcd(child_pcd_orig, util.matrix_from_list(child_obj_pose_world))
        else:
            child_pcd = pc_obs_info['pcd']['child']

        if child_pcd.shape[0] < 300:
            log_warn(f'Child point cloud not enough points, shape: {child_pcd.shape[0]}. Continuing')
            for pc in pcl:
                obj_id = pc_master_dict[pc]['pb_obj_id']
                for obj_id in pc_master_dict[pc]['pb_obj_id']:
                    pb_client.remove_body(obj_id)
                    recorder.remove_object(obj_id, mc_vis)
            mc_vis['scene/real'].delete()
            mc_vis['scene/infer'].delete()
            mc_vis['scene/child_pcd_guess'].delete()
            mc_vis['scene/child_pcd_predict'].delete()
            mc_vis['scene/final_child_pcd'].delete()
            continue

        print('Got point cloud')

        with recorder.meshcat_scene_lock:
            util.meshcat_pcd_show(mc_vis, parent_pcd, color=(255, 0, 0), name='scene/parent_pcd')
            util.meshcat_pcd_show(mc_vis, child_pcd, color=(0, 0, 255), name='scene/child_pcd')
        
        guess_rot = rot_grid[np.random.randint(rot_grid.shape[0])]
        tf1 = np.eye(4); tf1[:-1, -1] = -1.0 * np.mean(child_pcd, axis=0)
        tf2 = np.eye(4)
        if exp_args.eval.init_orig_ori:
            log_warn(f'!!! USING ORIGINAL ORIENTATION AS INITIAL GUESS ORIENTATION !!!')
        else:
            log_warn(f'!!! USING RANDOM SO(3) ROTATION AS INITIAL GUESS ORIENTATION !!!')
            tf2[:-1, :-1] = guess_rot
        
        if exp_args.eval.init_parent_mean_pos:
            log_warn(f'!!! USING PARENT MEAN AS INITIAL GUESS POSITION !!!')
            tf3 = np.eye(4); tf3[:-1, -1] = np.mean(parent_pcd, axis=0) + ((np.random.random(3) - 0.5) * 0.005)
        else:
            log_warn(f'!!! USING RANDOM PARENT POINT AS INITIAL GUESS POSITION !!!')
            tf3 = np.eye(4); tf3[:-1, -1] = parent_pcd[np.random.randint(parent_pcd.shape[0])] + (np.random.random(3) * 0.01)

        relative_trans_guess = np.matmul(tf3, np.matmul(tf2, tf1))

        time.sleep(0.3)

        child_pcd_guess = util.transform_pcd(child_pcd, relative_trans_guess)

        multi_mesh_dict = dict(
            parent_file=parent_obj_file_list,
            parent_scale=current_parent_scale_list,
            parent_pose=current_parent_pose_list,
            child_file=child_obj_file_list,
            child_scale=current_child_scale_list,
            child_pose=current_child_pose_list,
            multi=True)
        
        inlier_parent_idx = np.where(np.max(parent_pcd, axis=1) < 3.0)[0]
        parent_pcd = parent_pcd[inlier_parent_idx]
        infer_kwargs['gt_child_cent'] = np.matmul(relative_trans_guess, start_child_pose_mat)[:-1, -1]
        
        # args for exporting the visualization
        infer_kwargs['export_viz'] = args.export_viz
        infer_kwargs['export_viz_dirname'] = args.export_viz_dirname
        infer_kwargs['export_viz_relative_trans_guess'] = relative_trans_guess

        # args for computing + exporting coverage metrics
        infer_kwargs['compute_coverage_scores'] = args.compute_coverage
        infer_kwargs['out_coverage_dirname1'] = args.out_coverage_dirname + f'_{parent_class}_{child_class}'
        infer_kwargs['out_coverage_dirname2'] = osp.join(eval_save_dir, args.out_coverage_dirname + f'_{parent_class}_{child_class}')

        infer_kwargs['iteration'] = iteration

        # refine
        relative_trans_pred = infer_relation_policy(
            mc_vis, 
            parent_pcd, child_pcd_guess, 
            coarse_aff_model,
            pr_model, 
            success_model,
            scene_mean=args.data.coarse_aff.scene_mean, scene_scale=args.data.coarse_aff.scene_scale, 
            grid_pts=raster_pts, rot_grid=rot_grid, 
            viz=False, n_iters=exp_args.eval.n_refine_iters, 
            no_parent_crop=(not exp_args.parent_crop),
            return_top=(not exp_args.eval.return_rand), with_coll=exp_args.eval.with_coll, 
            run_affordance=exp_args.eval.run_affordance, init_k_val=exp_args.eval.init_k_val,
            no_sc_score=exp_args.eval.no_success_classifier, 
            init_parent_mean=exp_args.eval.init_parent_mean_pos, init_orig_ori=exp_args.eval.init_orig_ori,
            refine_anneal=exp_args.eval.refine_anneal,
            mesh_dict=multi_mesh_dict,
            add_per_iter_noise=exp_args.eval.add_per_iter_noise,
            per_iter_noise_kwargs=exp_args.eval.per_iter_noise_kwargs,
            variable_size_crop=exp_args.eval.variable_size_crop,
            timestep_emb_decay_factor=exp_args.eval.timestep_emb_decay_factor,
            **infer_kwargs)

        transformed_child1 = util.transform_pcd(child_pcd_guess, relative_trans_pred)

        with recorder.meshcat_scene_lock:
            util.meshcat_pcd_show(mc_vis, child_pcd_guess, color=[255, 0, 0], name='scene/child_pcd_guess')
            util.meshcat_pcd_show(mc_vis, transformed_child1, color=[255, 255, 0], name='scene/child_pcd_predict')

        relative_trans = np.matmul(relative_trans_pred, relative_trans_guess)

        child_obj_id = pc_master_dict['child']['pb_obj_id'][0]
        start_child_pose = np.concatenate(pb_client.get_body_state(child_obj_id)[:2]).tolist()
        start_child_pose_mat = util.matrix_from_pose(util.list2pose_stamped(start_child_pose))
        final_child_pose_mat = np.matmul(relative_trans, start_child_pose_mat)

        time.sleep(1.0)
        
        delta_pc_final_list = []
        for parent_pose in current_parent_pose_list:
            delta_pc_final = np.linalg.norm(np.array(parent_pose[:3]) - final_child_pose_mat[:-1, -1])
            delta_pc_final_list.append(delta_pc_final)

        place_parent_idx = np.argmin(delta_pc_final_list)
        print(f'place_parent_idx: {place_parent_idx}')
        print(f'pb_obj_id list: {pc_master_dict["parent"]["pb_obj_id"]}')
        try:
            place_parent_obj_id = pc_master_dict['parent']['pb_obj_id'][place_parent_idx]
        except IndexError as e:
            print(f'IndexError: {e}')
            print('here with place_parent_idx')
            from IPython import embed; embed()

        start_parent_pose = np.concatenate(pb_client.get_body_state(place_parent_obj_id)[:2]).tolist()
        start_parent_pose_mat = util.matrix_from_pose(util.list2pose_stamped(start_parent_pose))
        upright_orientation = upright_orientation_dict[pc_master_dict['parent']['class']]
        upright_parent_ori_mat = R.from_quat(upright_orientation).as_matrix()

        pause_mc_thread(False)

        pb_client.set_step_sim(True)
        if pc_master_dict['parent']['load_pose_type'] == 'any_pose':
            # get the relative transformation to make it upright
            upright_parent_pose_mat = copy.deepcopy(start_parent_pose_mat); upright_parent_pose_mat[:-1, :-1] = upright_parent_ori_mat
            relative_upright_pose_mat = np.matmul(upright_parent_pose_mat, np.linalg.inv(start_parent_pose_mat))

            upright_parent_pos, upright_parent_ori = start_parent_pose[:3], R.from_matrix(upright_parent_ori_mat).as_quat()
            pb_client.reset_body(parent_obj_id, upright_parent_pos, upright_parent_ori)

            final_child_pose_mat = np.matmul(relative_upright_pose_mat, final_child_pose_mat)

        final_child_pose_list = util.pose_stamped2list(util.pose_from_matrix(final_child_pose_mat))
        final_child_pos, final_child_ori = final_child_pose_list[:3], final_child_pose_list[3:]

        # reset object into the initial position (pre-approach)
        pb_client.reset_body(child_obj_id, final_child_pos, final_child_ori)

        # take an image to make sure it's good
        teleport_rgb1 = cams.cams[0].get_images(get_rgb=True)[0]
        teleport_rgb2 = cams.cams[1].get_images(get_rgb=True)[0]
        teleport_img_fname1 = osp.join(eval_teleport_imgs_dir, f'{iteration}_0.png')
        teleport_img_fname2 = osp.join(eval_teleport_imgs_dir, f'{iteration}_1.png')
        util.np2img(teleport_rgb1.astype(np.uint8), teleport_img_fname1)
        util.np2img(teleport_rgb2.astype(np.uint8), teleport_img_fname2)

        ########################################################################################
        # use the virtual hand to move the child object into place
        
        if (eval_bookshelf_task or eval_cabinet_task):
            # use either known info or prediction approach to obtain approach vector
            final_delta_pos = np.array([np.cos(bookshelf_yaw), np.sin(bookshelf_yaw), 0.0])
            final_child_pre_pos = np.array(final_child_pos) + 0.35 * final_delta_pos

            # reset object into the initial position (pre-approach)
            pb_client.reset_body(child_obj_id, final_child_pre_pos, final_child_ori)
            
        if eval_mug_rack_multi_task and exp_args.use_floating_hand_execution: # and False:
            # use either knwon info or prediction approach to obtain approach vector
            final_delta_pos = None

            mug_pose_mat = final_child_pose_mat
            offset_axis_pre_rot = mug_pose_mat[:-1, 0]  # mug x-axis, don't know forward/backward
            pt_near_handle = final_child_pose_mat[:-1, -1] + (-1.0 * mug_pose_mat[:-1, 2] * 0.075)
            rack_to_handle_vec = pt_near_handle - start_parent_pose_mat[:-1, -1]
            rack_to_handle_vec = rack_to_handle_vec / np.linalg.norm(rack_to_handle_vec)
            if np.dot(offset_axis_pre_rot, rack_to_handle_vec) < 0.0:
                offset_axis_pre_rot = -1.0 * offset_axis_pre_rot
            
            # if z-component is negative, we're probably on the wrong side
            if offset_axis_pre_rot[2] < 0:
                offset_axis_pre_rot = -1.0 * offset_axis_pre_rot

            offset_axis_post_rot = offset_axis_pre_rot

            final_delta_pos = offset_axis_post_rot
            final_child_pre_pos = np.array(final_child_pos) + 0.35 * final_delta_pos

            # reset object into the initial position (pre-approach)
            pb_client.reset_body(child_obj_id, final_child_pre_pos, final_child_ori)

        ########################################################################################

        final_child_pcd = util.transform_pcd(pc_obs_info['pcd']['child'], relative_trans)
        with recorder.meshcat_scene_lock:
            util.meshcat_pcd_show(mc_vis, final_child_pcd, color=[255, 0, 255], name='scene/final_child_pcd')

        safeCollisionFilterPair(child_obj_id, table_id, -1, table_base_id, enableCollision=False)

        time.sleep(3.0)

        ########################################################################################
        # use the virtual hand to move the child object into place

        success_crit_dict = {}
        kvs = {}
        
        if (eval_bookshelf_task or eval_cabinet_task or eval_mug_rack_multi_task) and exp_args.use_floating_hand_execution:

            safeRemoveConstraint(pc_master_dict['child']['o_cid'][0])

            current_parent_pos, current_parent_ori = p.getBasePositionAndOrientation(place_parent_obj_id)
            current_parent_cid = constraint_obj_world(place_parent_obj_id, current_parent_pos, current_parent_ori)
            current_child_pos, current_child_ori = p.getBasePositionAndOrientation(child_obj_id)
            current_child_cid = constraint_obj_world(child_obj_id, current_child_pos, current_child_ori)

            pb_client.set_step_sim(False)

            # evaluation criteria
            time.sleep(1.0)

            # move virtual hand into position and move until touch

            floating_hand.hand.set_ee_pose(final_child_pre_pos)
            floating_hand.hand.constraint_grasp_close(child_obj_id)

            if eval_bookshelf_task or eval_cabinet_task:
                safeRemoveConstraint(pc_master_dict['parent']['o_cid'][0])

            safeRemoveConstraint(pc_master_dict['child']['o_cid'][0])
            safeRemoveConstraint(current_child_cid)
            safeCollisionFilterPair(bodyUniqueIdA=child_obj_id, bodyUniqueIdB=table_id, linkIndexA=-1, linkIndexB=table_base_id, enableCollision=False)

            if eval_mug_rack_multi_task:
                for p_id in pc_master_dict['parent']['pb_obj_id']:
                    if p_id == place_parent_obj_id:
                        continue
                    safeCollisionFilterPair(bodyUniqueIdA=child_obj_id, bodyUniqueIdB=p_id, linkIndexA=-1, linkIndexB=-1, enableCollision=False)

            hand_delta_pos = np.array(final_child_pos) - np.array(final_child_pre_pos)
            hand_delta_dist = np.linalg.norm(hand_delta_pos)

            exec_attempts = 0
            unit_hand_delta_pos = hand_delta_pos / np.linalg.norm(hand_delta_pos)
            hand_delta_pos_to_exec = unit_hand_delta_pos * hand_delta_dist
            while True:
                out_move_until_touch = floating_hand.hand.move_ee_xyz_until_touch(
                    hand_delta_pos_to_exec,
                    eef_step=0.001, 
                    coll_id_pairs=(
                        (place_parent_obj_id, -1), 
                        (child_obj_id, -1)), 
                    use_force=True, 
                    force_thresh=15.0,
                    return_stop_motion=True)

                exec_current_child_pos = p.getBasePositionAndOrientation(child_obj_id)[0]
                delta_final = np.linalg.norm(np.asarray(final_child_pos).reshape(3,) - np.asarray(exec_current_child_pos).reshape(3,), axis=-1)
                print(f'Delta from final: {delta_final}')

                if delta_final < 0.02:
                    break
                
                # back off, and move orthogonal to approach
                hand_ee_pos = floating_hand.hand.get_ee_pose()[0]
                if exec_attempts == 0:
                    base_hand_ee_pos = hand_ee_pos
                back_hand_ee_pos = hand_ee_pos + 0.0075 * final_delta_pos
                floating_hand.hand.set_ee_pose(pos=back_hand_ee_pos)
                rnd_delta_pos = 0.0075 * util.sample_orthogonal_vector(final_delta_pos)

                floating_hand.hand.set_ee_pose(pos=base_hand_ee_pos + rnd_delta_pos)
                
                # random rotation in body frame
                rnd_euler = np.deg2rad(5) * (np.random.random(3) - 0.5)
                rnd_mat = R.from_euler('xyz', rnd_euler).as_matrix()
                hand_ee_mat = floating_hand.hand.get_ee_pose()[2]
                new_hand_ee_mat = np.matmul(hand_ee_mat, rnd_mat)
                new_hand_ee_quat = R.from_matrix(new_hand_ee_mat).as_quat()
                floating_hand.hand.set_ee_pose(ori=new_hand_ee_quat)

                hand_delta_pos_to_exec = unit_hand_delta_pos * delta_final

                if eval_cabinet_task:
                    break

                if exec_attempts >= 10:
                    break

                exec_attempts += 1

            if eval_cabinet_task:
                stop_motion_touch = out_move_until_touch[1]
                success_crit_dict['collision_free'] = not stop_motion_touch

            safeCollisionFilterPair(bodyUniqueIdA=child_obj_id, bodyUniqueIdB=place_parent_obj_id, linkIndexA=-1, linkIndexB=-1, enableCollision=True)
            floating_hand.hand.constraint_grasp_open()
            time.sleep(1.0)

            # floating_hand.hand.set_jpos([0]*6)
            floating_hand.hand.set_jpos([0]*3)
        else:
            pb_client.set_step_sim(True)
            pb_client.reset_body(child_obj_id, final_child_pos, final_child_ori)
            for o_cid in pc_master_dict['child']['o_cid']:
                safeRemoveConstraint(o_cid)
            safeCollisionFilterPair(child_obj_id, table_id, -1, -1, enableCollision=False)
            safeCollisionFilterPair(child_obj_id, table_id, -1, table_base_id, enableCollision=False)
            pb_client.set_step_sim(False)
            time.sleep(1.0)

        ########################################################################################

        obj_surf_contacts = p.getContactPoints(child_obj_id, place_parent_obj_id, -1, -1)
        touching_surf = len(obj_surf_contacts) > 0
        success_crit_dict['touching_surf'] = touching_surf

        if (parent_class == 'box_container' and child_class == 'bottle') or \
            (parent_class == 'mug' and child_class == 'bowl') or \
            ('syn_cabinet' in parent_class and child_class == 'syn_can') or \
            (parent_class == 'syn_cabinet' and child_class == 'syn_can'):
            child_final_pose = np.concatenate(p.getBasePositionAndOrientation(child_obj_id)[:2]).tolist()

            # get the y-axis in the body frame
            if child_class == 'syn_can':
                child_body_z = R.from_quat(child_final_pose[3:]).as_matrix()[:, 2]
                child_body_z = child_body_z / np.linalg.norm(child_body_z)

                # get the angle deviation from the vertical
                angle_from_upright = util.angle_from_3d_vectors(child_body_z, np.array([0, 0, 1]))
                if angle_from_upright > np.deg2rad(90):
                    angle_from_upright = np.abs(angle_from_upright - np.deg2rad(180))
            else:
                child_body_y = R.from_quat(child_final_pose[3:]).as_matrix()[:, 1]
                child_body_y = child_body_y / np.linalg.norm(child_body_y)

                # get the angle deviation from the vertical
                angle_from_upright = util.angle_from_3d_vectors(child_body_y, np.array([0, 0, 1]))
            child_upright = angle_from_upright < np.deg2rad(exp_args.upright_ori_diff_thresh_deg)

            success_crit_dict['child_upright'] = child_upright

        eval_rgb1 = cams.cams[0].get_images(get_rgb=True)[0]
        eval_rgb2 = cams.cams[1].get_images(get_rgb=True)[0]
        eval_img_fname1 = osp.join(eval_imgs_dir, f'{iteration}_0.png')
        eval_img_fname2 = osp.join(eval_imgs_dir, f'{iteration}_1.png')
        util.np2img(eval_rgb1.astype(np.uint8), eval_img_fname1)
        util.np2img(eval_rgb2.astype(np.uint8), eval_img_fname2)

        if not (eval_bookshelf_task or eval_cabinet_task):
            if len(cams.cams) > 3:
                eval_rgb3 = cams.cams[2].get_images(get_rgb=True)[0]
                eval_rgb4 = cams.cams[3].get_images(get_rgb=True)[0]
                eval_img_fname3 = osp.join(eval_imgs_dir, f'{iteration}_2.png')
                eval_img_fname4 = osp.join(eval_imgs_dir, f'{iteration}_3.png')
                util.np2img(eval_rgb3.astype(np.uint8), eval_img_fname3)
                util.np2img(eval_rgb4.astype(np.uint8), eval_img_fname4)

        # ##########################################################################
        if not exp_args.use_floating_hand_execution:
            # upside down check for too much inter-penetration
            pb_client.set_step_sim(True)

            safeRemoveConstraint(pc_master_dict['parent']['o_cid'][place_parent_idx])
            for o_cid in pc_master_dict['child']['o_cid']:
                safeRemoveConstraint(o_cid)
            
            # first, reset everything
            pb_client.reset_body(place_parent_obj_id, start_parent_pose[:3], start_parent_pose[3:])
            pb_client.reset_body(child_obj_id, start_child_pose[:3], start_child_pose[3:])

            # then, compute a new position + orientation for the parent object, that is upside down
            upside_down_ori_mat = np.matmul(R.from_euler('xyz', [np.pi, 0, 0]).as_matrix(), upright_parent_ori_mat)
            upside_down_pose_mat = np.eye(4); upside_down_pose_mat[:-1, :-1] = upside_down_ori_mat; upside_down_pose_mat[:-1, -1] = start_parent_pose[:3]
            upside_down_pose_mat[2, -1] += 0.15  # move up in z a bit
            parent_upside_down_pose_list = util.pose_stamped2list(util.pose_from_matrix(upside_down_pose_mat))

            # reset parent to this state and constrain to world
            pb_client.reset_body(place_parent_obj_id, parent_upside_down_pose_list[:3], parent_upside_down_pose_list[3:]) 
            ud_cid = constraint_obj_world(place_parent_obj_id, parent_upside_down_pose_list[:3], parent_upside_down_pose_list[3:]) 

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
        
        place_success_list.append(place_success)
        log_str = 'Iteration: %d, ' % iteration

        kvs['Place Success'] = sum(place_success_list) / float(len(place_success_list))

        if parent_class == 'box_container' and child_class == 'bottle':
            kvs['Angle From Upright'] = angle_from_upright

        for k, v in kvs.items():
            log_str += '%s: %.3f, ' % (k, v)
        for k, v in success_crit_dict.items():
            log_str += '%s: %s, ' % (k, v)

        id_str = f', parent_id: {parent_id_list}, child_id: {child_id_list}'
        log_info(log_str + id_str)

        eval_iter_dir = osp.join(eval_save_dir, f'trial_{iteration}')
        util.safe_makedirs(eval_iter_dir)
        sample_fname = osp.join(eval_iter_dir, 'success_rate_relation.npz')
        full_cfg_fname = osp.join(eval_iter_dir, 'full_config.json')
        results_txt_fname = osp.join(eval_iter_dir, 'results.txt')
        np.savez(
            sample_fname,
            parent_id=parent_id_list,
            place_parent_id=place_parent_idx,
            child_id=child_id_list,
            is_parent_shapenet_obj=is_parent_shapenet_obj,
            is_child_shapenet_obj=is_child_shapenet_obj,
            success_criteria_dict=success_crit_dict,
            place_success=place_success,
            place_success_list=place_success_list,
            parent_mesh_file=parent_obj_file_list,
            child_mesh_file=child_obj_file_list,
            args=config_util.recursive_dict(args)
        )
        json.dump(full_cfg_dict, open(full_cfg_fname, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

        results_txt_dict = {}
        results_txt_dict['place_success'] = place_success
        results_txt_dict['place_success_list'] = place_success_list
        results_txt_dict['current_success_rate'] = sum(place_success_list) / float(len(place_success_list))
        results_txt_dict['success_criteria_dict'] = success_crit_dict
        open(results_txt_fname, 'w').write(str(results_txt_dict))

        time.sleep(0.5)
        pause_mc_thread(True)
        time.sleep(0.5)

        for pc in pcl:
            obj_id = pc_master_dict[pc]['pb_obj_id']
            for obj_id in pc_master_dict[pc]['pb_obj_id']:
                pb_client.remove_body(obj_id)
                recorder.remove_object(obj_id, mc_vis)
        mc_vis['scene/real'].delete()
        mc_vis['scene/infer'].delete()
        mc_vis['scene/compute_coverage'].delete()
        mc_vis['scene/child_pcd_guess'].delete()
        mc_vis['scene/child_pcd_predict'].delete()
        mc_vis['scene/final_child_pcd'].delete()

        time.sleep(0.5)
        pause_mc_thread(False)
        time.sleep(0.5)


if __name__ == "__main__":
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_fname', type=str, required=True, help='Name of config file')
    parser.add_argument('-d', '--debug', action='store_true', help='If True, run in debug mode')
    parser.add_argument('-dd', '--debug_data', action='store_true', help='If True, run data loader in debug mode')
    parser.add_argument('-p', '--port_vis', type=int, default=6000, help='Port for ZMQ url (meshcat visualization)')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-l', '--local_dataset_dir', type=str, default=None, help='If the data is saved on a local drive, pass the root of that directory here')
    parser.add_argument('-ex', '--export_viz', action='store_true', help='If True, save data for post-processed visualization')
    parser.add_argument('--export_viz_dirname', type=str, default='rpdiff_export_viz')
    parser.add_argument('-cc', '--compute_coverage', action='store_true', help='If True, save data for post-processed visualization')
    parser.add_argument('--out_coverage_dirname', type=str, default='rpdiff_coverage_out')
    parser.add_argument('-nm', '--new_meshcat', action='store_true')

    args = parser.parse_args()

    eval_args = config_util.load_config(osp.join(path_util.get_eval_config_dir(), args.config_fname), demo_train_eval='eval')
    
    eval_args['debug'] = args.debug
    eval_args['debug_data'] = args.debug_data
    eval_args['port_vis'] = args.port_vis
    eval_args['seed'] = args.seed
    eval_args['local_dataset_dir'] = args.local_dataset_dir
    
    # other runtime options
    # if we want to export for post-process visualization
    eval_args['export_viz'] = args.export_viz
    eval_args['export_viz_dirname'] = args.export_viz_dirname

    # if we want to compute coverage metrics
    eval_args['compute_coverage'] = args.compute_coverage
    eval_args['out_coverage_dirname'] = args.out_coverage_dirname
    
    # if we want to override the port setting for meshcat and directly start our own
    eval_args['new_meshcat'] = args.new_meshcat

    eval_args = config_util.recursive_attr_dict(eval_args)

    main(eval_args)
