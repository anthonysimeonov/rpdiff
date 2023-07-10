import os.path as osp
import os
import time
import torch
import torch.nn as nn
import numpy as np
import argparse
import random
import copy
import datetime
import shutil
import json
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import meshcat

from rpdiff.utils import util, config_util, path_util
from rpdiff.utils.torch_util import dict_to_gpu
from rpdiff.utils.mesh_util import three_util
# from rpdiff.training.policy_local import dataio_chunked as dataio, losses
from rpdiff.training import dataio_full_chunked as dataio, losses

from rpdiff.model.transformer.policy import (
    NSMTransformerSingleTransformationRegression, 
    NSMTransformerSingleTransformationRegressionCVAE,
    NSMTransformerSingleSuccessClassifier)

# from rpdiff.training.policy_full.train_util import adjust_learning_rate
from rpdiff.model.coarse_affordance import CoarseAffordanceVoxelRot
from rpdiff.training.train_loops import (
    train_iter_coarse_aff, train_iter_refine_pose, train_iter_success)
from rpdiff.training.pred_util import (
    coarse_aff_to_refine_pose, coarse_aff_to_success, refine_pose_to_success, 
    coarse_aff_to_coarse_aff, refine_pose_to_refine_pose
    )

from typing import List, Tuple, Union, Callable
from torch.optim.optimizer import Optimizer


MC_SIZE = 0.005


def train(
        mc_vis: meshcat.Visualizer, 
        coarse_aff_model: nn.Module, refine_pose_model: nn.Module, success_model: nn.Module, 
        aff_optimizer: Optimizer, pr_optimizer: Optimizer, sc_optimizer: Optimizer,
        train_dataloader: DataLoader, test_dataloader: DataLoader, 
        aff_loss_fn: Callable, pr_loss_fn: Callable, sc_loss_fn: Callable,
        dev: torch.device, 
        logger: SummaryWriter, 
        logdir: str, 
        args: config_util.AttrDict,
        start_iter: int=0,
        **kwargs):

    coarse_aff_model.train()
    refine_pose_model.train()
    success_model.train()

    offset = np.array([0, 0.2, 0.0])
    bs = args.experiment.batch_size
    it = start_iter

    voxel_grid_pts = torch.from_numpy(train_dataloader.dataset.raster_pts).float().cuda()
    rot_mat_grid = torch.from_numpy(train_dataloader.dataset.rot_grid).float().cuda()

    args.experiment.dataset_length = len(train_dataloader.dataset)

    if 'coarse_aff_model2' in kwargs:
        coarse_aff_model2 = kwargs['coarse_aff_model2']['model']
        aff_optimizer2 = kwargs['coarse_aff_model2']['opt']
        aff_loss_fn2 = kwargs['coarse_aff_model2']['loss_fn']
        reso_grid2 = kwargs['coarse_aff_model2']['reso_grid']
        padding2 = kwargs['coarse_aff_model2']['padding']

        voxel_grid_pts2 = three_util.get_raster_points(reso_grid2, padding=padding2)

        # reshape to grid, and swap axes (permute x and z), B x reso x reso x reso x 3
        voxel_grid_pts2 = voxel_grid_pts2.reshape(reso_grid2, reso_grid2, reso_grid2, 3)
        voxel_grid_pts2 = voxel_grid_pts2.transpose(2, 1, 0, 3)

        # reshape back to B x N x 3
        voxel_grid_pts2 = torch.from_numpy(voxel_grid_pts2.reshape(-1, 3)).float().cuda()

        coarse_aff_model2 = coarse_aff_model2.train()

    while True:

        if it > args.experiment.num_iterations:
            break
        
        if args.debug_data:
            # sample = train_dataloader.dataset[0]
            sample = train_dataloader.dataset[371]
            # sample = train_dataloader.dataset[1963]
            print('[Debug Data] Here with sample')

            # for i in range(len(train_dataloader.dataset)):
            #     sample = train_dataloader.dataset[i]
            #     if 'parent_start_pcd' not in sample[1][0].keys():
            #         print(f'[Debug Data] Here with bad sample (index: {i})')
            #         from IPython import embed; embed()

            from IPython import embed; embed()
        for sample in train_dataloader:
            it += 1
            current_epoch = it * bs / len(train_dataloader.dataset)

            start_time = time.time()

            coarse_aff_sample, refine_pose_sample, success_sample = sample
            coarse_aff_mi, coarse_aff_gt = coarse_aff_sample
            refine_pose_mi, refine_pose_gt = refine_pose_sample
            success_mi, success_gt = success_sample
            
            coarse_aff_out = None
            refine_pose_out = None
            success_out = None

            loss_dict = {}

            if args.experiment.train.train_coarse_aff and (len(coarse_aff_mi) > 0):

                # prepare input and gt
                coarse_aff_mi = dict_to_gpu(coarse_aff_mi)
                coarse_aff_gt = dict_to_gpu(coarse_aff_gt)

                coarse_aff_out = train_iter_coarse_aff(
                    coarse_aff_mi,
                    coarse_aff_gt,
                    coarse_aff_model,
                    aff_optimizer,
                    aff_loss_fn,
                    args,
                    voxel_grid_pts, args.data.voxel_grid.reso_grid,
                    rot_mat_grid, args.data.rot_grid_bins,
                    it, current_epoch,
                    logger, 
                    mc_vis=mc_vis)

                # process output for logging
                for k, v in coarse_aff_out['loss'].items():
                    loss_dict[k] = v

                if args.experiment.train.coarse_aff_from_coarse_pred:
                    # process coarse prediction and refine input to create new refine input
                    
                    if args.model.coarse_aff.multi_model:
                        aff_refine_model = coarse_aff_model2
                        aff_refine_opt = aff_optimizer2
                        aff_refine_loss_fn = aff_loss_fn2
                        aff_refine_voxel_grid_pts = voxel_grid_pts2
                        aff_refine_reso_grid = reso_grid2
                    else:
                        aff_refine_model = coarse_aff_model
                        aff_refine_opt = aff_optimizer
                        aff_refine_loss_fn = aff_loss_fn
                        aff_refine_voxel_grid_pts = voxel_grid_pts
                        aff_refine_reso_grid = args.data.voxel_grid.reso_grid

                    coarse_aff_mi, coarse_aff_gt = coarse_aff_to_coarse_aff(
                        coarse_aff_mi, coarse_aff_out['model_output'], coarse_aff_gt,
                        rot_mat_grid, voxel_grid_pts, aff_refine_voxel_grid_pts, aff_refine_reso_grid, args, mc_vis=mc_vis)

                    coarse_aff_mi = dict_to_gpu(coarse_aff_mi)
                    coarse_aff_gt = dict_to_gpu(coarse_aff_gt)

                    # torch.cuda.empty_cache()

                    coarse_aff_out = train_iter_coarse_aff(
                        coarse_aff_mi,
                        coarse_aff_gt,
                        aff_refine_model,
                        aff_refine_opt,
                        aff_refine_loss_fn,
                        args,
                        aff_refine_voxel_grid_pts, aff_refine_reso_grid,
                        rot_mat_grid, args.data.rot_grid_bins,
                        it, current_epoch,
                        logger, refine_pred=True,
                        mc_vis=mc_vis)

                if args.experiment.train.refine_pose_from_coarse_pred:
                    # process coarse prediction and refine input to create new refine input
                    refine_pose_mi, refine_pose_gt = coarse_aff_to_refine_pose(
                        coarse_aff_mi, coarse_aff_out['model_output'], coarse_aff_gt,
                        refine_pose_mi, refine_pose_gt, 
                        rot_mat_grid, voxel_grid_pts, args, mc_vis=mc_vis)
                        

                if args.experiment.train.success_from_coarse_pred:
                    # process coarse prediction and success input to create new success input
                    success_mi, success_gt = coarse_aff_to_success(
                            coarse_aff_mi, coarse_aff_out['model_output'], coarse_aff_gt, 
                            success_mi,  success_gt, 
                            rot_mat_grid, voxel_grid_pts, args, mc_vis=mc_vis)

            if args.experiment.train.train_refine_pose and (len(refine_pose_mi) > 0):

                # prepare input and gt
                refine_pose_mi = dict_to_gpu(refine_pose_mi)
                refine_pose_gt = dict_to_gpu(refine_pose_gt)

                refine_pose_out = train_iter_refine_pose(
                    refine_pose_mi,
                    refine_pose_gt,
                    refine_pose_model,
                    pr_optimizer,
                    pr_loss_fn,
                    args,
                    it, current_epoch,
                    logger,
                    mc_vis=mc_vis)
                
                # process output for logging
                for k, v in refine_pose_out['loss'].items():
                    loss_dict[k] = v

                if args.experiment.train.success_from_refine_pred:
                    # process coarse prediction and success input to create new success input
                    success_mi, success_gt = refine_pose_to_success(
                            refine_pose_mi, refine_pose_out['model_output'], refine_pose_gt, 
                            success_mi, success_gt, 
                            args, mc_vis=mc_vis) 
                
            if args.experiment.train.train_success and (len(success_mi) > 0):
                if args.experiment.train.success_from_refine_pred and (len(refine_pose_mi) == 0):
                    print(f'Skipping success due to no refine pose model input')
                    continue

                # prepare input and gt
                success_mi = dict_to_gpu(success_mi)
                success_gt = dict_to_gpu(success_gt)

                success_out = train_iter_success(
                    success_mi,
                    success_gt,
                    success_model,
                    sc_optimizer,
                    sc_loss_fn,
                    args,
                    it, current_epoch,
                    logger,
                    mc_vis=mc_vis)

                # process output for logging
                for k, v in success_out['loss'].items():
                    loss_dict[k] = v

            #######################################################################
            # Logging and checkpoints

            if it % args.experiment.log_interval == 0 and args.experiment.train.out_log_full:
                string = f'Iteration {it} -- '

                for loss_name, loss_val in loss_dict.items():
                    if isinstance(loss_val, dict):
                        # don't let these loss dicts get more than two levels deep
                        for k, v in loss_val.items():
                            string += f'{k}: {v.mean().item():.6f} '
                            logger.add_scalar(k, v.mean().item(), it)
                    else:
                        string += f'{loss_name}: {loss_val.mean().item():.6f} '
                        logger.add_scalar(loss_name, loss_val.mean().item(), it)

                if args.experiment.debug:
                    from IPython import embed; embed()

                end_time = time.time()
                total_duration = end_time - start_time

                string += f'duration: {total_duration:.4f}'

                print(string)

            if it % args.experiment.save_interval == 0 and it > 0:
                model_path = osp.join(logdir, f'model_{it}.pth')
                model_path_latest = osp.join(logdir, 'model_latest.pth')

                ckpt = {'args': config_util.recursive_dict(args)}
                
                ckpt['coarse_aff_model_state_dict'] = coarse_aff_model.state_dict()
                ckpt['refine_pose_model_state_dict'] = refine_pose_model.state_dict()
                ckpt['success_model_state_dict'] = success_model.state_dict()

                ckpt['aff_optimizer_state_dict'] = aff_optimizer.state_dict()
                ckpt['pr_optimizer_state_dict'] = pr_optimizer.state_dict()
                ckpt['sc_optimizer_state_dict'] = sc_optimizer.state_dict()

                if args.model.coarse_aff.multi_model:
                    ckpt['coarse_aff_model_state_dict2'] = coarse_aff_model2.state_dict()
                    ckpt['aff_optimizer_state_dict2'] = aff_optimizer2.state_dict()
                
                torch.save(ckpt, model_path)
                torch.save(ckpt, model_path_latest)


def main(args: config_util.AttrDict):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ##############################################
    # Setup basic experiment params

    logdir = osp.join(
        path_util.get_rpdiff_model_weights(), 
        args.experiment.logdir, 
        args.experiment.experiment_name)
    util.safe_makedirs(logdir)

    # Set up experiment run/config logging
    nowstr = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_logs = osp.join(logdir, 'run_logs')
    util.safe_makedirs(run_logs)
    run_log_folder = osp.join(run_logs, nowstr)
    util.safe_makedirs(run_log_folder)

    # copy everything we would like to know about this run in the run log folder
    for fn in os.listdir(os.getcwd()):
        if not (fn.endswith('.py') or fn.endswith('.sh') or fn.endswith('.bash')):
            continue
        log_fn = osp.join(run_log_folder, fn)
        shutil.copy(fn, log_fn) 

    full_cfg_dict = copy.deepcopy(config_util.recursive_dict(args))
    full_cfg_fname = osp.join(run_log_folder, 'full_exp_cfg.txt')
    json.dump(full_cfg_dict, open(full_cfg_fname, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

    if args.experiment.meshcat_on and args.meshcat_ap:
        zmq_url=f'tcp://127.0.0.1:{args.port_vis}'
        mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
        mc_vis['scene'].delete()
    else:
        mc_vis = None

    # prepare dictionary for extra kwargs in train function
    train_kwargs = {}

    ##############################################
    # Prepare dataset and dataloader

    data_args = args.data
    if osp.exists(str(args.local_dataset_dir)):
        dataset_path = osp.join(
            args.local_dataset_dir, 
            data_args.data_root,
            data_args.dataset_path)
    else:
        dataset_path = osp.join(
            path_util.get_rpdiff_data(), 
            data_args.data_root,
            data_args.dataset_path)

    assert osp.exists(dataset_path), f'Dataset path: {dataset_path} does not exist'
    
    train_dataset = dataio.FullRelationPointcloudPolicyDataset(
        dataset_path, 
        data_args,
        phase='train', 
        train_coarse_aff=args.experiment.train.train_coarse_aff,
        train_refine_pose=args.experiment.train.train_refine_pose,
        train_success=args.experiment.train.train_success,
        mc_vis=mc_vis, 
        debug_viz=args.debug_data)
    val_dataset = dataio.FullRelationPointcloudPolicyDataset(
        dataset_path, 
        data_args,
        phase='val', 
        train_coarse_aff=args.experiment.train.train_coarse_aff,
        train_refine_pose=args.experiment.train.train_refine_pose,
        train_success=args.experiment.train.train_success,
        mc_vis=mc_vis,
        debug_viz=args.debug_data)

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.experiment.batch_size, 
        shuffle=True, 
        num_workers=args.experiment.num_train_workers, 
        drop_last=True)
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=2, 
        num_workers=1,
        shuffle=False, 
        drop_last=True)

    # grab some things we need for training
    args.experiment.epochs = args.experiment.num_iterations / len(train_dataloader) 
    args.data.rot_grid_bins = train_dataset.rot_grid.shape[0]

    ##############################################
    # Prepare networks

    ###
    # Load pose refinement model, loss, and optimizer
    ###

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

    pr_model_params = pr_model.parameters()

    # loss
    pr_loss_type = args.loss.refine_pose.type
    assert pr_loss_type in args.loss.refine_pose.valid_losses, f'Loss type: {pr_loss_type} not in {args.loss.refine_pose.valid_losses}'

    if pr_loss_type == 'tf_chamfer':
        tfc_mqa_wrapper = losses.TransformChamferWrapper(
            l1=args.loss.tf_chamfer.l1,
            trans_offset=args.loss.tf_chamfer.trans_offset)
        pr_loss_fn = tfc_mqa_wrapper.tf_chamfer
    elif pr_loss_type == 'tf_chamfer_w_kldiv':
        tfc_mqa_wrapper = losses.TransformChamferWrapper(
            l1=args.loss.tf_chamfer.l1,
            trans_offset=args.loss.tf_chamfer.trans_offset,
            kl_div=True)
        pr_loss_fn = tfc_mqa_wrapper.tf_chamfer_w_kldiv
    else:
        raise ValueError(f'Unrecognized: {pr_loss_fn}')

    # optimizer
    pr_opt_type = args.optimizer.refine_pose.type
    assert pr_opt_type in args.optimizer.refine_pose.valid_opts, f'Opt type: {pr_opt_type} not in {args.optimizer.refine_pose.valid_opt}'

    if pr_opt_type == 'Adam':
        pr_opt_cls = torch.optim.Adam 
    elif pr_opt_type == 'AdamW':
        pr_opt_cls = torch.optim.AdamW 
    else:
        raise ValueError(f'Unrecognized: {pr_opt_type}')

    pr_opt_kwargs = config_util.copy_attr_dict(args.optimizer[pr_opt_type])
    if args.optimizer.refine_pose.get('opt_kwargs') is not None:
        custom_pr_opt_kwargs = args.optimizer.refine_pose.opt_kwargs[pr_opt_type]
        config_util.update_recursive(pr_opt_kwargs, custom_pr_opt_kwargs)

    pr_optimizer = pr_opt_cls(pr_model_params, **pr_opt_kwargs)

    ###
    # Load success classifier and optimizer
    ###

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
    
    success_model = success_model_cls(
        mc_vis=mc_vis,
        feat_dim=args.model.success.feat_dim,
        **sc_args).cuda()

    sc_model_params = success_model.parameters()

    # loss
    sc_loss_type = args.loss.success.type
    assert sc_loss_type in args.loss.success.valid_losses, f'Loss type: {sc_loss_type} not in {args.loss.success.valid_losses}'
    if sc_loss_type == 'bce_wo_logits':
        sc_loss_fn = losses.success_bce
    elif sc_loss_type == 'bce_w_logits':
        double_batch = args.loss.bce_w_logits.double_batch_size
        batch_scalar = 2 if double_batch else 1  # in some experiments, we double the batch size for the success model
        bce_logits_wrapper = losses.BCEWithLogitsWrapper(pos_weight=args.loss.bce_w_logits.pos_weight, bs=args.experiment.batch_size*batch_scalar)
        sc_loss_fn = bce_logits_wrapper.success_bce_w_logits
    else:
        raise ValueError(f'Unrecognized: {sc_loss_type}')

    # optimizer
    sc_opt_type = args.optimizer.success.type
    assert sc_opt_type in args.optimizer.success.valid_opts, f'Opt type: {sc_opt_type} not in {args.optimizer.success.valid_opt}'

    if sc_opt_type == 'Adam':
        sc_opt_cls = torch.optim.Adam 
    elif sc_opt_type == 'AdamW':
        sc_opt_cls = torch.optim.AdamW 
    else:
        raise ValueError(f'Unrecognized: {sc_opt_type}')

    sc_opt_kwargs = config_util.copy_attr_dict(args.optimizer[sc_opt_type])
    if args.optimizer.success.get('opt_kwargs') is not None:
        custom_sc_opt_kwargs = args.optimizer.success.opt_kwargs[sc_opt_type]
        config_util.update_recursive(sc_opt_kwargs, custom_sc_opt_kwargs)

    sc_optimizer = sc_opt_cls(sc_model_params, **sc_opt_kwargs)

    ###
    # Load coarse affordance model, loss, and optimizer
    ###

    # model
    coarse_aff_type = args.model.coarse_aff.type
    coarse_aff_args = config_util.copy_attr_dict(args.model[coarse_aff_type])
    if args.model.coarse_aff.get('model_kwargs') is not None:
        custom_coarse_aff_args = args.model.coarse_aff.model_kwargs[coarse_aff_type]
        config_util.update_recursive(coarse_aff_args, custom_coarse_aff_args)

    if args.model.coarse_aff.multi_model:
        coarse_aff_args2 = config_util.copy_attr_dict(args.model[coarse_aff_type])
        if args.model.coarse_aff.get('model_kwargs2') is not None:
            custom_coarse_aff_args2 = args.model.coarse_aff.model_kwargs2[coarse_aff_type]
            config_util.update_recursive(coarse_aff_args2, custom_coarse_aff_args2)

    coarse_aff_model = CoarseAffordanceVoxelRot(
        mc_vis=mc_vis, 
        feat_dim=args.model.coarse_aff.feat_dim,
        rot_grid_dim=args.data.rot_grid_bins,
        padding=args.data.voxel_grid.padding,
        voxel_reso_grid=args.data.voxel_grid.reso_grid,
        euler_rot=args.model.coarse_aff.euler_rot,
        euler_bins_per_axis=args.model.coarse_aff.euler_bins_per_axis,
        scene_encoder_kwargs=coarse_aff_args).cuda()

    aff_model_params = coarse_aff_model.parameters()

    # loss
    aff_loss_type = args.loss.coarse_aff.type
    assert aff_loss_type in args.loss.coarse_aff.valid_losses, f'Loss type: {aff_loss_type} not in {args.loss.coarse_aff.valid_losses}'

    if aff_loss_type == 'voxel_affordance':
        aff_loss_fn = losses.voxel_affordance
    elif aff_loss_type == 'voxel_affordance_w_disc_rot':
        aff_loss_fn = losses.voxel_affordance_w_disc_rot
    elif aff_loss_type == 'voxel_affordance_w_disc_rot_euler':
        aff_loss_fn = losses.voxel_affordance_w_disc_rot_euler
    else:
        raise ValueError(f'Unrecognized: {aff_loss_type}')

    # optimizer
    aff_opt_type = args.optimizer.coarse_aff.type
    assert aff_opt_type in args.optimizer.coarse_aff.valid_opts, f'Opt type: {aff_opt_type} not in {args.optimizer.coarse_aff.valid_opt}'

    if aff_opt_type == 'AdamW':
        aff_opt_cls = torch.optim.AdamW
    elif aff_opt_type == 'Adam':
        aff_opt_cls = torch.optim.Adam
    else:
        raise ValueError(f'Unrecognized: {aff_opt_type}')

    aff_opt_kwargs = config_util.copy_attr_dict(args.optimizer[aff_opt_type])
    if args.optimizer.coarse_aff.get('opt_kwargs') is not None:
        custom_aff_opt_kwargs = args.optimizer.coarse_aff.opt_kwargs[aff_opt_type]
        config_util.update_recursive(aff_opt_kwargs, custom_aff_opt_kwargs)

    aff_optimizer = aff_opt_cls(aff_model_params, **aff_opt_kwargs)

    if args.model.coarse_aff.multi_model:

        coarse_aff_args2 = config_util.copy_attr_dict(args.model[coarse_aff_type])
        if args.model.coarse_aff.get('model_kwargs2') is not None:
            custom_coarse_aff_args2 = args.model.coarse_aff.model_kwargs2[coarse_aff_type]
            config_util.update_recursive(coarse_aff_args2, custom_coarse_aff_args2)

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

        aff_model_params2 = coarse_aff_model2.parameters()
        aff_loss_fn2 = aff_loss_fn
        aff_optimizer2 = aff_opt_cls(aff_model_params2, **aff_opt_kwargs)

        train_kwargs['coarse_aff_model2'] = {}
        train_kwargs['coarse_aff_model2']['model'] = coarse_aff_model2
        train_kwargs['coarse_aff_model2']['opt'] = aff_optimizer2
        train_kwargs['coarse_aff_model2']['loss_fn'] = aff_loss_fn2
        train_kwargs['coarse_aff_model2']['reso_grid'] = voxel_reso_grid2
        train_kwargs['coarse_aff_model2']['padding'] = padding2


    # model_dict = dict(model=model, rot=rot_model)
    # if args.debug:
    #     print('Coarse affordance model: ')
    #     print(coarse_aff_model)
    #     print('Refine pose model: ')
    #     print(pr_model)
    #     print('Success model: ')
    #     print(success_model)

    ##############################################
    # Load checkpoints if resuming

    if args.experiment.resume and args.resume_ap:
        # find the latest iteration
        ckpts = [int(val.split('model_')[1].replace('.pth', '')) for val in os.listdir(logdir) if (val.endswith('.pth') and 'latest' not in val)]
        args.experiment.resume_iter = max(ckpts)

    if args.experiment.resume_iter != 0:
        print(f'Resuming at iteration: {args.experiment.resume_iter}')
        # model_path = osp.join(logdir, 'model_latest.pth')
        model_path = osp.join(logdir, f'model_{args.experiment.resume_iter}.pth')
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

        if args.experiment.train.train_refine_pose:
            pr_model.load_state_dict(checkpoint['refine_pose_model_state_dict'])
            pr_optimizer.load_state_dict(checkpoint['pr_optimizer_state_dict'])

        if args.experiment.train.train_success:
            success_model.load_state_dict(checkpoint['success_model_state_dict'])
            sc_optimizer.load_state_dict(checkpoint['sc_optimizer_state_dict'])

        if args.experiment.train.train_coarse_aff:
            coarse_aff_model.load_state_dict(checkpoint['coarse_aff_model_state_dict'])
            aff_optimizer.load_state_dict(checkpoint['aff_optimizer_state_dict'])
            if args.model.coarse_aff.multi_model:
                coarse_aff_model2.load_state_dict(checkpoint['coarse_aff_model_state_dict2'])
                aff_optimizer.load_state_dict(checkpoint['aff_optimizer_state_dict2'])

    logger = SummaryWriter(logdir)
    it = args.experiment.resume_iter
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        raise ValueError('Cuda not available')

    ##############################################
    # Perform other checks to ensure experiment config is valid
    train_exp_args = config_util.copy_attr_dict(args.experiment.train)
    if train_exp_args.refine_pose_from_coarse_pred:
        assert train_exp_args.train_coarse_aff and train_exp_args.train_refine_pose, 'Must be training both coarse and refine to use predictions as refinement input'
    
    assert not (train_exp_args.success_from_coarse_pred and train_exp_args.success_from_refine_pred), 'Cannot predict success from both refine pred and coarse pred. Please only set one of these to True'

    if train_exp_args.success_from_coarse_pred: 
        assert train_exp_args.train_coarse_aff and train_exp_args.train_success, 'Must be training both coarse and success to use predictions as success input'

    if train_exp_args.success_from_refine_pred: 
        assert train_exp_args.train_refine_pose and train_exp_args.train_success, 'Must be training both refine and success to use predictions as success input'

    # train_kwargs = config_util.recursive_attr_dict(train_kwargs)

    train(
        mc_vis, 
        coarse_aff_model, pr_model, success_model, 
        aff_optimizer, pr_optimizer, sc_optimizer, 
        train_dataloader, val_dataloader, 
        aff_loss_fn, pr_loss_fn, sc_loss_fn,
        device, 
        logger, 
        logdir, 
        args,
        it, 
        **train_kwargs)


if __name__ == "__main__":
    """Parse input arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_fname', type=str, required=True, help='Name of config file')
    parser.add_argument('-d', '--debug', action='store_true', help='If True, run in debug mode')
    parser.add_argument('-dd', '--debug_data', action='store_true', help='If True, run data loader in debug mode')
    parser.add_argument('-p', '--port_vis', type=int, default=6000, help='Port for ZMQ url (meshcat visualization)')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Random seed')
    parser.add_argument('-l', '--local_dataset_dir', type=str, default=None, help='If the data is saved on a local drive, pass the root of that directory here')
    parser.add_argument('-r', '--resume', action='store_true', help='If set, resume experiment (required to be set in config as well)')
    parser.add_argument('-m', '--meshcat', action='store_true', help='If set, run with meshcat visualization (required to be set in config as well)')

    args = parser.parse_args()

    train_args = config_util.load_config(osp.join(path_util.get_train_config_dir(), args.config_fname))
    
    train_args['debug'] = args.debug
    train_args['debug_data'] = args.debug_data
    train_args['port_vis'] = args.port_vis
    train_args['seed'] = args.seed
    train_args['local_dataset_dir'] = args.local_dataset_dir
    train_args['meshcat_ap'] = args.meshcat
    train_args['resume_ap'] = args.resume

    train_args = config_util.recursive_attr_dict(train_args)

    main(train_args)
