import os.path as osp
import os
import time
import torch
import numpy as np
import argparse
import random
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import meshcat
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from imageio import imread

import clip

from airobot import log_debug, log_info, log_warn, log_critical, set_log_level

from rpdiff.utils import util, torch3d_util, path_util
from rpdiff.utils.plotly_save import plot3d
from rpdiff.utils.torch_util import transform_pcd_torch, dict_to_gpu


class QueryHelper:
    def __init__(self, caption):
        self.setup_clip_model()
        self.set_clip_image_caption(caption)

    def setup_clip_model(self):
        # CLIP model settings
        clip_model = "ViT-B/32" # ["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"]

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(clip_model, device=self.device, jit=False)


    def set_clip_image_caption(self, caption):
        log_info(f'Setting image caption for CLIP: {caption}')
        self.image_caption = caption

    def get_clip_scores(self, image_paths, viz=False):
        # image_caption = 'the mug is hanging on the rack'
        image_caption = self.image_caption
        device = self.device
        model = self.clip_model

        reso = (480, 640)
        batch = []
        batch_np = []
        for image_path in image_paths:
            image_input = self.preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image_np = imread(image_path)

            batch.append(image_input)
            batch_np.append(image_np)
            
        batch = torch.cat(batch, dim=0).to(device)
        text_input = clip.tokenize([image_caption]).to(device)

        with torch.no_grad():
            patch_embs = model.encode_image(batch).float()
            text_embs = model.encode_text(text_input).float()
            patch_embs = patch_embs / patch_embs.norm(dim=-1, keepdim=True)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)
            sim = patch_embs @ text_embs.t()
                
        scores = sim.cpu().numpy()
        ranked_scores = np.sort(scores, 0).squeeze()[::-1]
        ranked_idx = np.argsort(scores, 0).squeeze()[::-1]
        # print(ranked_idx)
        # print(ranked_scores)
        
        if viz:
            n_row, n_col = int(len(image_paths)/5) + 1, 5
            _, axs = plt.subplots(n_row, n_col, figsize=(24, 4*n_row))
            axs = axs.flatten()
            for i, ax in enumerate(axs):
                score_idx = ranked_idx[i]
                score = ranked_scores[i]
                img = batch_np[score_idx]
                ax.imshow(img)
                ax.set_title(f'CLIP Score: {score:.5f}')
            plt.show()
            plt.savefig('clip_scores.png')

        return scores
        
    def visualize_top_mid_bottom_clip(self, n_total, clip_mid_n, stable_image_names, clip_scores):
        mid_idx_list = [
            int(n_total / 2),
            int(clip_mid_n / 2),
            n_total - int(clip_mid_n / 2)
        ]
        save_fig_list = ['middle', 'top', 'bottom']
        for mi, mid_idx in enumerate(mid_idx_list):
            top_idx = mid_idx + int(clip_mid_n/2)
            bottom_idx = mid_idx - int(clip_mid_n/2)

            query_idxs = []
            for qix in range(bottom_idx, top_idx):
                query_idxs.append(ranked_idx[qix])

            query_img_np = []
            for qid in query_idxs:
                # show human whatever is relevant
                stable_img = imread(stable_image_names[qid])
                query_img_np.append(stable_img) 

            n_row, n_col = int(clip_mid_n/5), 5
            _, axs = plt.subplots(n_row, n_col, figsize=(24, 4*n_row))
            axs = axs.flatten()
            for i, ax in enumerate(axs):
                qix = query_idxs[i]
                img = imread(stable_image_names[qix])
                score = clip_scores[qix]
                if isinstance(score, list):
                    score = score[0]
                ax.imshow(img)
                ax.set_title(f'CLIP Score: {score:.5f}')
            plt.savefig(f'clip_{save_fig_list[mi]}_scores.png')
            # plt.show()

    def write_success_with_queries(self, mc_vis, vd, model, rpdiff_dict, test_dataloader, loss_fn, dev, logger, logdir, args, step=0, viz=False):

        ####################################################################################
        # first, go through all the data, collect the corresponding image paths, and get CLIP scores

        model = model.eval()
        parent_rpdiff = rpdiff_dict['parent']
        child_rpdiff = rpdiff_dict['child']

        bs = args.batch_size

        # debug_idx = 0
        debug_idx = args.debug_idx
        val_step = 0
        
        success_samples = []
        stable_image_names = []
        clip_scores = []

        for i, sample in enumerate(test_dataloader):

            parent_rpdiff_mi, child_rpdiff_mi, policy_mi, gt = sample

            # obtain descriptor values for the input (parent RPDIFF evaluated at child pcd)
            parent_rpdiff_mi = dict_to_gpu(parent_rpdiff_mi)
            if args.debug:
                # debug_idx = 0
                debug_idx = args.debug_idx
                util.meshcat_pcd_show(mc_vis, parent_rpdiff_mi['point_cloud'][debug_idx].detach().cpu().numpy(), (255, 0, 0), 'scene/parent_rpdiff_mi/point_cloud')
                util.meshcat_pcd_show(mc_vis, parent_rpdiff_mi['coords'][debug_idx].detach().cpu().numpy(), (0, 0, 255), 'scene/parent_rpdiff_mi/coord')
            parent_latent = parent_rpdiff.model.extract_latent(parent_rpdiff_mi).detach()  # assumes we have already centered based on the parent
            parent_rpdiff_child_desc = parent_rpdiff.model.forward_latent(parent_latent, parent_rpdiff_mi['coords']).detach()

            # obtain descriptor values for the input (child RPDIFF evaluated at parent pcd)
            child_rpdiff_mi = dict_to_gpu(child_rpdiff_mi)
            if args.debug:
                util.meshcat_pcd_show(mc_vis, child_rpdiff_mi['point_cloud'][debug_idx].detach().cpu().numpy(), (255, 0, 0), 'scene/child_rpdiff_mi/point_cloud')
                util.meshcat_pcd_show(mc_vis, child_rpdiff_mi['coords'][debug_idx].detach().cpu().numpy(), (0, 0, 255), 'scene/child_rpdiff_mi/coord')
            child_latent = child_rpdiff.model.extract_latent(child_rpdiff_mi).detach()  # assumes we have already centered based on the child
            child_rpdiff_parent_desc = child_rpdiff.model.forward_latent(child_latent, child_rpdiff_mi['coords']).detach()

            # prepare inputs to the policy 
            policy_mi = dict_to_gpu(policy_mi)
            policy_mi['parent_rpdiff_child_desc'] = parent_rpdiff_child_desc
            policy_mi['child_rpdiff_parent_desc'] = child_rpdiff_parent_desc
            policy_mi['parent_latent'] = parent_latent
            policy_mi['child_latent'] = child_latent
            
            if args.debug:
                util.meshcat_pcd_show(mc_vis, policy_mi['parent_start_pcd'][debug_idx].detach().cpu().numpy(), (255, 0, 0), 'scene/policy_mi/parent_pcd')
                util.meshcat_pcd_show(mc_vis, policy_mi['child_start_pcd'][debug_idx].detach().cpu().numpy(), (0, 0, 255), 'scene/policy_mi/child_pcd')
                util.meshcat_pcd_show(
                    mc_vis, 
                    (policy_mi['parent_start_pcd'][debug_idx].detach().cpu().numpy() + policy_mi['parent_start_pcd_mean'][debug_idx].detach().cpu().numpy()), 
                    (255, 0, 0), 
                    'scene/policy_mi/parent_pcd_uncent')
                util.meshcat_pcd_show(
                    mc_vis, 
                    (policy_mi['child_start_pcd'][debug_idx].detach().cpu().numpy() + policy_mi['child_start_pcd_mean'][debug_idx].detach().cpu().numpy()),
                    (0, 0, 255), 
                    'scene/policy_mi/child_pcd_uncent')
                util.meshcat_pcd_show(mc_vis, gt['parent_final_pcd'][debug_idx].detach().cpu().numpy(), (255, 0, 255), 'scene/gt/parent_final_pcd')
                util.meshcat_pcd_show(mc_vis, gt['child_final_pcd'][debug_idx].detach().cpu().numpy(), (0, 255, 255), 'scene/gt/child_final_pcd')
                p2c_offset = policy_mi['parent_start_pcd_mean'] - policy_mi['child_start_pcd_mean']
                # print('p2c_offset: ', p2c_offset)
                # print('gt_p2c_offset: ', gt['parent_to_child_offset'])
            model_output = model(policy_mi)

            batch_succ = []
            batch_image_names = []
            for b in range(bs):
                succ_pred = model_output['success'][b].detach().cpu().numpy().squeeze() > 0.5

                sample_name_full = test_dataloader.dataset.files[i*bs + b]
                sample_data_dict = util.npz2dict(np.load(sample_name_full, allow_pickle=True))
                sample_data_dict['pred_success'] = succ_pred
                np.savez(sample_name_full, **sample_data_dict)

                sample_name = sample_name_full.split('/')[-1]

                # fname_suffix = sample_name.split('_')[-1]
                fname_prefix, fname_suffix = '_'.join(sample_name.split('_')[:-1]), sample_name.split('_')[-1]
                stable_image_name = fname_prefix + '_stable_' + fname_suffix.replace('.npz', '.png')
                stable_image_name_full = osp.join(test_dataloader.dataset.data_path, stable_image_name)

                batch_image_names.append(stable_image_name_full)
                stable_image_names.append(stable_image_name_full)

                if succ_pred:
                    # write the name of this sample to our list of successes
                    success_samples.append(sample_name)
                    batch_succ.append(sample_name)

            print(f'Predicted {len(batch_succ)} success samples on this batch, iteration: {i}')

            batch_scores = self.get_clip_scores(batch_image_names).squeeze().tolist()
            clip_scores.extend(batch_scores)

        ####################################################################################
        # then, run sample selection methods

        query_idxs = []

        # CLIP middle few
        ranked_scores = np.sort(clip_scores, 0).squeeze()[::-1]
        ranked_idx = np.argsort(clip_scores, 0).squeeze()[::-1]
        
        clip_mid_n = 10
        n_total = len(ranked_idx)
        mid_idx = int(n_total / 2)

        # visualize_top_mid_bottom_clip(n_total, clip_mid_n, stable_image_names, clip_scores)
        top_idx = mid_idx + int(clip_mid_n/2)
        bottom_idx = mid_idx - int(clip_mid_n/2)

        for qix in range(bottom_idx, top_idx):
            query_idxs.append(ranked_idx[qix])
        
        if viz:
            n_row, n_col = int(clip_mid_n/5), 5
            _, axs = plt.subplots(n_row, n_col, figsize=(24, 4*n_row))
            axs = axs.flatten()
            for i, ax in enumerate(axs):
                qix = query_idxs[i]
                # img = query_img_np[i]
                # score_idx = ranked_idx[qix]
                # score = ranked_scores[qix]
                img = imread(stable_image_names[qix])
                score = clip_scores[qix]
                if isinstance(score, list):
                    score = score[0]
                ax.imshow(img)
                ax.set_title(f'CLIP Score: {score:.5f}')
            plt.savefig('clip_middle_scores.png')

        # # random sample
        # rnd_n = 5
        # rnd_sample = random.sample(range(n_total), rnd_n)
        # query_idxs.extend(rnd_sample)

        # ensemble disagreement (TODO)
        
        query_samples = []
        query_img_np = []
        vdwin = None
        for qid in query_idxs:
            # show human whatever is relevant
            stable_img = imread(stable_image_names[qid])
            query_img_np.append(stable_img) 
            if vdwin is None:
                vdwin = vd.image(stable_img[:, :, :3].transpose(2, 0, 1))
            else:
                vdwin = vd.image(stable_img[:, :, :3].transpose(2, 0, 1), win=vdwin)
            # plt.imshow(stable_img)
            # plt.show()
            parent_rpdiff_mi, child_rpdiff_mi, policy_mi, gt = test_dataloader.dataset[qid]
            util.meshcat_pcd_show(mc_vis, gt['parent_final_pcd'], (255, 0, 255), 'scene/gt_query/parent_final_pcd')
            util.meshcat_pcd_show(mc_vis, gt['child_final_pcd'], (0, 255, 255), 'scene/gt_query/child_final_pcd')

            # get their label
            input_value = input('\nEnter "s" if this is a successful sample, otherwise enter anything else\n')
            
            queried_success = input_value == 's'

            # overwrite the relevant parts of the data in this sample
            sample_name_full = test_dataloader.dataset.files[qid]
            sample_data_dict = util.npz2dict(np.load(sample_name_full, allow_pickle=True))
            sample_data_dict['queried_success'] = queried_success
            np.savez(sample_name_full, **sample_data_dict)

            # update the names of the samples included in the demo split
            sample_name = sample_name_full.split('/')[-1]
            query_samples.append(sample_name)

        # write the list of successful samples to a special split
        success_split_fname = osp.join(test_dataloader.dataset.data_path, 'split_info', args.success_split_fname) 
        success_split_str = '\n'.join(success_samples)
        with open(success_split_fname, 'w') as f:
            f.write(success_split_str)
        print(f'Wrote success sample split to {success_split_fname}')

        # append the list of queried samples to the demo_train split
        query_split_fname = osp.join(test_dataloader.dataset.data_path, 'split_info', 'train_demo_split.txt')
        query_split_str = '\n'.join(query_samples)
        query_split_str = f'\n{query_split_str}'
        with open(query_split_fname, 'a') as f:
            f.write(query_split_str)
        print(f'Appended query samples to demo_train split (file: {query_split_fname})')

        from IPython import embed; embed()
