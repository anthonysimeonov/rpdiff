# From: https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from rpdiff.utils.torch3d_util import euler_angles_to_matrix

from rpdiff.utils import batch_pcd_util
from rpdiff.utils.torch_scatter_utils import fps_downsample

# import MinkowskiEngine as ME
from rpdiff.utils.config_util import AttrDict
from torch.optim.optimizer import Optimizer


def adjust_learning_rate(
        optimizer: Optimizer, 
        epoch: float, 
        args: AttrDict) -> float:
    """Decay the learning rate with half-cycle cosine after warmup"""
    if args.fixed_lr:
        return args.lr
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
            (1. + np.cos(np.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr

    return lr


def get_linear_warmup_cosine_decay_sched(
        epoch: float, 
        warmup_epochs: int, 
        total_epochs: int, 
        input_value: float=1.0, 
        min_value: float=0.0) -> float:
    """Decay arbitrary value with half-cycle cosine after warmup"""
    if epoch < warmup_epochs:
        output_value = input_value * epoch / warmup_epochs 
    else:
        if epoch > total_epochs:
            epoch = total_epochs
        output_value = min_value + (input_value - min_value) * 0.5 * \
            (1. + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))

    return output_value 


def get_binary(it, thresh, mq_int):
    return ((it % mq_int) / mq_int) > thresh


def get_thresh_sched_binary(it, it_max, warmup=0, mq_int=1000, full_by_frac=0.5, verbose=False):
    if it < warmup:
        # return get_binary(it, 1.0, mq_int=mq_int)
        return False
    else:
        thresh = 1.0 * ((it_max * full_by_frac) - (it - warmup)) / (it_max * full_by_frac)
        thresh = np.clip(0.0, 1.0, thresh)
        if verbose:
            print(f'thresh: {thresh}')
        return get_binary(it, thresh, mq_int=mq_int)


def get_grad_norm(model: torch.nn.Module) -> float:
    total_norm = 0.0
    parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
    for p in parameters:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def check_enough_points(pcd: np.ndarray, n_pts: int) -> np.ndarray:
    if pcd.shape[0] < n_pts:
        while True:
            rand_rix = np.random.permutation(pcd.shape[0])
            rnd_pts = pcd[rand_rix[:100]]
            pcd = np.concatenate([pcd, rnd_pts], axis=0)

            if pcd.shape[0] >= n_pts:
                break
    return pcd


def sample_rot_perturb(scale: float, normal: bool=False) -> np.ndarray:
    if normal:
        small_rpy = scale * np.random.normal(size=(3,))
    else:
        small_rpy = scale * (np.random.random(3) - 0.5)
    return small_rpy


def sample_trans_perturb(scale: float, normal: bool=False) -> np.ndarray:
    if normal:
        small_trans = scale * np.random.normal(size=(3,))
    else:
        small_trans = scale * (np.random.random(3) - 0.5)
    return small_trans


def crop_pcd_batch(pcd, positions, box_length=0.18, use_extents=False, use_extents_pcd=None, 
                   fill_empty=True, fill_empty_pos='mean', build_mask=False):

    if fill_empty_pos not in ['mean', 'rand']:
        fill_empty_pos = 'mean'

    bs, n_pts = pcd.shape[0], pcd.shape[1]

    # create batched version of the point cloud
    pcd_batched_full = batch_pcd_util.batched_coordinates([p / (1.1/1024) for p in pcd]).cuda()
    pcd_batch_idx = pcd_batched_full[:, 0]
    pcd_batched = pcd_batched_full[:, 1:].float() * (1.1/1024)

    if use_extents:
        assert use_extents_pcd is not None
        high_batch = torch.max(use_extents_pcd, dim=1)
        low_batch = torch.min(use_extents_pcd, dim=1)
    else:
        # make big array of position coordinates 
        pos_batch_flat = positions.reshape(bs, 1, 3).repeat((1, n_pts, 1)).reshape(-1, 3)

        high_batch_flat = pos_batch_flat + box_length
        low_batch_flat = pos_batch_flat - box_length

    # grab indices that are inside these values
    below = pcd_batched < high_batch_flat
    above = pcd_batched > low_batch_flat
    crop_bool = torch.logical_and(above, below).all(-1)
    crop_idx = torch.where(crop_bool)[0]

    c_pcd_batch_idx = pcd_batch_idx[crop_idx]
    c_pcd_batched = pcd_batched[crop_idx]

    # build mask
    bd_mats = []
    n_pts_per_crop = []
    for b in range(bs):
        b_idx = torch.where(c_pcd_batch_idx == b)[0]
        n_batch = b_idx.shape[0]

        if fill_empty and (n_batch == 0):
            pcd_b = pcd[b].reshape(-1, 3)
            print(f'Filling empty...')

            if use_extents:
                ex_pcd_b = use_extents_pcd[b]
                high_pos = torch.max(ex_pcd_b, dim=1)
                low_pos = torch.min(ex_pcd_b, dim=1)
            else:
                crop_pos = torch.mean(pcd_b, dim=0).reshape(-1, 3)
                # high_pos = crop_pos + box_length
                # low_pos = crop_pos - box_length
                high_pos = torch.max(pcd_b, dim=0)[0]  # single point cloud in the batch, use dim=0
                low_pos = torch.min(pcd_b, dim=0)[0]

            below = pcd_b < high_pos
            above = pcd_b > low_pos

            new_crop = pcd_b[torch.where(torch.logical_and(above, below).all(-1))[0]]
            
            pre_idx = sum(n_pts_per_crop[:b])
            c_pcd_batched = torch.cat((c_pcd_batched[:pre_idx], new_crop, c_pcd_batched[pre_idx:]), dim=0)
            c_pcd_batch_idx = torch.cat((c_pcd_batch_idx[:pre_idx], torch.ones(new_crop.shape[0]).long().cuda() * b, c_pcd_batch_idx[pre_idx:]), dim=0)

            n_batch = new_crop.shape[0]

        n_pts_per_crop.append(n_batch)
        if build_mask:
            mask_mat = torch.ones((n_batch, n_batch)).long().cuda()
            bd_mats.append(mask_mat)

    n_pts_per_crop = torch.Tensor(n_pts_per_crop).long().cuda()

    if build_mask:
        mask = torch.block_diag(*bd_mats)
        return c_pcd_batched, c_pcd_batch_idx, n_pts_per_crop, mask

    return c_pcd_batched, c_pcd_batch_idx, n_pts_per_crop


def pad_flat_pcd_batch(pcd, batch_idx, n_pts_per_batch, n_pts_des, pcd_default):
    bs = torch.max(batch_idx).item() + 1
    pcd_batch = torch.empty((bs, n_pts_des, 3))

    for b in range(bs):
        pcd_b = pcd[torch.where(batch_idx == b)[0]].reshape(-1, 3)
        if n_pts_per_batch[b] == 0:
            pcd_b = pcd_default[b]

        n_pts = pcd_b.shape[0] 
        if n_pts < n_pts_des:
            n_missing = n_pts_des - n_pts 
            idx_fill = torch.randint(n_pts, size=(n_missing,))
            pcd_b = torch.cat((pcd_b, pcd_b[idx_fill]), dim=0)

            pcd_batch[b] = pcd_b[:n_pts_des]
        else:
            pcd_batch[b] = fps_downsample(pcd_b.reshape(1, -1, 3), n_pts_des).reshape(n_pts_des, 3)

    return pcd_batch


def euler_disc_pred_to_rotmat(euler_disc_pred, bins_per_axis):
    bs = euler_disc_pred.shape[0]

    disc_euler_rotx = torch.linspace(-np.pi, np.pi, bins_per_axis).reshape(1, -1).repeat((bs, 1)).float().cuda()
    disc_euler_roty = torch.linspace(-np.pi, np.pi, bins_per_axis).reshape(1, -1).repeat((bs, 1)) .float().cuda()
    disc_euler_rotz = torch.linspace(-np.pi, np.pi, bins_per_axis).reshape(1, -1).repeat((bs, 1)).float().cuda()

    rot_predx, rot_predy, rot_predz = torch.chunk(euler_disc_pred.detach(), 3, dim=1)
    euler_x = torch.gather(disc_euler_rotx, dim=1, index=torch.argmax(rot_predx, dim=1)[:, None]).detach() #.numpy()
    euler_y = torch.gather(disc_euler_roty, dim=1, index=torch.argmax(rot_predy, dim=1)[:, None]).detach() #.numpy()
    euler_z = torch.gather(disc_euler_rotz, dim=1, index=torch.argmax(rot_predz, dim=1)[:, None]).detach() #.numpy()

    euler_angles_t = torch.stack([euler_x, euler_y, euler_z], dim=1).reshape(bs, 3)
    top_rot_mats = euler_angles_to_matrix(euler_angles_t, 'ZYX')
    # euler_angles = torch.stack([euler_x, euler_y, euler_z], dim=1).numpy().reshape(bs, 3)
    # top_rot_mats = R.from_euler('ZYX', euler_angles).as_matrix()
    # top_rot_mats = torch.from_numpy(top_rot_mats).float().cuda()

    return top_rot_mats

