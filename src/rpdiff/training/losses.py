import torch

from rpdiff.utils.torch3d_util import matrix_to_quaternion, quaternion_to_matrix

bce = torch.nn.BCELoss()
bce_logits = torch.nn.BCEWithLogitsLoss()

class BCEWithLogitsWrapper:
    def __init__(self, pos_weight, bs):
        self.pos_weight = pos_weight
        self.bs = bs

        self.build_bce_logits()

    def build_bce_logits(self):
        self.bce_logits = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight*torch.ones([self.bs]).float().cuda())
    
    def get_bce_logits(self):
        return self.bce_logits

    def success_bce_w_logits(self, model_outputs, ground_truth, val=False):
        loss_dict = dict()
        label = ground_truth['success']

        loss_dict['success'] = self.bce_logits(model_outputs['success'].squeeze(), label.squeeze())

        return loss_dict

ce = torch.nn.CrossEntropyLoss(reduction='mean')

def occupancy(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['occ']
    label = (label + 1) / 2.

    loss_dict['occ'] = -1 * (label * torch.log(model_outputs['occ'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['occ'] + 1e-5)).mean()
    return loss_dict


def success_bce(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['success']

    loss_dict['success'] = -1 * (label * torch.log(model_outputs['success'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['success'] + 1e-5)).mean()
    return loss_dict


def success_bce_w_logits(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['success']

    loss_dict['success'] = bce_logits(model_outputs['success'].squeeze(), label.squeeze())

    return loss_dict


def voxel_affordance(model_outputs, ground_truth, reso, val=False):
    loss_dict = dict()
    label = ground_truth['child_centroid_voxel_index']

    # make one-hot
    label_onehot = torch.nn.functional.one_hot(label.long(), num_classes=reso**3)
    # from IPython import embed; embed()
    pred = model_outputs['voxel_affordance']

    # affordance_loss = bce(pred, label_onehot.float())
    affordance_loss = ce(pred, label_onehot.float())
    # loss_dict['affordance'] = affordance_loss * 1000
    loss_dict['voxel_affordance'] = affordance_loss

    return loss_dict


def voxel_affordance_w_disc_rot(model_outputs, ground_truth, reso, rot_reso, val=False):
    loss_dict = dict()
    label = ground_truth['child_centroid_voxel_index']
    rot_label = ground_truth['rot_mat_grid_index']

    # make one-hot
    label_onehot = torch.nn.functional.one_hot(label.long(), num_classes=reso**3)
    pred = model_outputs['voxel_affordance']

    affordance_loss = ce(pred, label_onehot.float())
    loss_dict['voxel_affordance'] = affordance_loss

    # make one-hot
    rot_label_onehot = torch.nn.functional.one_hot(rot_label.long(), num_classes=rot_reso)
    rot_pred = model_outputs['rot_affordance']

    rot_affordance_loss = ce(rot_pred, rot_label_onehot.float())
    loss_dict['rot_affordance'] = rot_affordance_loss

    return loss_dict


def voxel_affordance_w_disc_rot_euler(model_outputs, ground_truth, reso, rot_reso, val=False):
    loss_dict = dict()
    label = ground_truth['child_centroid_voxel_index']
    rot_label = ground_truth['euler_onehot']

    # make one-hot
    label_onehot = torch.nn.functional.one_hot(label.long(), num_classes=reso**3)
    pred = model_outputs['voxel_affordance']

    affordance_loss = ce(pred, label_onehot.float())
    loss_dict['voxel_affordance'] = affordance_loss

    # make one-hot
    rot_predx, rot_predy, rot_predz = torch.chunk(model_outputs['rot_affordance'], 3, dim=1)
    rlossx = ce(rot_predx, rot_label['x'].argmax(-1))
    rlossy = ce(rot_predx, rot_label['y'].argmax(-1))
    rlossz = ce(rot_predx, rot_label['z'].argmax(-1))

    loss_dict['rot_affordance'] = rlossx + rlossy + rlossz

    return loss_dict


def success_bce_offset(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['success'].view(-1, 1)
    offset_label = ground_truth['parent_to_child_offset']

    # loss_dict['success'] = -1 * (label * torch.log(model_outputs['success'] + 1e-5) + (1 - label) * torch.log(1 - model_outputs['success'] + 1e-5)).mean()
    loss_dict['success'] = bce(model_outputs['success'], label)

    # parent to child offset loss
    loss_dict['offset'] = torch.norm(offset_label - model_outputs['p2c_offset'], p=2, dim=-1).mean()

    return loss_dict


def success_bce(model_outputs, ground_truth, val=False):
    loss_dict = dict()
    label = ground_truth['success'].view(-1, 1)

    loss_dict['success'] = bce(model_outputs['success'], label)

    return loss_dict
    

def tf_chamfer_offset(model_outputs, ground_truth, quat_norm_beta=0.1, val=False):
    loss_dict = {}
    
    trans_label = ground_truth['trans']
    rot_mat_label = ground_truth['rot_mat']
    quat_label = matrix_to_quaternion(rot_mat_label)
    offset_label = ground_truth['parent_to_child_offset']
    child_pcd = ground_truth['child_final_pcd']

    # translation loss
    trans_loss = torch.norm(trans_label - model_outputs['trans'], p=2, dim=-1).mean()

    # rotation loss (geodesic, and encourage normalized quat)
    # rot_loss = torch.norm(quat_label - model_outputs[''], p=2, dim=-1).mean()
    quat_scalar_prod = torch.sum(model_outputs['quat'] * quat_label, axis=1)
    quat_dist = 1 - torch.pow(quat_scalar_prod, 2)

    norm_loss = quat_norm_beta*torch.pow((1 - torch.norm(model_outputs['unnorm_quat'], p=2, dim=1)), 2)
    rot_loss = (quat_dist + norm_loss).mean()

    # chamfer loss
    pcd_dist_matrix = torch.norm(child_pcd[:, :, None, :] - model_outputs['child_pcd_final_pred'][:, None, :, :], p=2, dim=-1)
    loss_pcd_dist = pcd_dist_matrix.min(dim=1)[0] + pcd_dist_matrix.min(dim=2)[0]
    chamf_loss = loss_pcd_dist.mean()

    # parent to child offset loss
    # offset_loss = torch.norm(offset_label - model_outputs['p2c_offset'], p=2, dim=-1).mean()

    loss_dict['trans'] = trans_loss
    loss_dict['rot'] = rot_loss
    loss_dict['chamf'] = chamf_loss
    # loss_dict['offset'] = offset_loss
    # loss_dict['offset'] = 0.0

    return loss_dict


class TransformChamferWrapper:
    def __init__(self, l1=False, trans_offset=False, kl_div=False):
        
        if l1:
            self.trans_loss_fn = torch.abs
            self.trans_loss_kwargs = {}
        else:
            self.trans_loss_fn = torch.norm
            self.trans_loss_kwargs = {'p': 2, 'dim': -1}
        
        self.trans_offset = trans_offset

    def tf_chamfer(self, model_outputs, ground_truth, quat_norm_beta=0.1, val=False):
        loss_dict = {}
        
        trans_label = ground_truth['trans']
        rot_mat_label = ground_truth['rot_mat']
        quat_label = matrix_to_quaternion(rot_mat_label)
        child_pcd = ground_truth['child_final_pcd']
        bs = child_pcd.shape[0]

        # translation loss (l1 or l2)
        trans_loss = self.trans_loss_fn(trans_label - model_outputs['trans'], **self.trans_loss_kwargs).mean()

        # rotation loss (geodesic, and encourage normalized quat)
        # rot_loss = torch.norm(quat_label - model_outputs[''], p=2, dim=-1).mean()
        quat_scalar_prod = torch.sum(model_outputs['quat'] * quat_label, axis=1)
        quat_dist = 1 - torch.pow(quat_scalar_prod, 2)

        norm_loss = quat_norm_beta*torch.pow((1 - torch.norm(model_outputs['unnorm_quat'], p=2, dim=1)), 2)
        rot_loss = (quat_dist + norm_loss).mean()

        # chamfer loss
        pcd_dist_matrix = torch.norm(child_pcd[:, :, None, :] - model_outputs['child_pcd_final_pred'][:, None, :, :], p=2, dim=-1)
        loss_pcd_dist = pcd_dist_matrix.min(dim=1)[0] + pcd_dist_matrix.min(dim=2)[0]
        chamf_loss = loss_pcd_dist.mean()

        loss_dict['trans'] = trans_loss
        loss_dict['rot'] = rot_loss
        loss_dict['chamf'] = chamf_loss

        if self.trans_offset:
            trans_offset_label = ground_truth['trans_offset']
            trans_offset_loss = torch.norm(trans_offset_label - model_outputs['trans_offset'], p=2, dim=-1).mean()
            loss_dict['trans_offset'] = trans_offset_loss

        return loss_dict

    def tf_chamfer_w_kldiv(self, model_outputs, ground_truth, quat_norm_beta=0.1, val=False):
        loss_dict = self.tf_chamfer(
            model_outputs=model_outputs,
            ground_truth=ground_truth,
            quat_norm_beta=quat_norm_beta,
            val=val)
        
        # mu = model_outputs['z_mu']
        # logvar = model_outputs['z_logvar']
        # kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar)))
        # loss_dict['kl'] = kl_loss
        trans_mu = model_outputs['z_mu_trans']
        trans_logvar = model_outputs['z_logvar_trans']
        trans_kl_loss = torch.mean(-0.5 * torch.sum(1 + trans_logvar - torch.pow(trans_mu, 2) - torch.exp(trans_logvar)))

        rot_mu = model_outputs['z_mu_rot']
        rot_logvar = model_outputs['z_logvar_rot']
        rot_kl_loss = torch.mean(-0.5 * torch.sum(1 + rot_logvar - torch.pow(rot_mu, 2) - torch.exp(rot_logvar)))

        loss_dict['trans_kl'] = trans_kl_loss
        loss_dict['rot_kl'] = rot_kl_loss

        return loss_dict


class TransformChamferMultiQueryAffordanceWrapper:
    def __init__(self, l1=False, trans_offset=False, kl_div=False):
        
        if l1:
            self.trans_loss_fn = torch.abs
            self.trans_loss_kwargs = {}
        else:
            self.trans_loss_fn = torch.norm
            self.trans_loss_kwargs = {'p': 2, 'dim': -1}
        
        self.trans_offset = trans_offset

    def tf_chamfer_multi_query_affordance(self, model_outputs, ground_truth, quat_norm_beta=0.1, val=False):
        loss_dict = {}
        
        trans_label = ground_truth['trans']
        rot_mat_label = ground_truth['rot_mat']
        quat_label = matrix_to_quaternion(rot_mat_label)
        child_pcd = ground_truth['child_final_pcd']
        bs = child_pcd.shape[0]

        rot_aff_label = ground_truth['rot_multi_query_affordance']
        trans_aff_label = ground_truth['trans_multi_query_affordance']
        rot_aff_pred = model_outputs['rot_multi_query_affordance']
        trans_aff_pred = model_outputs['trans_multi_query_affordance']

        # rot_aff_label = torch.argmax(rot_aff_label, dim=-1).long()
        # trans_aff_label = torch.argmax(trans_aff_label, dim=-1).long()
    
        # # rot_affordance_loss = ce(rot_aff_pred, rot_aff_label.float()[:, :, None])
        # # trans_affordance_loss = ce(trans_aff_pred, trans_aff_label.float()[:, :, None])
        # rot_affordance_loss = ce(rot_aff_pred.reshape(bs, -1), rot_aff_label)
        # trans_affordance_loss = ce(trans_aff_pred.reshape(bs, -1), trans_aff_label)
        rot_affordance_loss = bce_logits(rot_aff_pred.reshape(bs, -1), rot_aff_label.float())
        trans_affordance_loss = bce_logits(trans_aff_pred.reshape(bs, -1), trans_aff_label.float())

        # translation loss (l1 or l2)
        trans_loss = self.trans_loss_fn(trans_label - model_outputs['trans'], **self.trans_loss_kwargs).mean()

        # rotation loss (geodesic, and encourage normalized quat)
        # rot_loss = torch.norm(quat_label - model_outputs[''], p=2, dim=-1).mean()
        quat_scalar_prod = torch.sum(model_outputs['quat'] * quat_label, axis=1)
        quat_dist = 1 - torch.pow(quat_scalar_prod, 2)

        norm_loss = quat_norm_beta*torch.pow((1 - torch.norm(model_outputs['unnorm_quat'], p=2, dim=1)), 2)
        rot_loss = (quat_dist + norm_loss).mean()

        # chamfer loss
        pcd_dist_matrix = torch.norm(child_pcd[:, :, None, :] - model_outputs['child_pcd_final_pred'][:, None, :, :], p=2, dim=-1)
        loss_pcd_dist = pcd_dist_matrix.min(dim=1)[0] + pcd_dist_matrix.min(dim=2)[0]
        chamf_loss = loss_pcd_dist.mean()

        loss_dict['trans'] = trans_loss
        loss_dict['rot'] = rot_loss
        loss_dict['chamf'] = chamf_loss
        loss_dict['rot_affordance'] = rot_affordance_loss
        loss_dict['trans_affordance'] = trans_affordance_loss

        if self.trans_offset:
            trans_offset_label = ground_truth['trans_offset']
            trans_offset_loss = torch.norm(trans_offset_label - model_outputs['trans_offset'], p=2, dim=-1).mean()
            loss_dict['trans_offset'] = trans_offset_loss

        return loss_dict

    def tf_chamfer_multi_query_affordance_w_kldiv(self, model_outputs, ground_truth, quat_norm_beta=0.1, val=False):
        loss_dict = self.tf_chamfer_multi_query_affordance(
            model_outputs=model_outputs,
            ground_truth=ground_truth,
            quat_norm_beta=quat_norm_beta,
            val=val)
        
        # mu = model_outputs['z_mu']
        # logvar = model_outputs['z_logvar']
        # kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - torch.pow(mu, 2) - torch.exp(logvar)))
        # loss_dict['kl'] = kl_loss
        trans_mu = model_outputs['z_mu_trans']
        trans_logvar = model_outputs['z_logvar_trans']
        trans_kl_loss = torch.mean(-0.5 * torch.sum(1 + trans_logvar - torch.pow(trans_mu, 2) - torch.exp(trans_logvar)))

        rot_mu = model_outputs['z_mu_rot']
        rot_logvar = model_outputs['z_logvar_rot']
        rot_kl_loss = torch.mean(-0.5 * torch.sum(1 + rot_logvar - torch.pow(rot_mu, 2) - torch.exp(rot_logvar)))

        loss_dict['trans_kl'] = trans_kl_loss
        loss_dict['rot_kl'] = rot_kl_loss

        return loss_dict


def tf_chamfer_multi_query_affordance(model_outputs, ground_truth, quat_norm_beta=0.1, val=False, l1=False):
    loss_dict = {}
    
    trans_label = ground_truth['trans']
    rot_mat_label = ground_truth['rot_mat']
    quat_label = matrix_to_quaternion(rot_mat_label)
    child_pcd = ground_truth['child_final_pcd']

    rot_aff_label = ground_truth['rot_multi_query_affordance']
    trans_aff_label = ground_truth['trans_multi_query_affordance']
    rot_aff_pred = model_outputs['rot_multi_query_affordance']
    trans_aff_pred = model_outputs['trans_multi_query_affordance']

    rot_affordance_loss = ce(rot_aff_pred, rot_aff_label.float()[:, :, None])
    trans_affordance_loss = ce(trans_aff_pred, trans_aff_label.float()[:, :, None])

    # translation loss
    if l1:
        trans_loss = torch.abs(trans_label - model_outputs['trans']).mean()
    else:
        trans_loss = torch.norm(trans_label - model_outputs['trans'], p=2, dim=-1).mean()
        # trans_loss = torch.norm(trans_label - model_outputs['trans'], p=1, dim=-1).mean()

    # rotation loss (geodesic, and encourage normalized quat)
    # rot_loss = torch.norm(quat_label - model_outputs[''], p=2, dim=-1).mean()
    quat_scalar_prod = torch.sum(model_outputs['quat'] * quat_label, axis=1)
    quat_dist = 1 - torch.pow(quat_scalar_prod, 2)

    norm_loss = quat_norm_beta*torch.pow((1 - torch.norm(model_outputs['unnorm_quat'], p=2, dim=1)), 2)
    rot_loss = (quat_dist + norm_loss).mean()

    # chamfer loss
    pcd_dist_matrix = torch.norm(child_pcd[:, :, None, :] - model_outputs['child_pcd_final_pred'][:, None, :, :], p=2, dim=-1)
    loss_pcd_dist = pcd_dist_matrix.min(dim=1)[0] + pcd_dist_matrix.min(dim=2)[0]
    chamf_loss = loss_pcd_dist.mean()

    loss_dict['trans'] = trans_loss
    loss_dict['rot'] = rot_loss
    loss_dict['chamf'] = chamf_loss
    loss_dict['rot_affordance'] = rot_affordance_loss
    loss_dict['trans_affordance'] = trans_affordance_loss

    return loss_dict

def tf_chamfer_multi_query_affordance_no_agg(model_outputs, ground_truth, quat_norm_beta=0.1, val=False, l1=False):
    loss_dict = {}
    
    trans_label = ground_truth['trans']
    rot_mat_label = ground_truth['rot_mat']
    quat_label = matrix_to_quaternion(rot_mat_label)
    child_pcd = ground_truth['child_final_pcd']

    rot_aff_label = ground_truth['rot_multi_query_affordance']
    trans_aff_label = ground_truth['trans_multi_query_affordance']
    rot_aff_pred = model_outputs['rot_multi_query_affordance']
    trans_aff_pred = model_outputs['trans_multi_query_affordance']

    rot_affordance_loss = ce(rot_aff_pred, rot_aff_label.float()[:, :, None])
    trans_affordance_loss = ce(trans_aff_pred, trans_aff_label.float()[:, :, None])

    # translation loss
    if l1:
        trans_loss = torch.abs(trans_label - model_outputs['trans'])
    else:
        trans_loss = torch.norm(trans_label - model_outputs['trans'], p=2, dim=-1)
        # trans_loss = torch.norm(trans_label - model_outputs['trans'], p=1, dim=-1).mean()

    # rotation loss (geodesic, and encourage normalized quat)
    # rot_loss = torch.norm(quat_label - model_outputs[''], p=2, dim=-1).mean()
    quat_scalar_prod = torch.sum(model_outputs['quat'] * quat_label, axis=1)
    quat_dist = 1 - torch.pow(quat_scalar_prod, 2)

    norm_loss = quat_norm_beta*torch.pow((1 - torch.norm(model_outputs['unnorm_quat'], p=2, dim=1)), 2)
    rot_loss = (quat_dist + norm_loss)

    # chamfer loss
    pcd_dist_matrix = torch.norm(child_pcd[:, :, None, :] - model_outputs['child_pcd_final_pred'][:, None, :, :], p=2, dim=-1)
    loss_pcd_dist = pcd_dist_matrix.min(dim=1)[0] + pcd_dist_matrix.min(dim=2)[0]
    chamf_loss = loss_pcd_dist

    loss_dict['trans'] = trans_loss
    loss_dict['rot'] = rot_loss
    loss_dict['chamf'] = chamf_loss
    loss_dict['rot_affordance'] = rot_affordance_loss
    loss_dict['trans_affordance'] = trans_affordance_loss

    return loss_dict
