import torch
import numpy as np
from collections import OrderedDict
from model.utils.structure_utils.atom_geometry import *
from model.utils.structure_utils.mmcif_labels import structure_loss_labels
from model.utils.structure_utils import residue_constants as rc
from typing import Dict, List
import ml_collections
eps = 1e-6

# Missing confidence heads retain zero-valued report slots.
# comp_structure_loss returns the twelve unweighted loss terms below, followed
# by aggregate/quality metrics and three repeated terms. Keep this order aligned
# with comp_structure_loss so report columns remain interpretable.
STRUCTURE_LOSS_ITEM_NAMES = (
    "loss_translation_est",
    "loss_quaternion_est",
    "loss_ce",
    "loss_dihedral",
    "loss_quaternion",
    "loss_fape",
    "lddt_loss",
    "sidechain_fape_loss",
    "angles_loss",
    "structure_violation_loss",
    "plddt_confidence_loss",
    "pae_confidence_loss",
    "weighted_loss_sum",
    "structure_loss_raw",
    "real_fape",
    "lddt",
    "lddt_best",
    "sidechain_fape_loss_repeated",
    "angles_loss_repeated",
    "structure_violation_loss_repeated",
)

STRUCTURE_TEST_REPORT_NAMES = STRUCTURE_LOSS_ITEM_NAMES[:17]


def SelectAncher(embedding, AncherList, SelectAxis, BatchAxis=None):
    # embedding: [..., batch, ..., length, ...]
    # AncherList: torch.LongTensor([1, 2, 3]) or torch.LongTensor([[0, 1, 2], [1, 2, 3]])
    # SelectAxis: int
    # BatchAxis:  int
    AncherList = torch.LongTensor(AncherList).to(embedding.device)
    if AncherList.dim() == 1:
        ret = embedding.index_select(SelectAxis, AncherList)
        return ret
    elif AncherList.dim() == 2 and BatchAxis != None:
        ret = []
        if SelectAxis == 0:
            if BatchAxis != 0:
                embedding = embedding.transpose(0, BatchAxis)
                for i in range(AncherList.shape[0]):
                    ret.append(embedding[i].index_select(BatchAxis - 1, AncherList[i]))
                ret = torch.stack(ret).transpose(0, BatchAxis)
                return ret
            else:
                print('1 Wrong!!!!')
        else:
            embedding = embedding.transpose(0, BatchAxis)
            for i in range(AncherList.shape[0]):
                ret.append(embedding[i].index_select(SelectAxis - 1, AncherList[i]))
            ret = torch.stack(ret).transpose(0, BatchAxis)
            return ret
    else:
        print('2 Wrong!!!!')

def NormQuaternion(q):
    q = q/torch.sqrt((q * q).sum(-1, keepdim=True) + eps)
    q = torch.sign(torch.sign(q[..., 0]) + 0.5).unsqueeze(-1) * q
    return q

def QuaternionMM(q1, q2):
    if q1.dim() == q2.dim():
        q1_shape = torch.tensor([i for i in q1.shape])
        q2_shape = torch.tensor([i for i in q2.shape])

        q_shape_max, _ = torch.stack([q1_shape, q2_shape]).max(0)

        q1 = q1.repeat(list((q_shape_max/q1_shape).long()))
        q2 = q2.repeat(list((q_shape_max/q2_shape).long()))

        a = q1[..., 0] * q2[..., 0] - (q1[..., 1:] * q2[..., 1:]).sum(-1)
        bcd = torch.cross(q2[..., 1:], q1[..., 1:], dim=-1) + q1[..., 0].unsqueeze(-1) * q2[..., 1:] + q2[..., 0].unsqueeze(-1) * q1[..., 1:]
        q = torch.cat([a.unsqueeze(-1), bcd], dim=-1)
        return q
    else:
        print('Shape of q1&q2 not correct!')

def NormQuaternionMM(q1, q2):
    q = QuaternionMM(q1, q2)
    return NormQuaternion(q)
    

def TranslationRotation(q, t):
    if q.dim() == t.dim() and q.dim() > 1 and q.shape[-1] == 4 and t.shape[-1] == 3:
        q_shape = torch.tensor([i for i in q.shape[:-1]])
        t_shape = torch.tensor([i for i in t.shape[:-1]])
        qt_shape_max, _ = torch.stack([q_shape, t_shape]).max(0)

        q_shape = list(torch.cat([(qt_shape_max/q_shape), torch.ones(1)]).long())
        q = q.repeat(q_shape)

        t_shape = list(torch.cat([(qt_shape_max/t_shape), torch.ones(1)]).long())
        t = t.repeat(t_shape)

        t4 = torch.cat([torch.zeros_like(t[..., 0])[..., None], t], dim=-1)
        q_inv = torch.cat([q[..., 0][..., None], -q[..., 1:]], dim=-1)
        return QuaternionMM(QuaternionMM(q, t4), q_inv)[..., 1:]
    else:
        print('Shape of q&t not correct!')

def comp_CE(pred, y_dist36bin,mask):
    y_dist36bin = y_dist36bin.to(pred.device).reshape(-1)
    pred = pred.reshape(-1, 36)
    pred = torch.log(torch.softmax(pred + 1e-4, dim=-1) + 1e-4)
    loss_func = torch.nn.NLLLoss(reduction='none')
    loss = loss_func(pred, y_dist36bin)
    # print(loss.shape)
    mask2 = (mask[:, :, None] * mask[:, None, :]).reshape(-1)
    loss = loss[mask2 > 0.5]
    return loss.mean()

def get_all_atoms(translation, rotation, seq):
    k = translation.shape[1]
    batch_size, nres = translation.shape[0], translation.shape[2]
    C_N_CB = torch.tensor([
        [1.523, -0.518, -0.537],
        [0.,     1.364, -0.769],
        [0.,     0,     -1.208]
    ]).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(translation.device)

    C  = TranslationRotation(rotation, C_N_CB[..., 0].repeat(batch_size, k, nres, nres, 1)) + translation
    N  = TranslationRotation(rotation, C_N_CB[..., 1].repeat(batch_size, k, nres, nres, 1)) + translation
    CB = TranslationRotation(rotation, C_N_CB[..., 2].repeat(batch_size, k, nres, nres, 1)) + translation
    seq = seq.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).to(translation.device)
    CB = torch.where(seq == 5, translation, CB)
    return C, N, CB

def comp_all_fape(y_translation, seq, pred_quaternion_step, pred_translation_step, qAll, AncherList,mask):

    device= pred_quaternion_step[0].device
    
    def comp_FAPE(pred, label):
        loss_func = torch.nn.MSELoss(reduction='none')
        loss = torch.sqrt(loss_func(pred, label).sum(-1) + eps)
        return loss

    k = pred_translation_step[0].shape[1]
    fape_translation = y_translation.unsqueeze(1).repeat(1, k, 1, 1, 1, 1)
    FAPE_loss = []
    for s in range(len(pred_translation_step)):
        pred_quaternion = pred_quaternion_step[s].unsqueeze(-2) * (torch.tensor([1., -1, -1, -1])).to(device)
        pred_CA = TranslationRotation(pred_quaternion, pred_translation_step[s].unsqueeze(-3))
        pred_diag = torch.einsum('bkiid->bkid', pred_CA)
        pred_CA = pred_CA - pred_diag[:, :, :, None]
        pred_C, pred_N, pred_CB = get_all_atoms(pred_CA, qAll[s], seq)
        pred = torch.stack([pred_CA, pred_C, pred_N, pred_CB], dim=-2)
        FAPE_loss.append(comp_FAPE(pred, fape_translation))
    FAPE_loss = torch.stack(FAPE_loss)
     
    mask2 = ((mask[:, :, None] * mask[:, None, :])[None, :, None]).repeat(FAPE_loss.shape[0], 1, FAPE_loss.shape[2], 1, 1).reshape(-1)
    FAPE_loss = FAPE_loss.reshape(-1, 4)[mask2 > 0.5]
    
    FAPE_CA_10 = torch.clamp_max(FAPE_loss[..., 0], max=10.).mean()
    FAPE_CA_inf = FAPE_loss[..., 0].mean()
    FAPE_CNCB_10 = torch.clamp_max(FAPE_loss[..., 1:], max=10.).mean()
    FAPE_CNCB_inf = FAPE_loss[..., 1:].mean()

    fape_loss = (FAPE_CA_10 * 0.9 + FAPE_CA_inf * 0.1) * 0.05 + (FAPE_CNCB_10 * 0.9 + FAPE_CNCB_inf * 0.1) * 0.05

    def comp_realFAPE(pred, label,mask):
        label = label.unsqueeze(3) - label.unsqueeze(2)
        pred = pred.unsqueeze(3) - pred.unsqueeze(2)

        loss_func = torch.nn.MSELoss(reduction='none')
        loss = torch.sqrt(loss_func(pred, label).sum(-1) + eps)
        mask2 = ((mask[:, :, None] * mask[:, None, :]))  # [batch ,L,L]
        slect_mask = SelectAncher(mask2, AncherList, SelectAxis=1, BatchAxis=0)  # [batch ,m,L]
        slect_mask = torch.einsum('b m i, b j -> b m i j',slect_mask,mask)
        return loss[slect_mask>0.5].mean()

    # AncherList = torch.LongTensor(AncherList).to(device)
    # print(AncherList.device, y_translation.device)
    y_CA_Ancher = SelectAncher(y_translation, AncherList, SelectAxis=1, BatchAxis=0)[:, :, :, 0].to(device)
    real_fape = comp_realFAPE(pred_translation_step[-1].detach(), y_CA_Ancher,mask)
    
    return fape_loss, real_fape

def comp_LDDT_loss(y_CA, pred_CA, mask):
    def getLDDT(predcadist,truecadist):
        ### predcadist: (N,K,L,L) 由xyzLLL3计算
        ### truecadist: (N,L,L)
        ###
        
        '''
        比较邻居个数
        
        jupyter notebook: /export/disk4/xyg/mixnet/analysis.ipynb

        对于一个残基，考虑序号间隔至少为s的(non local)，并且欧式距离小于15(空间足够接近，存在相互作用)所有残基。
        然后遍历所有残基
        然后计算平均值,如果不取平均值，则可以得到每个残基的lDDT,即一组数据,长度与序列长度同，alphafold2可预测此值
        lDDT分数取值范围：

        D: true distance
        d: predicted distance

        s: minimum sequence separation. lDDT original paper: default s=0
        t: threshold [0.5,1,2,4] the same ones used to compute the GDT-HA score
        R0: inclusion radius,far definition,according to lDDT paper

        Referenece
        0. AlphaFold1 SI
        1. lDDT original paper: doi:10.1093/bioinformatics/btt473
        
        '''
        
        N,K,L,L=predcadist.shape
        truecadist=torch.tile(truecadist[:,None,:,:],(1,K,1,1))
        
        R0=15.0
        maskfar=torch.as_tensor(truecadist<=R0,dtype=torch.float32) # (N,K,L,L)
        
        s=0  #  lDDT original paper: default s=0
        a=torch.arange(L).reshape([1,L]).to(maskfar.device)
        maskLocal=torch.as_tensor(torch.abs(a-a.T)>s,dtype=torch.float32) # (L,L)
        maskLocal=torch.tile(maskLocal[None,None,:,:],(N,K,1,1))
        fenmu=maskLocal*maskfar

        Ratio=0
        t=[0.5,1,2,4] # the same ones used to compute the GDT-HA score
        for t0 in t:
            preserved=torch.as_tensor(torch.abs(truecadist-predcadist)<t0,dtype=torch.float32)
            fenzi=maskLocal*maskfar*preserved
            Ratio+=torch.sum(fenzi,dim=3)/(torch.sum(fenmu,dim=3)+eps)
        lddt=Ratio/4.0  # (N,K,L)  range (0,1]
        return lddt

    def comp_distance(xyz, eps):
        distance = xyz.unsqueeze(-2) - xyz.unsqueeze(-3)
        distance = torch.sqrt((distance * distance).sum(-1) + eps)
        return distance

    # y_CA: [batch, L, 3]   pred_CA: [batch, len(StructureModule), m, L, 3]
    # pLDDT: [len(StructureModule), batch, 1, L, L]
    # pred_CA = pred_CA.detach()
    
    y_ditance = comp_distance(y_CA, eps)              # [batch, L, L]
    pred_distance = comp_distance(pred_CA, eps)       # [batch, len(StructureModule), m, L, L]

    trueLDDT = torch.stack([getLDDT(pred_distance[:, i], y_ditance) for i in range(pred_distance.shape[1])])   # [len(StructureModule), batch, k/m, L]
    # LDDT_loss_best = trueLDDT[-1].mean(dim=2).max(dim=1)[0].mean()
    # LDDT_loss = trueLDDT.mean(dim=(1, 2, 3))                                                              # [len(StructureModule)]
    # LDDT_loss_items = trueLDDT[-1].mean(dim=(1, 2))
    
        # [batch, L, L]



    LDDT_loss = [] 
    LDDT_loss_best = []
    for idx in range(mask.shape[0]):
        trueLDDT_single = trueLDDT[:, idx]
        n, k = trueLDDT_single.shape[:2]                                                                                                
        
        m = mask[idx][None, None, :].repeat(n, k, 1)
        trueLDDT_single = trueLDDT_single[m > 0.5].reshape(n, k, -1)                                                           
        LDDT_loss_best.append(trueLDDT_single[-1].mean(1).max().detach())

        LDDT_loss.append(trueLDDT_single.mean(dim=(1, 2)))




    LDDT_loss = torch.stack(LDDT_loss).mean(0)
    LDDT_loss_best = torch.stack(LDDT_loss_best).mean()

    return 1. - LDDT_loss[-1], LDDT_loss[-1], LDDT_loss_best

def comp_loss_quaternion(pred_quaternion, y_quaternion,mask):
    mask2 = (mask[:, :, None] * mask[:, None, :])[None, :, None].repeat(len(pred_quaternion), 1, pred_quaternion[0].shape[1], 1, 1)
    ret_qAll = []
    for i in range(len(pred_quaternion)):
        q_left = pred_quaternion[i] * (torch.tensor([1., -1, -1, -1]).to(y_quaternion.device))
        q_right = pred_quaternion[i].unsqueeze(-3)
        ret_qAll.append(NormQuaternionMM(q_left.unsqueeze(-2), q_right))
    loss = (torch.stack(ret_qAll) - y_quaternion[None, :, None]).abs().mean(-1)[mask2 > 0.5]
    return loss.mean(), ret_qAll

def comp_loss_af_plddt(pred_plddt, y_plddt,AncherList):
    pred_plddt = pred_plddt.squeeze(1)
    select_pLDDT = SelectAncher(pred_plddt, AncherList, SelectAxis=1, BatchAxis=0)
    y_plddt = y_plddt.unsqueeze(1)
    loss = torch.abs(select_pLDDT - y_plddt)
    # loss_func = torch.nn.MSELoss(reduction='none')
    # loss = loss_func(select_pLDDT, y_plddt)
    return loss.mean()

def comp_loss_Dihedral(pred, y,mask):
    mask = mask[:, :-1] * mask[:, 1:]
    loss = (torch.abs(pred - y).mean(-1))[mask > 0.5]
    return loss.nanmean()


# Confidence-loss formulas ported from:
# /home/clouduser/Cerebra-single/model-esmc_mHC_li_adjust_for_yx/plddt_train_loss.py


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])

def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss

def lddt(
    all_atom_pred_pos: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = all_atom_mask.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_positions[..., None, :]
                - all_atom_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * all_atom_mask
        * permute_final_dims(all_atom_mask, (1, 0))
        * (1.0 - torch.eye(n, device=all_atom_mask.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25
    
    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    ture_score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score,ture_score

def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca

def _select_conf_anchors(embedding,  AncherList, SelectAxis, BatchAxis=None):
# embedding: [..., batch, ..., length, ...]
# AncherList: torch.LongTensor([1, 2, 3]) or torch.LongTensor([[0, 1, 2], [1, 2, 3]])
    # SelectAxis: int
    # BatchAxis:  int
    if AncherList.dim() == 1:
        ret = embedding.index_select(SelectAxis, AncherList)
        return ret
    elif AncherList.dim() == 2 and BatchAxis != None:
        ret = []
        if SelectAxis == 0:
            if BatchAxis != 0:
                embedding = embedding.transpose(0, BatchAxis)
                for i in range(AncherList.shape[0]):
                    ret.append(embedding[i].index_select(BatchAxis - 1, AncherList[i]))
                ret = torch.stack(ret).transpose(0, BatchAxis)
                return ret
            else:
                print('1 Wrong!!!!')
        else:
            embedding = embedding.transpose(0, BatchAxis)
            for i in range(AncherList.shape[0]):
                ret.append(embedding[i].index_select(SelectAxis - 1, AncherList[i]))
            ret = torch.stack(ret).transpose(0, BatchAxis)
            return ret
    else:
        print('2 Wrong!!!!')

def compute_pae_loss(
    logits: torch.Tensor,
    translation: torch.Tensor,
    batch,
    AncherList,
    cutoff: float = 15.0,
    no_bins: int = 64,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    AncherList= AncherList.to(logits.device)
    label = batch['xyz'][:,:,:,0].to(logits.device)     #[b,l,l,3]
    label_ca = _select_conf_anchors(label, AncherList, 1, 0)    #[B,K,L,3]
    sq_diff = torch.sum(
        ((translation) - (label_ca)) ** 2, dim=-1
    )
    sq_diff = sq_diff.detach()                       #[b,k,l]
    boundaries = torch.linspace(
        0, 31, steps=(no_bins - 1), device=logits.device
    )
    boundaries = boundaries ** 2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)
    
    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_bins, no_bins)
    )
    square_mask = (
        batch['mask'].unsqueeze(-1) * batch['mask'].unsqueeze(-2)   #[b,l,l,]
    ).to(logits.device)
    square_mask = _select_conf_anchors(square_mask, AncherList, 1, 0)    #[b,k,l]

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale
    if 'is_resolution3' in batch.keys():
        loss = loss * (batch['is_resolution3'][:,None].to(logits.device))
    # loss = loss * (
    #     (resolution >= min_resolution) & (resolution <= max_resolution)
    # )

    # Average over the batch dimension
    loss = torch.mean(loss)
    return loss

def lddt_loss(
    logits: torch.Tensor,
    translation: torch.Tensor,
    batch,
    AncherList,
    cutoff: float = 15.0,
    no_bins: int = 50,

    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    # n = all_atom_mask.shape[-2]

    all_atom_pred_pos = translation
    all_atom_positions = batch['all_atoms_pos'][:, :, 1, :].to(translation.device)
    all_atom_mask = batch['mask'].unsqueeze(-1).to(translation.device)# keep dim
    all_atom_positions = all_atom_positions[:, None, :, :].repeat(1, all_atom_pred_pos.shape[1], 1, 1)
    all_atom_mask = all_atom_mask[:, None, :, :].repeat(1, all_atom_pred_pos.shape[1], 1, 1)
    # print('all_atoms_pos',all_atom_pred_pos.shape)
    # print('atom_pos_label',all_atom_positions.shape)
    # print('atom_mask',all_atom_mask.shape)
    score,ture_score = lddt(
        all_atom_pred_pos,
        all_atom_positions,
        all_atom_mask,
        cutoff=cutoff,
        eps=eps
    )
    # print('score',score.shape)
    # print(score.mean())
    score = torch.nan_to_num(score, nan=torch.nanmean(score))
    score[score < 0] = 0

    score = score.detach()
    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(
        bin_index, num_classes=no_bins
    )
    # print(logits.sum())
    # print(lddt_ca_one_hot.shape)
    # print(logits.shape)
    # print(lddt_ca_one_hot.sum())
    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    # print('errors',errors.shape)
    all_atom_mask = all_atom_mask.squeeze(-1)           #[b,k,l]
    square_mask = all_atom_mask.unsqueeze(-1) *all_atom_mask.unsqueeze(-2)    #[b,l,l]

    # square_mask = square_mask.unsqueeze(-1)* all_atom_mask[:,None,None,:]          # [b,k,l,l]
    # print('square_mask',square_mask.shape)
    loss = torch.sum(errors * square_mask, dim=-1) / (
        eps + torch.sum(square_mask, dim=-1)
    )

    # Average over the batch dimension
    if 'is_resolution3' in batch.keys():
        loss = loss * (batch['is_resolution3'][:,None,None].to(translation.device))
    loss = torch.mean(loss)
    pred_lddt = compute_plddt(logits)
    # print('pred_lddt',pred_lddt.shape)
    # print('ture_lddt',ture_score)
    return loss,score.mean(),pred_lddt.mean()

def pred_conf_loss(plddt,pae,translation,batch,AncherList):
    AncherList = torch.as_tensor(AncherList, dtype=torch.long, device=plddt.device)
    plddt_loss,ture_lddt,pred_lddt = lddt_loss(plddt,translation,batch,AncherList)
    pae_loss = compute_pae_loss(pae,translation,batch,AncherList)
    loss = plddt_loss + pae_loss
    return loss,ture_lddt,pred_lddt,plddt_loss,pae_loss





def comp_loss_quaternion_new(pred_quaternion, y_quaternion,mask):
    mask2 = (mask[:, :, None] * mask[:, None, :])[None, :, None].repeat(len(pred_quaternion), 1, pred_quaternion[0].shape[1], 1, 1)
    ret_qAll = []
    for i in range(len(pred_quaternion)):
        q_left = pred_quaternion[i] * (torch.tensor([1., -1, -1, -1]).to(y_quaternion.device))
        q_right = pred_quaternion[i].unsqueeze(-3)
        ret_qAll.append(NormQuaternionMM(q_left.unsqueeze(-2), q_right))
    loss = 1- torch.einsum('n b k l L i, n b k l L i-> n b k l L',torch.stack(ret_qAll), y_quaternion[None, :, None])**2
    loss = loss[mask2 > 0.5]
    return loss.mean(), ret_qAll

def side_chain_feats_get(batch,AncherList):
    batch_num = batch['seq'].shape[0]
    device = batch['seq'].device
    batch_extra_feat ={}
    for i in range(batch_num):
        if AncherList.dim() == 1:
            single_ancher_list = AncherList
        elif AncherList.dim() == 2:
            single_ancher_list = AncherList[i]

        single_extra_feat = hu_model_single_protein_extra_feat_get(batch['seq'][i],batch['all_atoms_pos'][i],batch['all_atoms_mask'][i],single_ancher_list,batch['mask'][i])
        for f in single_extra_feat.keys():
            if f in batch_extra_feat:
                tmp = batch_extra_feat[f]
                tmp.append(single_extra_feat[f])
                batch_extra_feat[f] = tmp
            else:
                tmp = [single_extra_feat[f]]
                batch_extra_feat[f] = tmp
    feats_preload = {}
    feats_preload['residue_index'] = batch['residue_index'].unsqueeze(1).repeat(1,len(single_ancher_list),1)

    for k, v in batch_extra_feat.items():
        v = torch.stack(v)
        feats_preload[k] = v
    return feats_preload
def comp_side_chain_loss(outputs, batch, AncherList):
    pred_q = outputs['quaternion'][-1]
    pred_t = outputs['translation'][-1]
    unnorm_angles,angles = outputs['angles']
    aatype = batch['aatype'].to(pred_q.device)
    angles = angles.to(pred_q.device)
    pred_all_atoms_pos = hu_model_pred_to_atom14_pos(pred_q,pred_t,angles,aatype)
    side_chain_fape_loss = comp_single_pdb_sidechain_fape(pred_q,pred_t,pred_all_atoms_pos,batch)
    return side_chain_fape_loss,pred_all_atoms_pos

def chi_angles_loss(outputs,batch):
    unnormalized_angles_sin_cos,angles_sin_cos = outputs['angles']
    aatype = batch['noancher_aatype']
    seq_mask = batch['noancher_seq_mask']
    chi_mask = batch['noancher_torsion_angles_mask'][...,3:]
    chi_angles_sin_cos = batch["noancher_torsion_angles_sin_cos"][..., 3:, :]
    return supervised_chi_loss(angles_sin_cos,unnormalized_angles_sin_cos,aatype,seq_mask,chi_mask,chi_angles_sin_cos)

def comp_structure_loss(outputs, batch, AncherList,output_pos =False):
    """Structure loss with independently optional pLDDT/PAE confidence terms.

    Missing confidence heads are not evaluated; their fixed report slots are zero.
    """
    if "sequence" in batch:
        batch = {key: value.unsqueeze(0).to(outputs["CE"].device)
                 for key, value in structure_loss_labels(batch).items()}
    device = outputs['CE'].device


    seq =  batch['seq']
    mask = batch['mask'].to(device)

    loss_config = OrderedDict()
    loss_config ={
        # 'loss_translation_est': 0.0, 
        # 'loss_quaternion_est': 0.0, 
        'loss_translation_est': 0.01, 
        'loss_quaternion_est': 0.01, 
        'loss_CE': 0.3, 
        'loss_Dihedral': 0.1, 
        'loss_quaternion': 2, 
        'loss_fape': 1, 
        'plddt_loss': 0.1, 
        'LDDT_loss': 0.1, 
        'sidechain_fape_loss':0.2,
        'angles_loss':1,
        'structure_loss':0.05, #0.01,1,
        'pae_loss':0.1
    }
    loss_items = {}
    length = batch['xyz'].shape[1]
    loss_items['loss_translation_est'] = torch.stack(outputs['translation_est'])[:].mean()
    loss_items['loss_quaternion_est']  = torch.stack(outputs['quaternion_est'])[:].mean()
    loss_items['loss_CE']              = comp_CE(outputs['CE'], batch['CB_dist'],mask)
    loss_items['loss_Dihedral']        = comp_loss_Dihedral(outputs['PsiPhi'], batch['psi_phi'].to(device),mask)
    loss_items['loss_quaternion'], qAll= comp_loss_quaternion_new(outputs['quaternion'], batch['quaternion'].to(device),mask)
    loss_items['loss_fape'], real_fape = comp_all_fape(batch['xyz'].to(device), seq, outputs['quaternion'], outputs['translation'], qAll, AncherList,mask)
    loss_items['LDDT_loss'], LDDT, LDDT_loss_best = comp_LDDT_loss(batch['xyz'][:, 0, :, 0].to(device), torch.stack(outputs['translation']).permute(1, 0, 2, 3, 4),mask)
    # Evaluate before side_chain_feats_get replaces batch with sidechain labels.
    plddt_loss = outputs["CE"].new_zeros(())
    pae_loss = outputs["CE"].new_zeros(())
    if "pLDDT" in outputs:
        confidence_anchors = torch.as_tensor(
            AncherList, dtype=torch.long, device=outputs["pLDDT"].device
        )
        plddt_loss, _, _ = lddt_loss(
            outputs["pLDDT"], outputs["translation"][-1], batch, confidence_anchors
        )
    if "pAE" in outputs:
        confidence_anchors = torch.as_tensor(
            AncherList, dtype=torch.long, device=outputs["pAE"].device
        )
        pae_loss = compute_pae_loss(
            outputs["pAE"], outputs["translation"][-1], batch, confidence_anchors
        )
    AncherList = torch.tensor(AncherList,dtype=torch.long,device=batch['xyz'].device)
    batch = side_chain_feats_get(batch,AncherList)
    for k,v in batch.items():
        v= v.to(device)
        batch[k]=v
    # AncherList = torch.tensor(AncherList,dtype=torch.long,device=device)
    loss_items['sidechain_fape_loss'],pred_all_atoms_pos = comp_side_chain_loss(outputs, batch, AncherList)
    loss_items['angles_loss'] = chi_angles_loss(outputs,batch)
    loss_items['structure_loss'] = structure_loss(batch,pred_all_atoms_pos)
    loss_items['plddt_loss'] = plddt_loss
    loss_items['pae_loss'] = pae_loss
    ret_lossitems = []
    print('loss_Dihedral',loss_items['loss_Dihedral'])
    loss_sum = outputs["CE"].new_zeros(1)
    for k, v in loss_items.items():
        ret_lossitems.append(v.detach().cpu().item())
        if torch.isnan(v).any():
            v = torch.zeros(1, requires_grad=True).to(device)
        if torch.isinf(v).any():
            v = torch.zeros(1, requires_grad=True).to(device)
        if v >= 10e4:
            v = torch.zeros(1, requires_grad=True).to(device)
        loss_sum = loss_sum + loss_config[k] * v
        if k in ['sidechain_fape_loss','angles_loss','structure_loss']:
            print(k,v)
    # print('plddt_loss',loss_items['plddt_loss'])
    # print('pae_loss',loss_items['pae_loss'])
    # if v >= 4:
        #     v = torch.zeros(1, requires_grad=True).to(device)
        # v = torch.clip(v, max=5)
        
        
    
    loss = loss_sum * np.sqrt(length)

    ret_lossitems.append(loss_sum.detach().cpu().item())
    ret_lossitems.append(loss.detach().cpu().item())
    ret_lossitems.append(real_fape.detach().cpu().item())
    ret_lossitems.append(LDDT.detach().cpu().item())
    ret_lossitems.append(LDDT_loss_best.detach().cpu().item())
    # ret_lossitems.append(pred_lddt.detach().cpu().item())
    ret_lossitems.append(loss_items['sidechain_fape_loss'].detach().cpu().item())
    ret_lossitems.append(loss_items['angles_loss'].detach().cpu().item())
    ret_lossitems.append(loss_items['structure_loss'].detach().cpu().item())
    ret_lossitems = np.array(ret_lossitems)
    if output_pos ==True:
        return loss, ret_lossitems,pred_all_atoms_pos,batch
    return loss, ret_lossitems





def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,  # (*, N, 37/14, 3)
    pred_atom_mask: torch.Tensor,  # (*, N, 37/14)
    residue_index: torch.Tensor,  # (*, N)
    aatype: torch.Tensor,  # (*, N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-6,
) -> Dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      aatype: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = aatype[..., 1:] == rc.resname_to_idx["PRO"]
    gt_length = (
        ~next_is_proline
    ) * rc.between_res_bond_length_c_n[
        0
    ] + next_is_proline * rc.between_res_bond_length_c_n[
        1
    ]
    gt_stddev = (
        ~next_is_proline
    ) * rc.between_res_bond_length_stddev_c_n[
        0
    ] + next_is_proline * rc.between_res_bond_length_stddev_c_n[
        1
    ]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    # print('c_n_loss',(mask * c_n_loss_per_residue).shape)
    # print((mask * c_n_loss_per_residue))
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_violation_mask = mask * (
        c_n_bond_length_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )
    

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = rc.between_res_cos_angles_ca_c_n[0]
    gt_stddev = rc.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(
        eps + (ca_c_n_cos_angle - gt_angle) ** 2
    )
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    ca_c_n_violation_mask = mask * (
        ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = rc.between_res_cos_angles_c_n_ca[0]
    gt_stddev = rc.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(
        eps + torch.square(c_n_ca_cos_angle - gt_angle)
    )
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = (
        c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    )
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
        + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
        
    }

def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      atom14_pred_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residue_index: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    fp_type = atom14_pred_positions.dtype

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None]
        < residue_index[..., None, :, None, None]
    )

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(2), num_classes=14
    )
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(0), num_classes=14
    )
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (
        residue_index[..., :, None, None, None] + 1
    ) == residue_index[..., None, :, None, None]
    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = rc.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(
        *((1,) * len(residue_index.shape[:-1])), 1
    ).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, axis=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, axis=(-4, -2)),
        torch.amax(clash_mask, axis=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
    }

def within_residue_violations(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:

    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape
    )
    dists_masks = (
        atom14_atom_exists[..., :, :, None]
        * atom14_atom_exists[..., :, None, :]
        * dists_masks
    )

    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, :, None, :]
                - atom14_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * (
        (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
    )

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0]
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
    }


def find_structural_violations(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations."""

    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
        aatype=batch["aatype"],
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [
        rc.van_der_waals_radius[name[0]]
        for name in rc.atom_types
    ]
    atomtype_radius = atom14_pred_positions.new_tensor(atomtype_radius)
    atom14_atom_radius = (
        batch["atom14_atom_exists"]
        * atomtype_radius[batch["residx_atom14_to_atom37"]]
    )

    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_atom_radius=atom14_atom_radius,
        residue_index=batch["residue_index"],
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = rc.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_atom_exists = batch["atom14_atom_exists"]
    atom14_dists_lower_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[batch["aatype"]]
    atom14_dists_upper_bound = atom14_pred_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[batch["aatype"]]
    residue_violations = within_residue_violations(
        atom14_pred_positions=atom14_pred_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(
                    between_residue_clashes["per_atom_clash_mask"], dim=-1
                )[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],  # ()
            "angles_ca_c_n_loss_mean": connection_violations[
                "ca_c_n_loss_mean"
            ],  # ()
            "angles_c_n_ca_loss_mean": connection_violations[
                "c_n_ca_loss_mean"
            ],  # ()
            "connections_per_residue_loss_sum": connection_violations[
                "per_residue_loss_sum"
            ],  # (N)
            "connections_per_residue_violation_mask": connection_violations[
                "per_residue_violation_mask"
            ],  # (N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # ()
            "clashes_per_atom_loss_sum": between_residue_clashes[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "clashes_per_atom_clash_mask": between_residue_clashes[
                "per_atom_clash_mask"
            ],  # (N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "per_atom_violations": residue_violations[
                "per_atom_violations"
            ],  # (N, 14),
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
    }





def extreme_ca_ca_distance_violations(
    pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
    pred_atom_mask: torch.Tensor,  # (N, 37(14))
    residue_index: torch.Tensor,  # (N)
    max_angstrom_tolerance=1.5,
    eps=1e-6,
) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are
    more than 'max_angstrom_tolerance' apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residue_index: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.
    Returns:
      Fraction of consecutive CA-CA pairs with violation.
    """
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0
    ca_ca_distance = torch.sqrt(
        eps + torch.sum((this_ca_pos - next_ca_pos) ** 2, dim=-1)
    )
    violations = (
        ca_ca_distance - rc.ca_ca
    ) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    mean = masked_mean(mask, violations, -1)
    return mean


def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    ret = {}
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
    )
    ret["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
    ret["violations_between_residue_bond"] = masked_mean(
        batch["seq_mask"],
        violations["between_residues"][
            "connections_per_residue_violation_mask"
        ],
        dim=-1,
    )
    ret["violations_between_residue_clash"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    ret["violations_within_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["within_residues"]["per_atom_violations"], dim=-1
        )[0],
        dim=-1,
    )
    ret["violations_per_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return ret


def violation_loss(
    violations: Dict[str, torch.Tensor],
    atom14_atom_exists: torch.Tensor,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    num_atoms = torch.sum(atom14_atom_exists)
    l_clash = torch.sum(
        violations["between_residues"]["clashes_per_atom_loss_sum"]
        + violations["within_residues"]["per_atom_loss_sum"]
    )
    l_clash = l_clash / (eps + num_atoms)
    loss = (
        violations["between_residues"]["bonds_c_n_loss_mean"]
        + violations["between_residues"]["angles_ca_c_n_loss_mean"]
        + violations["between_residues"]["angles_c_n_ca_loss_mean"]
        + l_clash
    )
    mean = torch.mean(loss)

    return mean

def structure_loss(batch,atom_14_positions):
    violation_config= {
                "violation_tolerance_factor": 12.0,
                "clash_overlap_tolerance": 1.5,
                "eps": eps,  # 1e-6,
                "weight": 0.0,
            }
    violation =  find_structural_violations(batch,atom_14_positions,**violation_config)
    comp_violation_loss = violation_loss(violation,batch["atom14_atom_exists"])
    return comp_violation_loss


def find_structural_violations_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    config: ml_collections.ConfigDict,
) -> Dict[str, np.ndarray]:
    to_tensor = lambda x: torch.tensor(x)
    batch = tree_map(to_tensor, batch, np.ndarray)
    atom14_pred_positions = to_tensor(atom14_pred_positions)

    out = find_structural_violations(batch, atom14_pred_positions, **config)

    to_np = lambda x: np.array(x)
    np_out = tensor_tree_map(to_np, out)

    return np_out

def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    atom14_pred_positions: torch.Tensor,  # (N, 14, 3)
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    ret = {}
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=atom14_pred_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residue_index=batch["residue_index"],
    )
    ret["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
    ret["violations_between_residue_bond"] = masked_mean(
        batch["seq_mask"],
        violations["between_residues"][
            "connections_per_residue_violation_mask"
        ],
        dim=-1,
    )
    ret["violations_between_residue_clash"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    ret["violations_within_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["within_residues"]["per_atom_violations"], dim=-1
        )[0],
        dim=-1,
    )
    ret["violations_per_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return ret


def compute_violation_metrics_np(
    batch: Dict[str, np.ndarray],
    atom14_pred_positions: np.ndarray,
    violations: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    to_tensor = lambda x: torch.tensor(x)
    batch = tree_map(to_tensor, batch, np.ndarray)
    atom14_pred_positions = to_tensor(atom14_pred_positions)
    violations = tree_map(to_tensor, violations, np.ndarray)

    out = compute_violation_metrics(batch, atom14_pred_positions, violations)

    to_np = lambda x: np.array(x)
    return tree_map(to_np, out, torch.Tensor)