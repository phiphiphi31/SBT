import torch
import torch.nn as nn

import torch.nn.functional as F
from util import box_ops
import time
import numpy as np

class TrackingMatcher(nn.Module):
    """This class computes an assignment between the ground-truth and the predictions of the network.
    The corresponding feature vectors within the ground-truth box are matched as positive samples.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Always tensor([0]) represents the foreground,
                           since single target tracking has only one foreground category
                 "boxes": Tensor of dim [1, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order),
                  and it is always 0, because single target tracking has only one target per image
            For each batch element, it holds:
                len(index_i) = len(index_j)
        """
        indices = []
        #bs, num_queries = outputs["pred_logits"].shape[:2]
        for k, v in outputs.items():
            bs, num_queries = outputs[k].shape[:2]
        neg_flag = []
        for i in range(bs):
            cx, cy, w, h = targets[i]['boxes'][0]
            cx = cx.item(); cy = cy.item(); w = w.item(); h = h.item()
            xmin = cx-w/2; ymin = cy-h/2; xmax = cx+w/2; ymax = cy+h/2
            len_feature = int(np.sqrt(num_queries))
            Xmin = int(np.ceil(xmin*len_feature))
            Ymin = int(np.ceil(ymin*len_feature))
            Xmax = int(np.ceil(xmax*len_feature))
            Ymax = int(np.ceil(ymax*len_feature))
            if Xmin == Xmax:
                Xmax = Xmax+1
            if Ymin == Ymax:
                Ymax = Ymax+1
            a = np.arange(0, num_queries, 1)
            b = a.reshape([len_feature, len_feature])
            c = b[Ymin:Ymax,Xmin:Xmax].flatten()
            d = np.zeros(len(c), dtype=int)
            indice = (c,d)
            indices.append(indice)
            if w == 0 or h ==0:
                neg_flag.append(torch.as_tensor(i, dtype=torch.int64))
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices], neg_flag

def build_matcher():
    return TrackingMatcher()


class SetCriterion(nn.Module):
    """ This class computes the loss for TransT.
    The process happens in two steps:
        1) we compute assignment between ground truth box and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, always be 1 for single object tracking.
            matcher: module able to compute a matching between target and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, neg_flag, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        if len(neg_flag) > 0:
            a = idx[0].cuda(device = target_classes_o.device)
            for n in neg_flag:
                n.cuda(device=target_classes_o.device)
                target_classes_o = torch.where(a== n, torch.ones(1, device=target_classes_o.device,dtype=torch.int64), target_classes_o)

        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)  # [bs, wh, 2]->transpose
        losses = {'loss_ce': loss_ce}
        #src_logits.transpose(1, 2): [bs, 2, hw], target_classes: [b, hw]
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            #losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            pass

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes, neg_flag):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes))
        giou = torch.diag(giou)
        iou = torch.diag(iou)
        loss_giou = 1 - giou
        iou = iou
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        losses['iou'] = iou.sum() / num_boxes
        return losses

    def loss_iou(self, outputs, targets, indices, num_boxes, neg_flag, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        src_logits = outputs['pred_iou'].squeeze()  # [bs, hw, 1]
        pred_bbox = outputs['best_bbox']  # [bs, 4]
        gt_boxes =torch.cat([t['boxes'] for t in targets]) # [bs, 4]

        giou, iou = box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(pred_bbox),
            box_ops.box_cxcywh_to_xyxy(gt_boxes))
        iou = torch.diag(iou) # [bs]

        idx = self._get_src_permutation_idx(indices) # (ba, index)
        target_classes_o = torch.cat([t["labels"][J] + iou_bs for iou_bs,  t, (_, J) in zip(iou, targets, indices)])

        src_boxes = src_logits[idx]

        loss_l1_iouPredict = F.l1_loss(src_boxes, target_classes_o, reduction='none')  # [bs wh]
        loss_l1_iouPredict_avg = loss_l1_iouPredict.sum() / num_boxes

        losses = {'loss_l1_iou_predict': loss_l1_iouPredict_avg}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            # losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            pass

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, neg_flag):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'iou': self.loss_iou,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, neg_flag)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the target
        indices, neg_flag = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes_pos = sum(len(t[0]) for t in indices)

        num_boxes_pos = torch.as_tensor([num_boxes_pos], dtype=torch.float, device=next(iter(outputs.values())).device)

        num_boxes_pos = torch.clamp(num_boxes_pos, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes_pos, neg_flag))

        return losses


def block_ms_loss(settings, weight_dict, type = 'reg'):
    if type=='reg':

        if weight_dict is None: weight_dict = {'loss_bbox': 70, 'loss_giou': 50}

        num_classes = 1
        matcher = build_matcher()
        losses = ['boxes']
        criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=0.0625, losses=losses)
        device = torch.device(settings.device)
        criterion.to(device)

    elif type == 'cls':
        if weight_dict is None: weight_dict = {'loss_ce': 120}

        num_classes = 1
        matcher = build_matcher()
        losses = ['labels']
        criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=0.0625, losses=losses)
        device = torch.device(settings.device)
        criterion.to(device)

    elif type == 'iou':
        if weight_dict is None: weight_dict = {'loss_l1_iou_predict': 120}

        num_classes = 1
        matcher = build_matcher()
        losses = ['iou']
        criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                                 eos_coef=0.0625, losses=losses)
        device = torch.device(settings.device)
        criterion.to(device)


    return criterion