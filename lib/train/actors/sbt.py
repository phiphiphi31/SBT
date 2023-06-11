from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
import numpy as np

class SBTActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        if len(data['search_images'].shape) ==5:
            data['search_images'] = data['search_images'][0]
            data['search_anno'] = data['search_anno'][0]
            data['search_masks'] = data['search_masks'][0]

            data['template_images'] = data['template_images'][0]
            data['template_anno'] = data['template_anno'][0]
            data['template_masks'] = data['template_masks'][0]

        out_dict = self.forward_pass(data)
        # generate labels
        target_labels = self.generate_labels(data)
        # compute losses
        loss, status = self.compute_losses(out_dict, target_labels)

        return loss, status

    def forward_pass(self, data):

        out_dict = self.net(template=data['template_images'], search=data['search_images'])

        return out_dict

    def generate_labels(self, data):
        # generate labels
        #if data['search_anno'].min() <0:
         #   print(str(data['search_anno'].min()))
        #if data['template_anno'].min() < 0:
         #   print(str(data['template_anno'].min()))

        targets = []
        targets_origin = data['search_anno']
        for i in range(len(targets_origin)):
            h, w = data['search_images'][i][0].shape
            target_origin = targets_origin[i]
            target = {}
            target_origin = target_origin.reshape([1, -1])
            target_origin[0][0] += target_origin[0][2] / 2
            #target_origin[0][0] /= w
            target_origin[0][1] += target_origin[0][3] / 2
            #target_origin[0][1] /= h
            #target_origin[0][2] /= w
            #target_origin[0][3] /= h
            target['boxes'] = target_origin
            label = np.array([0])
            label = torch.tensor(label, device=data['search_anno'].device)
            target['labels'] = label
            targets.append(target)

        return targets

    def compute_losses(self, outputs, targets):
        # Compute loss
        # outputs:(center_x, center_y, width, height)
        if isinstance(self.objective, dict):
            loss_dict = self.objective['tracking'](outputs, targets)
            weight_dict = self.objective['tracking'].weight_dict

        else:
            loss_dict = self.objective(outputs, targets)
            weight_dict = self.objective.weight_dict

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # Return training stats
        stats = {'Loss/total': losses.item(),
                 'Loss/ce': loss_dict['loss_ce'].item(),
                 'Loss/bbox': loss_dict['loss_bbox'].item(),
                 'Loss/giou': loss_dict['loss_giou'].item(),
                 'iou': loss_dict['iou'].item()
                 }

        return losses, stats

    def compute_losses1(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss