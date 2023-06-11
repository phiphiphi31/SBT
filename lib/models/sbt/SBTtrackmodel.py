"""
Basic SBT1 tracking model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
import lib.models.sbt.SBTv2model as SBTv2


class SBTtrack(nn.Module):
    """ This is the base class for SBTtrack """

    def __init__(self, visionTransformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            visionTransformer: torch module of the vision transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = visionTransformer
        self.box_head = box_head
        self.aux_loss = aux_loss

        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self,  search: torch.Tensor, template=None):
        if template is None:
            template = self.template_z
        out_dict_vit = self.backbone( x=search, z=template)
        # Forward head
        feat_x = out_dict_vit['x']

        out_prediction = self.forward_head(feat_x, None)
        out_prediction['backbone_feat'] = feat_x
        return out_prediction

    def forward_head(self, x_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = x_feature #cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        H = W =  int(HW**0.5)
        opt_feat = opt.view(-1, C, H, W)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}

        elif self.head_type == "TransT_box_prediction":
            # run the TransT box prediction head
            outputs_class, outputs_coord = self.box_head(opt_feat)  # [1, 1, hw, 2]

            out = {'score_map': outputs_class, 'pred_boxes': outputs_coord}

        elif self.head_type == "MLP_mixer":
            # run the TransT box prediction head
            outputs_class, outputs_coord = self.box_head(opt_feat)  # [1, 1, hw, 2]

            out = {'score_map': outputs_class, 'pred_boxes': outputs_coord}
        else:
            raise NotImplementedError

        return out

    def template(self, z):
        self.template_z = z
        return  self.template_z


def build_SBTtrack(cfg, training=True):

    if cfg.MODEL.BACKBONE.TYPE == 'build_sbtv2_base_model':
        backbone = SBTv2.build_sbtv2_base_model()
        hidden_dim = backbone.embed_dim
    elif cfg.MODEL.BACKBONE.TYPE == 'build_sbtv2_large_model':
        backbone = SBTv2.build_sbtv2_large_model()
        hidden_dim = backbone.embed_dim
    elif cfg.MODEL.BACKBONE.TYPE == 'build_sbtv2_huge_model':
        backbone = SBTv2.build_sbtv2_huge_model()
        hidden_dim = backbone.embed_dim
    else:
        raise NotImplementedError

    box_head = build_box_head(cfg, hidden_dim)
    model = SBTtrack(backbone, box_head, head_type=cfg.MODEL.HEAD_TYPE)

    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if training:
       if cfg.MODEL.BACKBONE.WEIGHT:
          pretrained = os.path.join(pretrained_path, cfg.MODEL.BACKBONE.WEIGHT)
          checkpoint = torch.load(pretrained, map_location="cpu")
          print('load pretrained weight only for vision transformer backbone')
          missing_keys, unexpected_keys = model.backbone.load_state_dict(checkpoint, strict=False)
          print('missing_keys: ' + str(missing_keys))
          print('unexpected_keys: ' + str(unexpected_keys))
          print('Load pretrained model from: ' + cfg.MODEL.BACKBONE.WEIGHT)
       else:
          print('No pretrained weights are loaded !!!')

    return model
