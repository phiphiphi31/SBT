from lib.test.tracker.basetracker import BaseTracker
import torch
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
from lib.utils.merge import merge_template_search
from lib.models.sbt.SBTtrackmodel import build_SBTtrack
from lib.test.tracker.stark_utils import Preprocessor
from lib.utils.box_ops import clip_box

import numpy as np

class SBT(BaseTracker):
    def __init__(self, params, dataset_name):
        super(SBT, self).__init__(params)
        network = build_SBTtrack(params.cfg, training=False)
        pth= torch.load(self.params.checkpoint, map_location='cpu')
        network.load_state_dict(pth['net'], strict=False)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        # for debug
        self.debug = False
        self.frame_id = 0
        if self.debug:
            self.save_dir = "debug"
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        hanning = np.hanning(self.feat_sz)  # 14
        window = np.outer(hanning, hanning)
        self.output_window = window.flatten()

    def initialize(self, image, info: dict):
        # forward the template once
        z_patch_arr, _, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = self.network.template(template.tensors)
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)
        with torch.no_grad():
            out_dict = self.network.forward(search= search.tensors, template=None)
            # merge the template and the search



        if 'TransT' in getattr(self.network, 'head_type') or 'MLP' in getattr(self.network, 'head_type'):
            # convert score and bbox
            score, pred_bbox = self.network.box_head.cal_bbox(pred_logits = out_dict['score_map'], pred_boxes = out_dict['pred_boxes'])

            # window penalty
            self.window_penalty = 0.49
            pscore = score * (1 - self.window_penalty) + self.output_window * self.window_penalty
            # find the best bbox and convert it to original coord
            bbox_best = self.map_box_back_transt(pscore, pred_bbox, resize_factor)
            # clip boundary
            self.state = clip_box(bbox_best, H, W, margin=10)
        else:
            pred_boxes = out_dict['pred_boxes'].view(-1, 4)
            # Baseline: Take the mean of all pred boxes as the final result
            pred_box = (pred_boxes.mean(
                dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
            # get the final box result
            self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def map_box_back_transt(self, score, pred_bbox, resize_factor):
        #cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        #cx, cy, w, h = pred_box
        #half_side = 0.5 * self.params.search_size / resize_factor
       # cx_real = cx + (cx_prev - half_side)
        #cy_real = cy + (cy_prev - half_side)

        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        half_side = 0.5 * self.params.search_size / resize_factor

        """
        w_x = self.size[0] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        h_x = self.size[1] + (4 - 1) * ((self.size[0] + self.size[1]) * 0.5)
        s_x = math.ceil(math.sqrt(w_x * h_x))
        """
        best_idx = np.argmax(score)
        bbox = pred_bbox[:, best_idx]
        bbox = bbox * (self.params.search_size / resize_factor)
        cx = bbox[0] + cx_prev - half_side
        cy = bbox[1] + cy_prev - half_side
        width = bbox[2]
        height = bbox[3]

        return [cx - 0.5 * width, cy - 0.5 * height, width, height]

def get_tracker_class():
    return SBT
