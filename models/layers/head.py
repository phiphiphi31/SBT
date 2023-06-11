import torch.nn as nn
import torch
import torch.nn.functional as F

from lib.models.layers.frozen_bn import FrozenBatchNorm2d


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        if BN:
            self.layers = nn.ModuleList(nn.Sequential(nn.Linear(n, k), nn.BatchNorm1d(k))
                                        for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = nn.ModuleList(nn.Linear(n, k)
                                        for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MLP_head(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class MLP_mixer(nn.Module):
    """ mixer, modeling channel, spatial """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, hw= 32*32):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers_channle = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.layers_spatial = nn.ModuleList(nn.Linear(n, k) for n, k in zip([hw] + h, h + [hw]))

    def forward(self, x):
        for i in range(self.num_layers):
            x = x.permute(0, 1, 3, 2)
            x = F.relu(self.layers_spatial[i](x))
            x = x.permute(0, 1, 3, 2)
            x = F.relu(self.layers_channle[i](x)) if i < self.num_layers - 1 else self.layers_channle[i](x)
        return x



class Corner_Predictor(nn.Module):
    """ Corner Predictor module"""

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)

        '''about coordinates and indexs'''
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)) \
                .view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x, return_dist=False, softmax=True):
        """ Forward pass with input x. """
        score_map_tl, score_map_br = self.get_score_map(x)
        if return_dist:
            coorx_tl, coory_tl, prob_vec_tl = self.soft_argmax(score_map_tl, return_dist=True, softmax=softmax)
            coorx_br, coory_br, prob_vec_br = self.soft_argmax(score_map_br, return_dist=True, softmax=softmax)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz, prob_vec_tl, prob_vec_br
        else:
            coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
            coorx_br, coory_br = self.soft_argmax(score_map_br)
            return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map, return_dist=False, softmax=True):
        """ get soft-argmax coordinate for a given heatmap """
        score_vec = score_map.view((-1, self.feat_sz * self.feat_sz))  # (batch, feat_sz * feat_sz)
        prob_vec = nn.functional.softmax(score_vec, dim=1)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        if return_dist:
            if softmax:
                return exp_x, exp_y, prob_vec
            else:
                return exp_x, exp_y, score_vec
        else:
            return exp_x, exp_y


class CenterPredictor(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(CenterPredictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_ctr2 = self.conv2_ctr(x_ctr1)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)
        x_offset2 = self.conv2_offset(x_offset1)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)
        x_size2 = self.conv2_size(x_size1)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

class CenterPredictor_mlp(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, num_layers= 3, freeze_bn=False):
        super(CenterPredictor_mlp, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        """
        class MLP(nn.Module):
           def __init__(self, input_dim, hidden_dim, output_dim, num_layers, BN=False):
        """
        self.conv1_ctr = MLP(input_dim=inplanes, hidden_dim=inplanes//4, output_dim=1, num_layers=num_layers)
        self.conv1_offset = MLP(input_dim=inplanes, hidden_dim=inplanes//4, output_dim=2, num_layers=num_layers)
        self.conv1_size = MLP(input_dim=inplanes, hidden_dim=inplanes//4, output_dim=2, num_layers=num_layers)


        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y
        """
        B, C, H, W = x_adjust.shape
        x_track_final = x_adjust  # # [1, 32*32, 256]
        x_track_final_reshape = x_track_final.reshape(B, C, -1).permute(2, 0, 1)  # [HW, B, C]
        reg_cls_feat = x_track_final_reshape.unsqueeze(0).transpose(1, 2)  # [1, HW, B, C] -> [1, B, HW, C]

        outputs_class = self.class_head(reg_cls_feat)  # [1, 1, 32*32, 2]
        """
        B, C, H, W = x.shape
        x_track_final_reshape = x.reshape(B, C, -1).permute(2, 0, 1)  # [HW, B, C]
        reg_cls_feat = x_track_final_reshape.unsqueeze(0).transpose(1, 2)  # [1, HW, B, C] -> [1, B, HW, C]


        # ctr branch
        score_map_ctr = self.conv1_ctr(reg_cls_feat)  # [1, B, hw, 1]
        score_map_ctr = score_map_ctr.squeeze(0) # [ B, hw, 1]
        score_map_ctr = score_map_ctr.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # [ B, c, h, w]

        # offset branch)
        score_map_offset = self.conv1_offset(reg_cls_feat)  # [1, 1, hw, 2]
        score_map_offset = score_map_offset.squeeze(0)  # [ B, hw, 1]
        score_map_offset = score_map_offset.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [ B, c, h, w]

        # size branch
        score_map_size = self.conv1_size(reg_cls_feat)  # [1, 1, hw, 2]
        score_map_size = score_map_size.squeeze(0)  # [ B, hw, 1]
        score_map_size = score_map_size.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [ B, c, h, w]


        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset

class CenterPredictor_large(nn.Module, ):
    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False, increased_layer=4):
        super(CenterPredictor_large, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride

        # corner predict
        self.conv1_ctr = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv1_ctr_increase = nn.Sequential(*[conv(channel, channel, freeze_bn=freeze_bn)  for i in range(increased_layer)])

        self.conv2_ctr = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_ctr = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_ctr = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_ctr = nn.Conv2d(channel // 8, 1, kernel_size=1)

        # size regress
        self.conv1_offset = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv1_offset_increase =  nn.Sequential(*[conv(channel, channel, freeze_bn=freeze_bn)  for i in range(increased_layer)])

        self.conv2_offset = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_offset = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_offset = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_offset = nn.Conv2d(channel // 8, 2, kernel_size=1)

        # size regress
        self.conv1_size = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv1_size_increase = nn.Sequential(*[conv(channel, channel, freeze_bn=freeze_bn)  for i in range(increased_layer)])

        self.conv2_size = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_size = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_size = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_size = nn.Conv2d(channel // 8, 2, kernel_size=1)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, gt_score_map=None):
        """ Forward pass with input x. """
        score_map_ctr, size_map, offset_map = self.get_score_map(x)

        # assert gt_score_map is None
        if gt_score_map is None:
            bbox = self.cal_bbox(score_map_ctr, size_map, offset_map)
        else:
            bbox = self.cal_bbox(gt_score_map.unsqueeze(1), size_map, offset_map)

        return score_map_ctr, bbox, size_map, offset_map

    def cal_bbox(self, score_map_ctr, size_map, offset_map, return_score=False):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        # cx, cy, w, h
        bbox = torch.cat([(idx_x.to(torch.float) + offset[:, :1]) / self.feat_sz,
                          (idx_y.to(torch.float) + offset[:, 1:]) / self.feat_sz,
                          size.squeeze(-1)], dim=1)

        if return_score:
            return bbox, max_score
        return bbox

    def get_pred(self, score_map_ctr, size_map, offset_map):
        max_score, idx = torch.max(score_map_ctr.flatten(1), dim=1, keepdim=True)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz

        idx = idx.unsqueeze(1).expand(idx.shape[0], 2, 1)
        size = size_map.flatten(2).gather(dim=2, index=idx)
        offset = offset_map.flatten(2).gather(dim=2, index=idx).squeeze(-1)

        # bbox = torch.cat([idx_x - size[:, 0] / 2, idx_y - size[:, 1] / 2,
        #                   idx_x + size[:, 0] / 2, idx_y + size[:, 1] / 2], dim=1) / self.feat_sz
        return size * self.feat_sz, offset

    def get_score_map(self, x):

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        # ctr branch
        x_ctr1 = self.conv1_ctr(x)
        x_incre = x_ctr1
        for i, layer_incre in enumerate(self.conv1_ctr_increase):
            x_conv = layer_incre(x_incre)
            x_incre = x_incre + x_conv

        x_ctr2 = self.conv2_ctr(x_incre)
        x_ctr3 = self.conv3_ctr(x_ctr2)
        x_ctr4 = self.conv4_ctr(x_ctr3)
        score_map_ctr = self.conv5_ctr(x_ctr4)

        # offset branch
        x_offset1 = self.conv1_offset(x)

        x_incre = x_offset1
        for i, layer_incre in enumerate(self.conv1_offset_increase):
            x_conv = layer_incre(x_incre)
            x_incre = x_incre + x_conv

        x_offset2 = self.conv2_offset(x_incre)
        x_offset3 = self.conv3_offset(x_offset2)
        x_offset4 = self.conv4_offset(x_offset3)
        score_map_offset = self.conv5_offset(x_offset4)

        # size branch
        x_size1 = self.conv1_size(x)

        x_incre = x_size1
        for i, layer_incre in enumerate(self.conv1_size_increase):
            x_conv = layer_incre(x_incre)
            x_incre = x_incre + x_conv

        x_size2 = self.conv2_size(x_incre)
        x_size3 = self.conv3_size(x_size2)
        x_size4 = self.conv4_size(x_size3)
        score_map_size = self.conv5_size(x_size4)
        return _sigmoid(score_map_ctr), _sigmoid(score_map_size), score_map_offset



class TransTboxPredictor(nn.Module):
    def __init__(self, input_dim=320, adjust_dim=None, mixer_hw=None):
        super(TransTboxPredictor, self).__init__()
        if adjust_dim is not None:
            self.adjust_head = nn.Conv2d(input_dim, adjust_dim, kernel_size=1)
        else:
            self.adjust_head = None
        self.TransTboxPredictor=True


        # TransT box predict
        if mixer_hw is not None:
            # input_dim, hidden_dim, output_dim, num_layers, hw= 32*32
            self.class_head = MLP_mixer(input_dim, input_dim, output_dim= 2, num_layers=2, hw = mixer_hw)
            self.bbox_head = MLP_mixer(input_dim, input_dim, output_dim= 4, num_layers=2, hw = mixer_hw)
        else:
            self.class_head = MLP_head(input_dim, input_dim, 1 + 1, 2)
            self.bbox_head = MLP_head(input_dim, input_dim, 4, 2)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        if self.adjust_head is None:
            x_adjust = x
        else:
            x_adjust = self.adjust_head(x)

        B, C, H, W = x_adjust.shape
        x_track_final = x_adjust  # # [1, 32*32, 256]
        x_track_final_reshape = x_track_final.reshape(B, C, -1).permute(2, 0, 1)  # [HW, B, C]
        reg_cls_feat = x_track_final_reshape.unsqueeze(0).transpose(1, 2)  # [1, HW, B, C] -> [1, B, HW, C]

        outputs_class = self.class_head(reg_cls_feat)  # [1, 1, 32*32, 2]
        outputs_coord = self.bbox_head(reg_cls_feat).sigmoid()  # [1, 1, 32*32, 4]
        return outputs_class[-1],  outputs_coord[-1]

    def cal_bbox(self, pred_logits, pred_boxes):
        pred_logits = pred_logits.permute(2, 1, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(pred_logits, dim=1).data[:, 0].cpu().numpy()

        pred_boxes = pred_boxes.permute(2, 1, 0).contiguous().view(4, -1)
        delta = pred_boxes.data.cpu().numpy()

        return score, delta




def build_box_head(cfg, hidden_dim):
    stride = cfg.MODEL.BACKBONE.STRIDE

    if cfg.MODEL.HEAD_TYPE == "MLP_mixer":
       # mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        mlp__mixer_head = TransTboxPredictor(input_dim=hidden_dim, adjust_dim=cfg.MODEL.ADJUST_DIM, mixer_hw= cfg.MODEL.HEAD_MIXER_HW)
        return mlp__mixer_head
    elif "CORNER" in cfg.MODEL.HEAD_TYPE:
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        channel = getattr(cfg.MODEL, "NUM_CHANNELS", 256)
        print("head channel: %d" % channel)
        if cfg.MODEL.HEAD.TYPE == "CORNER":
            corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=channel,
                                           feat_sz=feat_sz, stride=stride)
        else:
            raise ValueError()
        return corner_head
    elif cfg.MODEL.HEAD_TYPE == "CENTER":
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor(inplanes=in_channel, channel=out_channel, feat_sz=feat_sz, stride=stride)
        return center_head

    elif cfg.MODEL.HEAD_TYPE == "CENTER_MLP":
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        num_layers = cfg.MODEL.HEAD.NUM_LAYERS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor_mlp(inplanes=in_channel, channel=out_channel, feat_sz=feat_sz, stride=stride, num_layers= num_layers)
        return center_head
    elif cfg.MODEL.HEAD_TYPE == "CENTER_large":
        in_channel = hidden_dim
        out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        num_layers = cfg.MODEL.HEAD.NUM_LAYERS
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        center_head = CenterPredictor_large(inplanes=in_channel, channel=out_channel, feat_sz=feat_sz, stride=stride)
        return center_head

    elif cfg.MODEL.HEAD_TYPE == "TransT_box_prediction":
        in_channel = hidden_dim
        #out_channel = cfg.MODEL.HEAD.NUM_CHANNELS
        TransT_box_prediction_head = TransTboxPredictor(input_dim=in_channel, adjust_dim=cfg.MODEL.HEAD.ADJUST_DIM, mixer_hw= cfg.MODEL.HEAD.MIXER_HW)
        # input_dim=764, adjust_dim=256 input_dim=320, adjust_dim=None, mixer_hw=None
        return TransT_box_prediction_head

    else:
        raise ValueError("HEAD TYPE %s is not supported." % cfg.MODEL.HEAD_TYPE)
