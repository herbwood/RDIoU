import numpy as np
import torch
import torch.nn.functional as F

from typing import List


class AnchorGenerator():
    """default anchor generator for fpn.
    This class generate anchors by feature map in level.
    """
    def __init__(self, 
                base_size : int = 16, 
                ratios : List = [0.5, 1, 2],
                base_scale : int = 2):

        self.base_size = base_size
        self.base_scale = np.array(base_scale)
        self.anchor_ratios = np.array(ratios)


    def _whctrs(self, anchor):
        """convert anchor box into (w, h, ctr_x, ctr_y)
        """

        w = anchor[:, 2] - anchor[:, 0] + 1
        h = anchor[:, 3] - anchor[:, 1] + 1
        x_ctr = anchor[:, 0] + 0.5 * (w - 1)
        y_ctr = anchor[:, 1] + 0.5 * (h - 1)

        return w, h, x_ctr, y_ctr


    def get_plane_anchors(self, 
                          anchor_scales: np.ndarray):
        """get anchors per location on feature map.
        The anchor number is anchor_scales x anchor_ratios
        """

        # base_anchor : [0, 0, 15, 15]
        # anchor 생성
        # anchor의 width, height, center x, center y 
        base_anchor = np.array([[0, 0, self.base_size - 1, self.base_size - 1]])
        off = self.base_size // 2 - 8 # 0
        w, h, x_ctr, y_ctr = self._whctrs(base_anchor)

        # ratio enumerate
        # width scale, height scale 
        size = w * h
        size_ratios = size / self.anchor_ratios
        ws = np.round(np.sqrt(size_ratios))
        hs = np.round(ws * self.anchor_ratios)

        # anchor의 width, height에 따른 scale, aspect ratio
        anchor_scales = anchor_scales[None, ...]
        ws = (ws[:, None] * anchor_scales).reshape(-1, 1)
        hs = (hs[:, None] * anchor_scales).reshape(-1, 1)

        # make anchors
        anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1))) - off

        return anchors.astype(np.float32)


    def get_center_offsets(self, fm_map, stride):
        
        # feature map의 height, width 
        fm_height, fm_width = fm_map.shape[-2], fm_map.shape[-1]
        f_device = fm_map.device

        # 중심점 지점의 위치를 구함 
        """
        tensor([ 0,  2,  4,  6,  8, 10])
        tensor([ 0,  2,  4,  6,  8, 10, 12, 14])
        tensor([[ 0,  2,  4,  6,  8, 10]])      
        tensor([[ 0],
                [ 2],
                [ 4],
                [ 6],
                [ 8],
                [10],
                [12],
                [14]])
        tensor([[ 0,  2,  4,  6,  8, 10],       
                [ 0,  2,  4,  6,  8, 10],       
                [ 0,  2,  4,  6,  8, 10],       
                [ 0,  2,  4,  6,  8, 10],       
                [ 0,  2,  4,  6,  8, 10],       
                [ 0,  2,  4,  6,  8, 10],       
                [ 0,  2,  4,  6,  8, 10],       
                [ 0,  2,  4,  6,  8, 10]])
        tensor([[ 0,  0,  0,  0,  0,  0],
                [ 2,  2,  2,  2,  2,  2],
                [ 4,  4,  4,  4,  4,  4],
                [ 6,  6,  6,  6,  6,  6],
                [ 8,  8,  8,  8,  8,  8],
                [10, 10, 10, 10, 10, 10],
                [12, 12, 12, 12, 12, 12],
                [14, 14, 14, 14, 14, 14]])
        """
        shift_x = torch.arange(0, fm_width, device=f_device) * stride
        shift_y = torch.arange(0, fm_height, device=f_device) * stride
        broad_shift_x = shift_x.reshape(-1, shift_x.shape[0]).repeat(fm_height,1)
        broad_shift_y = shift_y.reshape(shift_y.shape[0], -1).repeat(1,fm_width)

        # flatten
        flatten_shift_x = broad_shift_x.flatten().reshape(-1,1)
        flatten_shift_y = broad_shift_y.flatten().reshape(-1,1)

        # concat하여 얼마만큼 이동하면 되는지에 대한 정보 출력 
        # shifts shape : (feature map height x width, 4)
        shifts = torch.cat(
            [flatten_shift_x, flatten_shift_y, flatten_shift_x, flatten_shift_y],
            axis=1)

        return shifts 
        


    def get_anchors_by_feature(self, fm_map, base_stride, off_stride):

        # shifts shape: [A, 4]
        shifts = self.get_center_offsets(fm_map, base_stride * off_stride)

        # plane_anchors shape: [B, 4], e.g. B=3
        plane_anchors = self.get_plane_anchors(self.base_scale * off_stride)
        plane_anchors = torch.tensor(plane_anchors, device=fm_map.device)

        # all_anchors shape: [A, B, 4]
        all_anchors = plane_anchors[None, :] + shifts[:, None]
        all_anchors = all_anchors.reshape(-1, 4)
        
        # all_anchors shape : (# of anchors, 4)
        return all_anchors


    @torch.no_grad()
    def __call__(self, featmap, base_stride, off_stride):
        return self.get_anchors_by_feature(featmap, base_stride, off_stride)