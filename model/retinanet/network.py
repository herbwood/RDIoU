import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import sys

from config import config
from backbone.resnet50 import ResNet50
from backbone.fpn import FPN
from backbone.bfp import BFP
from det_oprs.anchor_generator import AnchorGenerator
from det_oprs.retinanet_anchor_target import concated_anchor_target
from det_oprs.bbox_opr import bbox_transform_inv_opr
from det_oprs.loss_opr import emd_loss_repulsion_diou, emd_loss_focal, emd_loss_simple_diou, multi_task_loss
from det_oprs.utils import get_padded_tensor


# Image, Annotation -> DataLoader -> ResNet -> FPN -> Balanced Feature Pyramid -> R_Anchor + R_Head -> R_Criteria
class Network(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 3, 7)
        self.BFP = BFP(in_channels=256, num_levels=5) if args.refine else None

        self.num_pred = args.num_predictions

        self.R_Head = RetinaNet_Head(args)
        self.R_Anchor = RetinaNet_Anchor(args)
        self.R_Criteria = RetinaNet_Criteria(args)

    def forward(self, image, im_info=None, gt_boxes=None):

        # pre-processing the data
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)

        # do inference
        # stride: 128,64,32,16,8, p7->p3
        fpn_fms = self.FPN(image)
        fpn_fms = self.BFP(fpn_fms)
        anchors_list = self.R_Anchor(fpn_fms)
        pred_cls_list, pred_reg_list = self.R_Head(fpn_fms)

        # release the useless data
        if self.training:
            loss_dict = self.R_Criteria(pred_cls_list, pred_reg_list, 
                                        anchors_list, gt_boxes, im_info)
            return loss_dict

        else:
            #pred_bbox = union_inference(
            #        anchors_list, pred_cls_list, pred_reg_list, im_info)
            pred_bbox = per_layer_inference(anchors_list, pred_cls_list, 
                                            pred_reg_list, im_info, self.num_predictions)
            return pred_bbox.cpu().detach()


class RetinaNet_Anchor():

    def __init__(self, args):
        self.anchors_generator = AnchorGenerator(
                    config.anchor_base_size,
                    config.anchor_aspect_ratios,
                    config.anchor_base_scale)

    def __call__(self, fpn_fms):
        # get anchors
        all_anchors_list = []
        base_stride = 8
        off_stride = 2**(len(fpn_fms)-1) # 8

        # fpn_fms : FPN 혹은 BFP에서 출력한, 서로 다른 크기의 Feature map이 저장되어 있는 list, List[Tensor]
        for fm in fpn_fms:
            layer_anchors = self.anchors_generator(fm, base_stride, off_stride)
            off_stride = off_stride // 2
            all_anchors_list.append(layer_anchors)
        
        # 각각의 feature map에 대한 anchor에 대한 정보(anchor의 x, y, w, h)가 저장된 리스트 
        return all_anchors_list


class RetinaNet_Criteria(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.loss_normalizer = 100 # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_momentum = 0.9

        self.num_pred = args.num_predictions
        self.return_anchors = args.return_anchors
        self.loss = args.loss

    def __call__(self, pred_cls_list, pred_reg_list, anchors_list, gt_boxes, im_info):
        
        # RetinaNet_Anchors, RetinaNet_Head를 통해 얻은 결과를 통합한 후
        # GT box와 비교하여 loss를 구함 

        all_anchors = torch.cat(anchors_list, axis=0)

        # all_pred_cls shape : (# of anchors, 2)
        all_pred_cls = torch.cat(pred_cls_list, axis=1).reshape(-1, (config.num_classes-1)*self.num_pred)
        all_pred_cls = torch.sigmoid(all_pred_cls)

        # all_pred_reg shape : (# of anchors, 4 x 2)
        all_pred_reg = torch.cat(pred_reg_list, axis=1).reshape(-1, 4*self.num_pred)

        # bounding box를 gt box에 할당
        # labels : positive/negative 여부
        # bbox_targets : gt box에 대한 정보
        # anchors : anchor에 대한 정보
        # labels shape : (-1, 2)
        # bbox targets shape : (-1, 8)
        # anchors shape : (-1, 8)
        labels, bbox_targets, anchors = concated_anchor_target(all_anchors, gt_boxes, im_info, 
                                                               top_k=self.num_pred, return_anchors=self.return_anchors)

        loss_dict = {}                                                      

        if self.num_pred > 1:
            # all_pred_cls shape : (-1, 2, 1)
            # all_pred_reg shape : (-1, 2, 4)
            all_pred_cls = all_pred_cls.reshape(-1, self.num_pred, config.num_classes-1)
            all_pred_reg = all_pred_reg.reshape(-1, self.num_pred, 4)

            if self.loss == "smooth_l1":

                loss0 = emd_loss_focal(
                        all_pred_reg[:, 0], all_pred_cls[:, 0],
                        all_pred_reg[:, 1], all_pred_cls[:, 1],
                        bbox_targets, labels, anchors)
                        
                loss1 = emd_loss_focal(
                        all_pred_reg[:, 1], all_pred_cls[:, 1],
                        all_pred_reg[:, 0], all_pred_cls[:, 0],
                        bbox_targets, labels, anchors)

            elif self.loss == "diou":
                loss0 = emd_loss_simple_diou(
                        all_pred_reg[:, 0], all_pred_cls[:, 0],
                        all_pred_reg[:, 1], all_pred_cls[:, 1],
                        bbox_targets, labels, anchors)
                        
                loss1 = emd_loss_simple_diou(
                        all_pred_reg[:, 1], all_pred_cls[:, 1],
                        all_pred_reg[:, 0], all_pred_cls[:, 0],
                        bbox_targets, labels, anchors)

            # 동일한 anchor에 대하여 서로 다른 delta 값을 적용한 후 gt box와 비교하기 위함 
            # all_pred_reg[:, 0], all_pred_cls[:, 0] : class score, regressor에 대한 0번째 값
            # all_pred_reg[:, 1], all_pred_cls[:, 1] : class score, regressor에 대한 1번째 값
            elif self.loss == "rdiou":
                loss0 = emd_loss_repulsion_diou(
                        all_pred_reg[:, 0], all_pred_cls[:, 0],
                        all_pred_reg[:, 1], all_pred_cls[:, 1],
                        bbox_targets, labels, anchors)
                        
                loss1 = emd_loss_repulsion_diou(
                        all_pred_reg[:, 1], all_pred_cls[:, 1],
                        all_pred_reg[:, 0], all_pred_cls[:, 0],
                        bbox_targets, labels, anchors)

            # anchor에 대해 다른 pred delta 값을 loss 값을 concat함 
            loss = torch.cat([loss0, loss1], axis=1)
            _, min_indices = loss.min(axis=1)

            # 서로 다른 pred delta를 적용한 경우 중에서 최솟값만 취함 
            loss_emd = loss[torch.arange(loss.shape[0]), min_indices]

            # normalization 
            num_pos = (labels[:, 0] > 0).sum().item()
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer \
                                    + (1 - self.loss_normalizer_momentum) * max(num_pos, 1)

            loss_emd = loss_emd.sum() / self.loss_normalizer
            loss_dict['retina_emd_repulsion_diou'] = loss_emd 

        else:
            fg_mask = (labels > 0).flatten()
            valid_mask = (labels >= 0).flatten()    
            
            loss_reg, loss_cls = multi_task_loss(all_pred_reg, all_pred_cls, labels, bbox_targets, fg_mask, valid_mask)
            num_pos_anchors = fg_mask.sum().item()
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer \
                                    + (1 - self.loss_normalizer_momentum) * max(num_pos_anchors, 1)
            loss_reg = loss_reg.sum() / self.loss_normalizer
            loss_cls = loss_cls.sum() / self.loss_normalizer
            
            loss_dict['retina_focal_loss'] = loss_cls
            loss_dict['retina_smooth_l1'] = loss_reg

        del all_anchors
        del all_pred_cls
        del all_pred_reg

        return loss_dict 


class RetinaNet_Head(nn.Module):

    def __init__(self, args):
        super().__init__()

        # num_convs : sub-network에서 적용할 conv layer의 수 
        num_convs = 4
        in_channels = 256

        # cls_subnet : classification network의 conv layer를 저장하는 리스트
        # bbox subnet : bbox regression network의 conv layer를 저장하는 리스트 
        cls_subnet = []
        bbox_subnet = []

        # proposal당 생성할 bounding box의 수 
        self.num_pred = args.num_predictions

        # 각 subnet 리스트에 conv layer 추가 
        for _ in range(num_convs):
            cls_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_subnet.append(nn.ReLU(inplace=True))
            bbox_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_subnet.append(nn.ReLU(inplace=True))

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)

        # predictor
        # class score channels : 9 x 2  = (anchor의 수) x (생성할 bounding box의 class score)
        # bbox score channels : 9 x 4 x 2  = (anchor의 수 ) x (delta 값(x, y, w, h)) x (생성할 bounding box의 class score)
        self.cls_score = nn.Conv2d(in_channels, 
                                    config.num_cell_anchors * (config.num_classes-1) * self.num_pred,
                                    kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, config.num_cell_anchors * 4 * self.num_pred,
                                    kernel_size=3, stride=1, padding=1)

        # Initialization
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        prior_prob = 0.01
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)


    def forward(self, features):

        pred_cls = []
        pred_reg = []

        # Balanced Feature Pyramid에서 출력한 List[Tensor]를 순회하며
        # feature map별로 predicted class score, predicted delta 값을 저장 
        for feature in features:
            pred_cls.append(self.cls_score(self.cls_subnet(feature)))
            pred_reg.append(self.bbox_pred(self.bbox_subnet(feature)))

        # cls shape : (batch size, # of anchors, 2)
        # reg shape : (batch size, # of anchors, 8)
        assert pred_cls[0].dim() == 4
        pred_cls_list = [_.permute(0, 2, 3, 1).reshape(pred_cls[0].shape[0], -1, (config.num_classes-1)*self.num_pred)
                        for _ in pred_cls]
        pred_reg_list = [_.permute(0, 2, 3, 1).reshape(pred_reg[0].shape[0], -1, 4*self.num_pred)
                        for _ in pred_reg]

        return pred_cls_list, pred_reg_list


def per_layer_inference(anchors_list, pred_cls_list, pred_reg_list, im_info, num_pred):

    keep_anchors = []
    keep_cls = []
    keep_reg = []

    class_num = pred_cls_list[0].shape[-1] // num_pred # class_num = 1

    # anchors shape : (# of anchors, 4)
    # pred_cls shape : (# of anchors, 2)
    # pred_reg shape : (# of anchors, 8)
    for l_id in range(len(anchors_list)):
        anchors = anchors_list[l_id].reshape(-1, 4)
        pred_cls = pred_cls_list[l_id][0].reshape(-1, class_num*num_pred)
        pred_reg = pred_reg_list[l_id][0].reshape(-1, 4*num_pred)


        # class score 상위 1000개만 남김 
        if len(anchors) > config.test_layer_topk: # 1000
            ruler = pred_cls.max(axis=1)[0]
            
            # inds : class score 상위 1000개의 index 
            _, inds = ruler.topk(config.test_layer_topk, dim=0)
            # print(inds.shape)
            inds = inds.flatten()

            # class score 상위 1000개의 anchor, pred_cls, pred_reg만 남김 
            keep_anchors.append(anchors[inds])
            keep_cls.append(torch.sigmoid(pred_cls[inds]))
            keep_reg.append(pred_reg[inds])
        else:
            keep_anchors.append(anchors)
            keep_cls.append(torch.sigmoid(pred_cls))
            keep_reg.append(pred_reg)

    keep_anchors = torch.cat(keep_anchors, axis = 0)
    keep_cls = torch.cat(keep_cls, axis = 0)
    keep_reg = torch.cat(keep_reg, axis = 0)

    # multiclass
    tag = torch.arange(class_num).type_as(keep_cls) + 1 # 1
    tag = tag.repeat(keep_cls.shape[0], 1).reshape(-1,1) # 0, 0, 1, 1, ..... 

    if num_pred > 1:

        # pred_score0, 1 shape : (# of anchors, 1)
        pred_scores_0 = keep_cls[:, :class_num].reshape(-1, 1) 
        pred_scores_1 = keep_cls[:, class_num:].reshape(-1, 1)

        # pred_delta0, 1 shape : (# of anchors, 4) 
        pred_delta_0 = keep_reg[:, :4] 
        pred_delta_1 = keep_reg[:, 4:] 

        # restore bbox 
        pred_bbox_0 = restore_bbox(keep_anchors, pred_delta_0, False)
        pred_bbox_1 = restore_bbox(keep_anchors, pred_delta_1, False)
        pred_bbox_0 = pred_bbox_0.repeat(1, class_num).reshape(-1, 4)
        pred_bbox_1 = pred_bbox_1.repeat(1, class_num).reshape(-1, 4)

        # [x, y, w, h, class score, tag, x, y, w, h, class score, tag]
        pred_bbox_0 = torch.cat([pred_bbox_0, pred_scores_0, tag], axis=1)
        pred_bbox_1 = torch.cat([pred_bbox_1, pred_scores_1, tag], axis=1)

        pred_bbox = torch.cat((pred_bbox_0, pred_bbox_1), axis=1)

    else:
        pred_scores = keep_cls.reshape(-1, 1)
        pred_bbox = restore_bbox(keep_anchors, keep_reg, False)
        pred_bbox = pred_bbox.repeat(1, class_num).reshape(-1, 4)
        pred_bbox = torch.cat([pred_bbox, pred_scores, tag], axis=1)

    # [x, y, w, h, class score, tag, x, y, w, h, class score, tag]
    return pred_bbox


def restore_bbox(rois, deltas, unnormalize=False):
    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)

    return pred_bbox