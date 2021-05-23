import torch

import numpy as np
from config import config
from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr


@torch.no_grad()
def retina_anchor_target_diou(anchors, gt_boxes, im_info, top_k=1):

    total_anchor = anchors.shape[0]
    return_labels = []
    return_bbox_targets = []
    return_anchors = []

    for bid in range(config.train_batch_per_gpu):
        
        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        anchors = anchors.type_as(gt_boxes_perimg)
        

        # IoU between bbox and gt box
        # number of anchors x number of gt boxes
        # anchors = xywh_to_xyxy(anchors)
        # gt_boxes_perimg[:, :-1] = xywh_to_xyxy(gt_boxes_perimg[:, :-1])
        overlaps = box_overlap_opr(anchors, gt_boxes_perimg[:, :-1])

        # max_overlaps : 각 anchor와 gt box의 상위 2개의 IoU값
        # max_overlaps shape : number of anchors x 2
        # gt_assignment : 상위 2개의 IoU값의 index
        # gt_assignment shape : number of anchors x 2
        max_overlaps, gt_assignment = overlaps.topk(top_k, dim=1, sorted=True)

        # max_overlaps : bbox1-gt1 iou, bbox1-gt2 iou, bbox2-gt1 iou, bbox2-gt2 iou, .... 
        # gt_assignment : bbox1-gt1 index, bbox1-gt2 index, .... 
        # max_overlaps shape : (number of anchorsx2) x 1
        # gt_assignment shape : (number of anchorsx2) x 1
        max_overlaps = max_overlaps.flatten()
        gt_assignment = gt_assignment.flatten()

        # gt_assignment : bbox에 할당할 gt box의 index 
        _, gt_assignment_for_gt = torch.max(overlaps, axis=0)

        del overlaps 

        # positive/negative label
        labels = gt_boxes_perimg[gt_assignment, 4]
        labels = labels * (max_overlaps >= config.negative_thresh)

        # ignore label
        ignore_mask = (max_overlaps < config.positive_thresh) * (max_overlaps >= config.negative_thresh)
        labels[ignore_mask] = -1

        # target_boxes : box1 target1, box1 target2, box2 target1, ....
        # target_anchors : box1, box, box2, box2, box3, .... 
        # anchors.repeat(1, top_k).shape : number of anchor x 8(=2 x 4) => 행으로 복사
        # target_anchors shape: (number of anchors x 2) x 4 
        target_boxes = gt_boxes_perimg[gt_assignment, :4]
        target_anchors = anchors.repeat(1, top_k).reshape(-1, anchors.shape[-1])
        # print(target_anchors[:5, :])
        # print(target_boxes[:5, :])
        if config.allow_low_quality:
            labels[gt_assignment_for_gt] = gt_boxes_perimg[:, 4]
            # low_quality_bbox_targets = bbox_transform_opr(anchors[gt_assignment_for_gt], gt_boxes_perimg[:, :4])
            # bbox_targets[gt_assignment_for_gt] = low_quality_bbox_targets
            target_boxes[gt_assignment_for_gt] = gt_boxes_perimg[:, :4]


        # labels shape : (-1, 2)
        # target boxes shape : (-1, 8)
        # target anchors : (-1, 8)
        labels = labels.reshape(-1, 1 * top_k)
        target_boxes = target_boxes.reshape(-1, 4 * top_k)
        target_anchors = target_anchors.reshape(-1, 4 * top_k)

        return_labels.append(labels)
        return_bbox_targets.append(target_boxes)
        return_anchors.append(target_anchors)

    if config.train_batch_per_gpu == 1:
        return labels, target_boxes, target_anchors

    else:
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)
        return_anchors = torch.cat(return_anchors, axis=0)

        return return_labels, return_bbox_targets, return_anchors 


