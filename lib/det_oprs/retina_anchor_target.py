import torch

import numpy as np
from config import config
from det_oprs.bbox_opr import box_overlap_opr, bbox_transform_opr


@torch.no_grad()
def retina_anchor_target(anchors, gt_boxes, im_info, top_k=1):
    # anchors.shape, gt_boxes.shape, im_info.shape
    # torch.Size([219483, 4]) torch.Size([1, 500, 5]) torch.Size([1, 6])

    total_anchor = anchors.shape[0]
    return_labels = []
    return_bbox_targets = []

    # get per image proposals and gt_boxes
    for bid in range(config.train_batch_per_gpu):

        gt_boxes_perimg = gt_boxes[bid, :int(im_info[bid, 5]), :]
        anchors = anchors.type_as(gt_boxes_perimg)

        # IoU between bbox and gt box
        # number of anchors x number of gt boxes per image
        overlaps = box_overlap_opr(anchors, gt_boxes_perimg[:, :-1]) 
        

        # gt max and indices
        # shape : number of anchors x (top_k)
        # anchor box와 IoU 값이 높은 상위 2개의 ground truth box와의 IoU 값 
        # max overlaps.flatten
        # bbox1 top1 iou, bbox1 top2 iou, bbox2 top1 iou, bbox2 top2 iou, ... 
        # gt_assignment.flatten
        # bbox1 top1 index, bbox1 top2 index, bbox2 top1 index, bbox top2 index, ... 
        max_overlaps, gt_assignment = overlaps.topk(top_k, dim=1, sorted=True)
        max_overlaps= max_overlaps.flatten()
        gt_assignment= gt_assignment.flatten()

        # gt_assignment_for_gt : bbox에 할당할 gt box의 index 
        # shape : number of gtboxes indicies
        _, gt_assignment_for_gt = torch.max(overlaps, axis=0)
        del overlaps

        # positive, negative, ignore labels 
        labels = gt_boxes_perimg[gt_assignment, 4]
        labels = labels * (max_overlaps >= config.negative_thresh)
        ignore_mask = (max_overlaps < config.positive_thresh) * (max_overlaps >= config.negative_thresh)
        labels[ignore_mask] = -1
        

        # cons bbox targets
        target_boxes = gt_boxes_perimg[gt_assignment, :4]
        # anchors.repeat(1, top_k).shape : number of anchor x 8(=2 x 4)
        # target_anchors : (number of anchors x 2) x 4
        # top 2 anchors 
        target_anchors = anchors.repeat(1, top_k).reshape(-1, anchors.shape[-1])
        bbox_targets = bbox_transform_opr(target_anchors, target_boxes)

        if config.allow_low_quality:
            labels[gt_assignment_for_gt] = gt_boxes_perimg[:, 4]
            low_quality_bbox_targets = bbox_transform_opr(
                anchors[gt_assignment_for_gt], gt_boxes_perimg[:, :4])
            bbox_targets[gt_assignment_for_gt] = low_quality_bbox_targets

        labels = labels.reshape(-1, 1 * top_k)
        bbox_targets = bbox_targets.reshape(-1, 4 * top_k)
        return_labels.append(labels)
        return_bbox_targets.append(bbox_targets)

    if config.train_batch_per_gpu == 1:
        # labels shape : number of anchors x 2
        # bbox_targets shape : number of anchors x 8

        return labels, bbox_targets
    else:
        return_labels = torch.cat(return_labels, axis=0)
        return_bbox_targets = torch.cat(return_bbox_targets, axis=0)

        return return_labels, return_bbox_targets