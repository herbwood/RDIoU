import torch

from config import config

from det_oprs.bbox_opr import bbox_transform_inv_opr, box_overlap_opr 
from utils.misc_utils import xywh_to_xyxy


def softmax_loss(score, label, ignore_label=-1):

    with torch.no_grad():
        max_score = score.max(axis=1, keepdims=True)[0]

    score -= max_score
    log_prob = score - torch.log(torch.exp(score).sum(axis=1, keepdims=True))
    mask = label != ignore_label
    vlabel = label * mask

    onehot = torch.zeros(vlabel.shape[0], config.num_classes, device=score.device)
    onehot.scatter_(1, vlabel.reshape(-1, 1), 1)

    loss = -(log_prob * onehot).sum(axis=1)
    loss = loss * mask

    return loss


def smooth_l1_loss(pred, target, beta: float):

    if beta < 1e-5:
        loss = torch.abs(input - target)

    else:
        abs_x = torch.abs(pred- target)
        in_mask = abs_x < beta
        loss = torch.where(in_mask, 0.5 * abs_x ** 2 / beta, abs_x - 0.5 * beta)

    return loss.sum(axis=1)


def emd_loss_softmax(p_b0, p_s0, p_b1, p_s1, targets, labels):
    """
    Input example 

    p_b0 : Refined Box A loc feature map : torch.Size([512, 8])
    p_s0 : Refined Box A cls feature map : torch.Size([512, 2])
    p_b1 : Refined Box B loc feature map : torch.Size([512, 8])
    p_s1 : Refined Box B cls feature map : torch.Size([512, 2])
    
    targets : torch.Size([512, 8])
    labels : torch.Size([512, 2])
    """

    # reshape
    
    # pred_delta shape 
    # torch.Size([512, 8]), torch.Size([512, 8])
    # torch.Size([512, 16])
    # torch.Size([1024, 8])
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])

    # pred_score shape
    # torch.Size([512, 2]), torch.Size([512, 2])
    # torch.Size([512, 4])
    # torch.Size([1024, 2])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])

    # target shape : torch.Size([1024, 4])
    targets = targets.reshape(-1, 4)

    # labels shape : torch.Size([1024])
    labels = labels.long().flatten()

    # cons masks
    valid_masks = labels >= 0 # remove ignore labels
    fg_masks = labels > 0 # only positive labels 

    # multiple class
    pred_delta = pred_delta.reshape(-1, config.num_classes, 4) # [1024, 2, 4]
    fg_gt_classes = labels[fg_masks]
    pred_delta = pred_delta[fg_masks, fg_gt_classes, :]

    # loss for regression
    # only get loss for positive samples 
    localization_loss = smooth_l1_loss(
        pred_delta,
        targets[fg_masks],
        config.rcnn_smooth_l1_beta)

    # loss for classification
    objectness_loss = softmax_loss(pred_score, labels)
    loss = objectness_loss * valid_masks

    # total loss
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    
    return loss.reshape(-1, 1)

def focal_loss(inputs, targets, alpha=-1, gamma=2):

    class_range = torch.arange(1, inputs.shape[1] + 1, device=inputs.device)
    pos_pred = (1 - inputs) ** gamma * torch.log(inputs)
    neg_pred = inputs ** gamma * torch.log(1 - inputs)

    pos_loss = (targets == class_range) * pos_pred * alpha
    neg_loss = (targets != class_range) * neg_pred * (1 - alpha)
    loss = -(pos_loss + neg_loss)

    return loss.sum(axis=1)
    

def emd_loss_focal(p_b0, p_s0, p_b1, p_s1, targets, labels):

    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])
    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)


    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels, config.focal_loss_alpha, config.focal_loss_gamma)

    fg_masks = (labels > 0).flatten()
    localization_loss = smooth_l1_loss(
            pred_delta[fg_masks],
            targets[fg_masks],
            config.smooth_l1_beta)
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    loss = loss.reshape(-1, 1)

    return loss

#####################Repulsion DIoU loss####################################


def bbox_with_delta(rois, deltas, unnormalize=True):
    # if unnormalize:
    #     std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
    #     mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
    #     deltas = deltas * std_opr + mean_opr

    pred_bbox = bbox_transform_inv_opr(rois, deltas)

    return pred_bbox


def repulsion_diou_overlap(delta, anchors, bboxes2, get_iou=True, epsilon=5e-10, mask=None):
    
    # anchor boxes 
    bboxes1 = bbox_with_delta(anchors, delta)

    if isinstance(bboxes2, tuple):
        bbox2_anchor, bbox2_delta = bboxes2
        bboxes2 = bbox_with_delta(bbox2_anchor, bbox2_delta)

    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))

    if rows * cols == 0:
        return dious

    exchange = False

    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    # w1, h1 : bboxes1 width, height 
    # w1, h1 shape : (number of positive anchors, 1)
    # w2, h2 : bboxes2 width, height
    # w2, h2 shape : (number of target gt boxes, 1)
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    # pred bbox, gt box area 
    # shape : (numbef of boxes, 1)
    area1 = w1 * h1
    area2 = w2 * h2

    # pred bbox, gt box center point coord 
    # shape : (number of boxes, 1)
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    # inter area coord
    # shape : (number of boxes, 2)
    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])

    # outer area coord for C
    # shape : (number of boxes, 2)
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    # intersection area
    # inter shape : (number of boxes, 2)
    # inter_area shape : (number of boxes, 1)
    # diagonal distance between pred box and gt box
    # inter_diag shape : (number of boxes, 1)
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2

    # outer area 
    # outer : outer rectangle diagonal x, y coord
    # outer shape : (number of boxes, 2)
    # outer_diag : diagnoal distance of outer rectangle C
    # outer_diag shape : (number of boxes, 1)
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)

    # Union area of pred bbox and gt box
    # union shape : (number of boxes, 1)
    union = area1 + area2 - inter_area

    # IoU and center point distance 
    # dious : IoU - (center point distance) 
    # dious shape : (number of boxes, 1)
    iou = inter_area / (union + epsilon)
    center_point_distance = (inter_diag) / (outer_diag + epsilon)
    dious = torch.clamp(dious,min=-1.0,max = 1.0)

    if get_iou:
        dious = iou - center_point_distance
        dious = dious * mask 
        
        return dious

    return center_point_distance * mask


def repulsion_diou_loss(pred_delta, anchors, targets, alpha=0.5, beta=0.5):

    # Repulsion term 1 
    # between positive bbox and positive non target gt boxes 
    # Returns :
    # second_gt_indices : second gt indices
    # second_gt_mask : whether iou > 0
    second_gt_overlaps = box_overlap_opr(anchors, targets)
    max_gt_overlaps, gt_assignment = second_gt_overlaps.topk(2, dim=1, sorted=True)
    second_matched_gt_iou, second_gt_indices = max_gt_overlaps[:, 1], gt_assignment[:, 1]
    # second_gt_mask = torch.where(second_matched_gt_iou > 0, 1, 0).cuda()
    second_gt_mask = (second_matched_gt_iou > 0).flatten().cuda()

    # Repulsion term 2
    # between bbox and bboxes with different target
    # Returns :
    # second_bbox_indices : second bbox indices
    # second_bbox_mask : whether iou > 0, whether assigned to same gt box or not 
    second_bbox_overlaps = box_overlap_opr(anchors, anchors)
    max_bbox_overlaps, bbox_assignment = second_bbox_overlaps.topk(2, dim=1, sorted=True)
    second_matched_bbox_iou, second_bbox_indices = max_bbox_overlaps[:, 1], bbox_assignment[:, 1]
    # second_bbox_iou_mask = torch.where(second_matched_bbox_iou > 0, 1, 0)
    second_bbox_iou_mask = (second_matched_bbox_iou > 0).flatten()
    second_bbox_gt_mask = torch.all(targets != targets[second_bbox_indices], dim=1).int()
    second_bbox_mask = second_bbox_iou_mask * second_bbox_gt_mask
    second_bbox_mask = second_bbox_mask.cuda()

    first_gt_mask = torch.ones(anchors.shape[0]).cuda()

    del second_gt_overlaps 
    del second_bbox_overlaps 

    # bbox_gt1_diou : bbox and target gt box iou and center distance, shape : (number of boxes, 1)
    # bbox_gt2_diou : bbox and non target gt box center distance, shape : (number of boxes, 1)
    # bbox_bbox2_diou : bbox and 2nd bbox center distance , shape : (number of boxes, 1)
    bbox_gt1_diou = repulsion_diou_overlap(pred_delta, anchors, targets, get_iou=True, mask=first_gt_mask)
    bbox_gt2_diou = repulsion_diou_overlap(pred_delta, anchors, targets[second_gt_indices], get_iou=False, mask=second_gt_mask)
    bbox_bbox2_diou = repulsion_diou_overlap(pred_delta, anchors, (anchors[second_bbox_indices], pred_delta[second_bbox_indices]), 
                                    get_iou=False, mask=second_bbox_mask)   

    dious = bbox_gt1_diou + alpha * bbox_gt2_diou + beta * bbox_bbox2_diou
    # dious = bbox_gt1_diou + alpha * bbox_gt2_diou
    dious = torch.clamp(dious, min=-3.0, max=3.0) 

    dious = dious.reshape(-1, 1)
    loss = 1.0 - dious
    loss = loss.sum(axis=1)

    return loss 


def emd_loss_repulsion_diou(p_b0, p_s0, p_b1, p_s1, targets, labels, anchors):

    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1]) 

    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    anchors = anchors.reshape(-1, 4)

    # valid mask : positive/negative label mask(True or False)
    # valid mask shape : (-1, 1)
    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels, config.focal_loss_alpha, config.focal_loss_gamma)

    # fg_masks : positive label mask(True or False)
    fg_masks = (labels > 0).flatten()

    localization_loss = repulsion_diou_loss(pred_delta[fg_masks], anchors[fg_masks], targets[fg_masks], alpha=0.3, beta=0.3)

    # objectness_loss = objectness_loss.cuda()
    # localization_loss = localization_loss.cuda()

    # print(valid_mask.device, objectness_loss.device, fg_masks.device, localization_loss.device)

    # final loss : anchor top1+top2 loss, anchor2 top1+top2 loss, ... 
    # loss shape : (anchors x 2, 1) => (anchors, 1) 
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    
    return loss.reshape(-1, 1)



    ####################Simple DIOU###########################

def bbox_overlaps_diou(bboxes1, bboxes2):
    
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    dious = torch.zeros((rows, cols))

    if rows * cols == 0:
        return dious

    exchange = False

    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True

    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2
    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:],bboxes2[:, 2:])
    inter_min_xy = torch.max(bboxes1[:, :2],bboxes2[:, :2])
    out_max_xy = torch.max(bboxes1[:, 2:],bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2],bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2

    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1+area2-inter_area

    iou = inter_area / union
    center_distance = inter_diag / outer_diag
    # print("IoU average :", torch.mean(iou))
    # print("Center Distance :", torch.mean(center_distance))

    dious = (inter_area / union) - ((inter_diag) / outer_diag)
    dious = torch.clamp(dious,min=-1.0,max = 1.0)
    if exchange:
        dious = dious.T

    return dious

def simple_diou_loss(pred_delta, targets, anchors):

    bboxes = bbox_with_delta(anchors, pred_delta)
    dious = bbox_overlaps_diou(bboxes, targets)

    dious = dious.reshape(-1, 1)
    loss = 1.0 - dious
    loss = loss.sum(axis=1)

    return loss 

def emd_loss_simple_diou(p_b0, p_s0, p_b1, p_s1, targets, labels, anchors):
    
    pred_delta = torch.cat([p_b0, p_b1], axis=1).reshape(-1, p_b0.shape[-1])
    pred_score = torch.cat([p_s0, p_s1], axis=1).reshape(-1, p_s0.shape[-1])

    targets = targets.reshape(-1, 4)
    labels = labels.long().reshape(-1, 1)
    anchors = anchors.reshape(-1, 4)

    valid_mask = (labels >= 0).flatten()
    objectness_loss = focal_loss(pred_score, labels, config.focal_loss_alpha, config.focal_loss_gamma)

    fg_masks = (labels > 0).flatten()
    # fg_masks = (labels > 0) * (anchors > 0)
    # fg_masks = fg_masks.flatten()
    # overlaps = box_overlap_opr(anchors[fg_masks], targets[fg_masks])
    # max_overlaps, gt_assignment = overlaps.topk(2, dim=1, sorted=True)
    # # print(max_overlaps)
    # print(pred_delta)
    localization_loss = simple_diou_loss(pred_delta[fg_masks], targets[fg_masks], anchors[fg_masks]) 
            
    loss = objectness_loss * valid_mask
    loss[fg_masks] = loss[fg_masks] + localization_loss
    loss = loss.reshape(-1, 2).sum(axis=1)
    loss = loss.reshape(-1, 1)

    return loss