import numpy as np
import pdb


def set_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""

    # dets shape : [number of predictions, 6]
    # [number of predictions (x1, x2, x2, y2, confidence score, tag)]

    # get IoU between superior box and all the other inferior boxes
    def _overlap(det_boxes, basement, others):
        eps = 1e-8

        # coordinate of superior box
        x1_basement, y1_basement, x2_basement, y2_basement \
                = det_boxes[basement, 0], det_boxes[basement, 1], \
                  det_boxes[basement, 2], det_boxes[basement, 3]
        # coordinate of inferior boxes
        x1_others, y1_others, x2_others, y2_others \
                = det_boxes[others, 0], det_boxes[others, 1], \
                  det_boxes[others, 2], det_boxes[others, 3]

        areas_basement = (x2_basement - x1_basement) * (y2_basement - y1_basement)
        areas_others = (x2_others - x1_others) * (y2_others - y1_others)

        xx1 = np.maximum(x1_basement, x1_others)
        yy1 = np.maximum(y1_basement, y1_others)
        xx2 = np.minimum(x2_basement, x2_others)
        yy2 = np.minimum(y2_basement, y2_others)

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas_basement + areas_others - inter + eps)

        return ovr

    # sort detection in descending order by confidence score
    scores = dets[:, 4]
    order = np.argsort(-scores)
    dets = dets[order]

    numbers = dets[:, -1] # tag : set number 
    keep = np.ones(len(dets)) == 1 # ex) [True, Fals, True, ...]
    ruler = np.arange(len(dets)) # [0, 1, 2, ..., number of boxes]

    while ruler.size > 0:
        basement = ruler[0] # superior box index
        ruler=ruler[1:] # indexes of inferior boxes
        num = numbers[basement] # set number of superior box

        # calculate the body overlap
        overlap = _overlap(dets[:, :4], basement, ruler)
        indices = np.where(overlap > thresh)[0] # thresh = 0.5, indexes of IoU over threshold

        # index of inferior box with IoU over threshold and same set number as superior box
        loc = np.where(numbers[ruler][indices] == num)[0]

        # the mask won't change in the step
        mask = keep[ruler[indices][loc]]#.copy() # bool value of loc equal to 1 
        keep[ruler[indices]] = False # remove inferior boxes over IoU threshold and not same set number as superior box
        keep[ruler[indices][loc][mask]] = True # keep the same set number boxe
        ruler[~keep[ruler]] = -1 # 
        ruler = ruler[ruler > 0] # remove all erased boxes

    
    keep = keep[np.argsort(order)]

    return keep
    


def cpu_nms(dets, base_thr):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1) * (y2 - y1)
    order = np.argsort(-scores)

    keep = []
    eps = 1e-8

    while len(order) > 0:

        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + eps)

        inds = np.where(ovr <= base_thr)[0]
        indices = np.where(ovr > base_thr)[0]
        order = order[inds + 1]

    return np.array(keep)


def _test():
    box1 = np.array([33,45,145,230,0.7])[None,:]
    box2 = np.array([44,54,123,348,0.8])[None,:]
    box3 = np.array([88,12,340,342,0.65])[None,:]
    boxes = np.concatenate([box1,box2,box3],axis = 0)
    
    nms_thresh = 0.1
    keep = set_cpu_nms(boxes,nms_thresh)
    alive_boxes = boxes[keep]

if __name__=='__main__':
    _test()