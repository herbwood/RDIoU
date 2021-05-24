import os
import sys
import math
import argparse

sys.path.insert(0, '../lib')
sys.path.insert(0, '../model')

import numpy as np
from tqdm import tqdm
import torch
from torch.multiprocessing import Queue, Process

from data.CrowdHuman import CrowdHuman
from utils import misc_utils, nms_utils
from evaluate import compute_JI, compute_APMR
from evaluate.compute_APMR import evaluate_APMR


def eval_all(args, config, network):

    # model_path
    saveDir = os.path.join('../model', args.model_dir, config.model_dir)
    evalDir = os.path.join('../model', args.model_dir, config.eval_dir)
    misc_utils.ensure_dir(evalDir)
    model_file = os.path.join(saveDir, 'dump-{}.pth'.format(args.resume_weights))
    assert os.path.exists(model_file)

    # get devices
    str_devices = args.devices
    devices = misc_utils.device_parser(str_devices)

    # load data
    crowdhuman = CrowdHuman(config, if_train=False)

    #crowdhuman.records = crowdhuman.records[:10]
    # multiprocessing
    num_devs = len(devices)
    len_dataset = len(crowdhuman)

    # gpu당 처리할 이미지 
    num_image = math.ceil(len_dataset / num_devs)

    # 단방향으로만 정보를 전송
    result_queue = Queue()

    procs = []
    all_results = []

    for i in range(num_devs):
        start = i * num_image
        end = min(start + num_image, len_dataset)

        # Process : 단일 프로세스를 생성하는 경우 사용 
        # target 인자에 해당하는 함수에 args 전달 
        proc = Process(target=inference, args=(config, network, 
                                                model_file, devices[i], 
                                                crowdhuman, start, end, result_queue))
        proc.start()
        procs.append(proc)

    pbar = tqdm(total=len_dataset, ncols=50)

    for i in range(len_dataset):
        t = result_queue.get()
        all_results.append(t)
        pbar.update(1)

    pbar.close()

    for p in procs:
        p.join()

    fpath = os.path.join(evalDir, 'dump-{}.json'.format(args.resume_weights))
    misc_utils.save_json_lines(all_results, fpath)

    # AP, MR, JI 값 evaluate 
    eval_path = os.path.join(evalDir, 'eval-{}.json'.format(args.resume_weights))
    eval_fid = open(eval_path,'w')
    res_line, JI = compute_JI.evaluation_all(fpath, 'box')

    for line in res_line:
        eval_fid.write(line+'\n')

    AP, MR = evaluate_APMR(fpath, config.eval_source, 'box')
    line = 'AP:{:.4f}, MR:{:.4f}, JI:{:.4f}.'.format(AP, MR, JI)
    print(line)

    eval_fid.write(line+'\n')
    eval_fid.close()


def inference(config, network, model_file, device, dataset, start, end, result_queue):

    torch.set_default_tensor_type('torch.FloatTensor')
    torch.multiprocessing.set_sharing_strategy('file_system')

    # init model
    net = network()
    net.cuda(device)
    net = net.eval()
    check_point = torch.load(model_file)
    net.load_state_dict(check_point['state_dict'])

    # init data
    dataset.records = dataset.records[start:end];
    data_iter = torch.utils.data.DataLoader(dataset=dataset, shuffle=False)

    # inference
    for (image, gt_boxes, im_info, ID) in data_iter:
        pred_boxes = net(image.cuda(device), im_info.cuda(device))
        scale = im_info[0, 2] # scale 

        if config.test_nms_method == 'set_nms':

            assert pred_boxes.shape[-1] > 6, "Not EMD Network! Using normal_nms instead."
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"

            top_k = pred_boxes.shape[-1] // 6 # 2 
            n = pred_boxes.shape[0] # # of predicted boxes 
            pred_boxes = pred_boxes.reshape(-1, 6)

            # 0, 0, 1, 1, 2, 2 ..... idents를 붙임 
            idents = np.tile(np.arange(n)[:,None], (1, top_k)).reshape(-1, 1)
            pred_boxes = np.hstack((pred_boxes, idents))

            # class score가 threshold보다 높은 경우만 keep 
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]

            # set nms 수행 
            keep = nms_utils.set_cpu_nms(pred_boxes, 0.5)
            pred_boxes = pred_boxes[keep]

        elif config.test_nms_method == 'normal_nms':
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            pred_boxes = pred_boxes.reshape(-1, 6)
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]
            keep = nms_utils.cpu_nms(pred_boxes, config.test_nms)
            pred_boxes = pred_boxes[keep]

        elif config.test_nms_method == 'none':
            assert pred_boxes.shape[-1] % 6 == 0, "Prediction dim Error!"
            pred_boxes = pred_boxes.reshape(-1, 6)
            keep = pred_boxes[:, 4] > config.pred_cls_threshold
            pred_boxes = pred_boxes[keep]

        else:
            raise ValueError('Unknown NMS method.')

        pred_boxes[:, :4] /= scale
        pred_boxes[:, 2:4] -= pred_boxes[:, :2]
        gt_boxes = gt_boxes[0].numpy()
        gt_boxes[:, 2:4] -= gt_boxes[:, :2]

        result_dict = dict(ID=ID[0], height=int(im_info[0, -3]), 
                            width=int(im_info[0, -2]),
                            dtboxes=boxes_dump(pred_boxes), gtboxes=boxes_dump(gt_boxes))

        result_queue.put_nowait(result_dict)


def boxes_dump(boxes):

    if boxes.shape[-1] == 7:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5]),
                   'proposal_num':int(box[6])} for box in boxes]

    elif boxes.shape[-1] == 6:
        result = [{'box':[round(i, 1) for i in box[:4].tolist()],
                   'score':round(float(box[4]), 5),
                   'tag':int(box[5])} for box in boxes]

    elif boxes.shape[-1] == 5:
        result = [{'box':[round(i, 1) for i in box[:4]],
                   'tag':int(box[4])} for box in boxes]

    else:
        raise ValueError('Unknown box dim.')

    return result


def run_test():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '-md', default=None, required=True, type=str)
    parser.add_argument('--resume_weights', '-r', default=None, required=True, type=str)
    parser.add_argument('--devices', '-d', default='0', type=str)
    os.environ['NCCL_IB_DISABLE'] = '1'
    args = parser.parse_args()

    # import libs
    model_root_dir = os.path.join('../model/', args.model_dir)
    sys.path.insert(0, model_root_dir)

    from config import config
    from network import Network

    eval_all(args, config, Network) 


if __name__ == '__main__':
    """
    cd tools
    python test.py -md rcnn_emd_refine -r 1
    """
    run_test()