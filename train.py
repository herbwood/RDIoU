import os
import sys
import argparse
import torch

import time

sys.path.insert(0, './lib')
sys.path.insert(0, './model')

from data.CrowdHuman import CrowdHuman
from utils import misc_utils


class Train_config:
    # size
    world_size = 0
    mini_batch_size = 0
    iter_per_epoch = 0
    total_epoch = 0

    # learning
    warm_iter = 0
    learning_rate = 0
    momentum = 0
    weight_decay = 0
    lr_decay = 0

    # model
    log_dump_interval = 0
    resume_weights = None
    init_weights = None
    model_dir = ''
    log_path = ''


def do_train_epoch(net, data_iter, optimizer, rank, epoch, train_config):

    # rank : 
    if rank == 0:
        fid_log = open(train_config.log_path,'a')

    # learning rate decay 
    # train_config.lr_decay = [33, 43]
    if epoch >= train_config.lr_decay[0]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = train_config.learning_rate / 10

    if epoch >= train_config.lr_decay[1]:
        for param_group in optimizer.param_groups:
            param_group['lr'] = train_config.learning_rate / 100

    loss_all = 0.0

    # image
    # gtboxes : [x1, y1, x2, y2, tag], shape : (# of gt boxes per image, 5)
    # im_info : [0, 0, 1, height, width, # of gtboxes], shape : (# of gt boxes per image, 6)
    for (images, gt_boxes, im_info), step in zip(data_iter, range(0, train_config.iter_per_epoch)):

        optimizer.zero_grad()
        outputs = net(images.cuda(rank), im_info.cuda(rank), gt_boxes.cuda(rank))
        total_loss = sum([outputs[key].mean() for key in outputs.keys()])
        assert torch.isfinite(total_loss).all(), outputs 

        total_loss.backward()
        optimizer.step()

        # print, write results 
        if rank == 0 and step % train_config.log_dump_interval == 0:
            stastic_total_loss = total_loss.item()
            loss_all += stastic_total_loss * train_config.mini_batch_size

            line = 'Epoch:{}, iter:{}, lr:{:.5f}, loss is {:.4f}.'.format(
                    epoch, step, optimizer.param_groups[0]['lr'], loss_all / ((step+1) * train_config.mini_batch_size))
            print(line)
            fid_log.write(line+'\n')
            fid_log.write(str(outputs)+'\n')
            fid_log.flush()

    if rank == 0:
        fid_log.close()



def train_worker(rank, train_config, network, config, args):

    # set the parallel
    # 분산 환경 학습을 위한 초기화 
    # backend : gloo 방법
    # init_method : pytorch 환경변수를 통해 초기화
    # world_size : 사용하려는 gpu 수
    # rank : 현재 gpu의 index 
    torch.distributed.init_process_group(backend='gloo', 
                                        init_method='env://',
                                        world_size=train_config.world_size, 
                                        rank=rank)

    # initialize model
    net = network(args)

    # load pretrain model
    backbone_dict = torch.load(train_config.init_weights)
    net.resnet50.load_state_dict(backbone_dict['state_dict'])
    net.cuda(rank)
    begin_epoch = 1

    # build optimizer
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=train_config.learning_rate, 
                                momentum=train_config.momentum,
                                weight_decay=train_config.weight_decay)

    # resume weights
    if train_config.resume_weights:
        model_file = os.path.join(train_config.model_dir,
                                'dump-{}.pth'.format(train_config.resume_weights))
        check_point = torch.load(model_file, map_location=torch.device('cpu'))
        net.load_state_dict(check_point['state_dict'])
        begin_epoch = train_config.resume_weights + 1

    # using distributed data parallel
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[rank], broadcast_buffers=False)

    # build data loader
    crowdhuman = CrowdHuman(config, if_train=True)

    # collate_fn : 
    data_iter = torch.utils.data.DataLoader(dataset=crowdhuman,
                                            batch_size=train_config.mini_batch_size,
                                            num_workers=8,
                                            collate_fn=crowdhuman.merge_batch,
                                            shuffle=True,
                                            pin_memory=True)

    for epoch_id in range(begin_epoch, train_config.total_epoch+1):

        do_train_epoch(net, data_iter, optimizer, rank, epoch_id, train_config)

        if rank == 0:
            #save the model
            fpath = os.path.join(train_config.model_dir, 'dump-{}.pth'.format(epoch_id))
            model = dict(epoch = epoch_id,
                        state_dict = net.module.state_dict(),
                        optimizer = optimizer.state_dict())
            torch.save(model,fpath)


def multi_train(params, config, network):

    # check gpus
    if not torch.cuda.is_available():
        print('No GPU exists!')
        return

    else:
        num_gpus = torch.cuda.device_count()

    torch.set_default_tensor_type('torch.FloatTensor')

    # setting training config
    train_config = Train_config()
    train_config.world_size = num_gpus
    train_config.total_epoch = config.max_epoch
    train_config.iter_per_epoch = config.nr_images_epoch // (num_gpus * config.train_batch_per_gpu)
    train_config.mini_batch_size = config.train_batch_per_gpu
    train_config.warm_iter = config.warm_iter
    train_config.learning_rate = config.base_lr * config.train_batch_per_gpu * num_gpus
    train_config.momentum = config.momentum
    train_config.weight_decay = config.weight_decay
    train_config.lr_decay = config.lr_decay

    train_config.model_dir = os.path.join('./model/', params.model_dir, config.model_dir)
    line = 'network.lr.{}.train.{}'.format(train_config.learning_rate, train_config.total_epoch)
    train_config.log_path = os.path.join('./model/', params.model_dir, config.output_dir, line+'.log')
    train_config.resume_weights = params.resume_weights
    train_config.init_weights = config.init_weights
    train_config.log_dump_interval = config.log_dump_interval
    misc_utils.ensure_dir(train_config.model_dir)

    # print the training config
    """
    Num of GPUs:1, learning rate:0.00250, mini batch size:2,
    train_epoch:30, iter_per_epoch:7500, decay_epoch:[24, 27]
    """
    
    line = 'Num of GPUs:{}, learning rate:{:.5f}, mini batch size:{}, \
            \ntrain_epoch:{}, iter_per_epoch:{}, decay_epoch:{}'.format(
            num_gpus, train_config.learning_rate, train_config.mini_batch_size,
            train_config.total_epoch, train_config.iter_per_epoch, train_config.lr_decay)

    print(line)
    print("Init multi-processing training...")

    # generate processes
    torch.multiprocessing.spawn(train_worker, nprocs=num_gpus, args=(train_config, network, config, args))


def run_train(args):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '8888'

    # import libs
    model_root_dir = os.path.join('./model/', args.model_dir)
    sys.path.insert(0, model_root_dir)

    from config import config
    from network import Network

    multi_train(args, config, Network)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model_dir', '-md', default='retinanet', type=str,
                        help='Choose baseline model')
    parser.add_argument('--num_predictions', '-np', default=2, type=int,
                        help='Number of predictions per proposal')
    parser.add_argument('--return_anchors', '-ra', default=True, type=bool,
                        help='Whether to return anchors when matching anchor box with gt box')
    parser.add_argument('--loss', '-lf', default='rdiou', type=str,
                        help='Loss function to train on')
    parser.add_argument('--refine', '-bfp', default=True, type=bool,
                        help="Whether to apply Balanced Feature Pyramid")

    # Training
    parser.add_argument('--resume_weights', '-r', default=None,type=int,
                        help="Resume trained weights")    
    parser.add_argument('--gpu-id', '-g', default='0',type=str,
                        help="GPU numbers")

    args = parser.parse_args()

    run_train(args)