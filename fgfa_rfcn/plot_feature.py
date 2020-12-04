# coding=utf-8
# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuqing Zhu, Shuhao Fu, Xizhou Zhu, Yi Li, Haochen Zhang
# --------------------------------------------------------

import _init_paths

import argparse
import os
import glob
import sys
import time

import logging
import pprint
import cv2
from config.config import config as cfg
from config.config import update_config
from utils.image import resize, transform
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from flow_vis import flow_to_color, flow_uv_to_colors, make_colorwheel

# get config
# os.environ['PYTHONUNBUFFERED'] = '1'
# os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
# os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
cur_path = os.path.abspath(os.path.dirname(__file__))
update_config(cur_path + '/../experiments/fgfa_rfcn/cfgs/fgfa_rfcn_vid_demo.yaml')

sys.path.insert(0, os.path.join(cur_path, '../external/mxnet/', cfg.MXNET_VERSION))
import mxnet as mx
import time
# from core.tester import im_detect, Predictor, get_resnet_output, prepare_data, draw_all_detection
from core.tester_front import im_detect, Predictor, get_resnet_output, prepare_data, draw_all_detection
from symbols import *
from nms.seq_nms import seq_nms
from utils.load_model import load_param
from utils.tictoc import tic, toc
from nms.nms import py_nms_wrapper, cpu_nms_wrapper, gpu_nms_wrapper
from imageio import imread, imwrite

def parse_args():
    parser = argparse.ArgumentParser(description='Show Flow-Guided Feature Aggregation demo')
    args = parser.parse_args()
    return args

args = parse_args()

def flow2rgb(flow_map, max_value):
    flow_map_np = flow_map
    _, h, w = flow_map_np.shape
    flow_map_np[:,(flow_map_np[0] == 0) & (flow_map_np[1] == 0)] = float('nan')
    rgb_map = np.ones((3,h,w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5*(normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    return rgb_map.clip(0,1)


def plot_feature(predictor, data_batch):
    output_all = predictor.predict(data_batch)

    flow0 = output_all[0]['flow_output0'].asnumpy()
    flow1 = output_all[0]['flow_output1'].asnumpy()
    flow2 = output_all[0]['flow_output2'].asnumpy()
    flow3 = output_all[0]['flow_output3'].asnumpy()
    flow4 = output_all[0]['flow_output4'].asnumpy()
    flow5 = output_all[0]['flow_output5'].asnumpy()
    # flow0 = output_all[0]['flow1_output0'].asnumpy()
    # flow1 = output_all[0]['flow1_output1'].asnumpy()
    # flow2 = output_all[0]['flow1_output2'].asnumpy()
    # flow3 = output_all[0]['flow1_output3'].asnumpy()
    # flow4 = output_all[0]['flow1_output4'].asnumpy()
    # flow5 = output_all[0]['flow1_output5'].asnumpy()

    return [flow0,flow1,flow2,flow3,flow4,flow5]
    # out=output_all[0]['_plus9_output'].asnumpy()
    # return out



def save_image(output_dir, count, out_im):
    filename = str(count) + '.png'
    cv2.imwrite(output_dir + filename, out_im)

def main():
    # get symbol
    pprint.pprint(cfg)
    cfg.symbol = 'resnet_v1_101_flownet_rfcn'
    model = '/../model/fgfa_rfcn_vid_s'
    # 关键帧间隔*2+1为所有帧的间隔，论文中设置的KEY_FRAME_INTERVAL为10
    all_frame_interval = cfg.TEST.KEY_FRAME_INTERVAL +1
    # all_frame_interval = 7
    max_per_image = cfg.TEST.max_per_image
    feat_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()
    aggr_sym_instance = eval(cfg.symbol + '.' + cfg.symbol)()

    feat_sym = feat_sym_instance.get_feat_symbol(cfg)
    aggr_sym = aggr_sym_instance.get_plot_symbol(cfg)

    # set up class names
    num_classes = 2
    classes = ['__background__','smoke']

    # load demo data

    image_names = sorted(glob.glob(cur_path + '/../data/IR_smoke/Data/VID/val/8/*.png'))
    output_dir = cur_path + '/../demo/rfcn_fgfa_8_agg_1/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = []
    for im_name in image_names:
        assert os.path.exists(im_name), ('%s does not exist'.format(im_name))
        im = cv2.imread(im_name, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        target_size = cfg.SCALES[0][0]
        max_size = cfg.SCALES[0][1]
        im, im_scale = resize(im, target_size, max_size, stride=cfg.network.IMAGE_STRIDE)
        im_tensor = transform(im, cfg.network.PIXEL_MEANS)
        im_info = np.array([[im_tensor.shape[2], im_tensor.shape[3], im_scale]], dtype=np.float32)

        feat_stride = float(cfg.network.RCNN_FEAT_STRIDE)
        data.append({'data': im_tensor, 'im_info': im_info,  'data_cache': im_tensor,    'feat_cache': im_tensor})



    # get predictor

    print 'get-predictor'
    data_names = ['data', 'im_info', 'data_cache', 'feat_cache']
    label_names = []

    t1 = time.time()
    data = [[mx.nd.array(data[i][name]) for name in data_names] for i in xrange(len(data))]
    max_data_shape = [[('data', (1, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('data_cache', (6, 3, max([v[0] for v in cfg.SCALES]), max([v[1] for v in cfg.SCALES]))),
                       ('feat_cache', ((6, cfg.network.FGFA_FEAT_DIM,
                                                np.ceil(max([v[0] for v in cfg.SCALES]) / feat_stride).astype(np.int),
                                                np.ceil(max([v[1] for v in cfg.SCALES]) / feat_stride).astype(np.int))))]]
    provide_data = [[(k, v.shape) for k, v in zip(data_names, data[i])] for i in xrange(len(data))]
    provide_label = [None for _ in xrange(len(data))]

    arg_params, aux_params = load_param(cur_path + model, 1, process=True)

    feat_predictors = Predictor(feat_sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    aggr_predictors = Predictor(aggr_sym, data_names, label_names,
                          context=[mx.gpu(0)], max_data_shapes=max_data_shape,
                          provide_data=provide_data, provide_label=provide_label,
                          arg_params=arg_params, aux_params=aux_params)
    nms = py_nms_wrapper(cfg.TEST.NMS)


    # First frame of the video
    idx = 0
    data_batch = mx.io.DataBatch(data=[data[idx]], label=[], pad=0, index=idx,
                                 provide_data=[[(k, v.shape) for k, v in zip(data_names, data[idx])]],
                                 provide_label=[None])
    scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]
    all_boxes = [[[] for _ in range(len(data))]
                 for _ in range(num_classes)]
    data_list = deque(maxlen=all_frame_interval)
    feat_list = deque(maxlen=all_frame_interval)
    image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
    # append cfg.TEST.KEY_FRAME_INTERVAL padding images in the front (first frame)
    while len(data_list) < cfg.TEST.KEY_FRAME_INTERVAL:
        data_list.append(image)
        feat_list.append(feat)

    vis = False
    file_idx = 0
    thresh = 1e-3
    for idx, element in enumerate(data):

        data_batch = mx.io.DataBatch(data=[element], label=[], pad=0, index=idx,
                                     provide_data=[[(k, v.shape) for k, v in zip(data_names, element)]],
                                     provide_label=[None])
        scales = [data_batch.data[i][1].asnumpy()[0, 2] for i in xrange(len(data_batch.data))]

        if(idx != len(data)-1):

            if len(data_list) < all_frame_interval - 1:
                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                data_list.append(image)
                feat_list.append(feat)

            else:
                #################################################
                # main part of the loop
                #################################################
                image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
                data_list.append(image)
                feat_list.append(feat)

                prepare_data(data_list, feat_list, data_batch)
                flow = plot_feature(aggr_predictors, data_batch)

                # print flow.shape

                # if (cfg.TEST.SEQ_NMS==False):
                if file_idx==20:
                    # print flow.shape
                    # flow = flow.reshape(19, 24, -1)
                    # print flow.shape
                    # step = 3
                    # plt.quiver(np.arange(0, flow.shape[1], step), np.arange(flow.shape[0], -1, -step),
                    #            flow[::step, ::step, 0], flow[::step, ::step, 1])
                    #
                    # plt.savefig(output_dir + '/' + str(i) + '.png')
                    # plt.cla()
                    for i in range(len(flow)):
                        print flow[i].shape
                        flow[i]=flow[i].reshape(19, 24, -1)
                        print flow[i].shape
                        step=2
                        plt.quiver(np.arange(0,flow[i].shape[1],step),np.arange(flow[i].shape[0],-1,-step),
                                   flow[i][::step,::step,0],flow[i][::step,::step,1])

                        plt.savefig(output_dir+'/'+str(i)+'.png')
                        plt.cla()
                        # plt.show()
                        # flow[i] = flow[i].reshape(-1, 19, 24)
                        # print flow[i].shape
                        # rgb_flow = flow2rgb(20 * flow[i], max_value=None)
                        # to_save = (rgb_flow * 255).astype(np.uint8).transpose(1, 2, 0)
                        # imwrite(output_dir+'/'+str(i)+'.png', to_save)


                    break

                print 'testing {} '.format(str(file_idx)+'.png')
                file_idx += 1
        else:
            #################################################
            # end part of a video                           #
            #################################################

            end_counter = 0
            image, feat = get_resnet_output(feat_predictors, data_batch, data_names)
            while end_counter < cfg.TEST.KEY_FRAME_INTERVAL + 1:
                data_list.append(image)
                feat_list.append(feat)
                prepare_data(data_list, feat_list, data_batch)
                flow = plot_feature(aggr_predictors, data_batch)

                # print flow
                # if (cfg.TEST.SEQ_NMS == False):
                #     save_image(output_dir, file_idx, out_im)
                # print 'testing {} {:.4f}s'.format(str(file_idx) + '.png', total_time / (file_idx + 1))
                file_idx += 1
                end_counter+=1
        # break



    print 'done'

if __name__ == '__main__':
    main()
