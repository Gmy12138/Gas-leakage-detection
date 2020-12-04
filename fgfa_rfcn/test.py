# coding=utf-8
# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Yuwen Xiong
# --------------------------------------------------------
# Based on:
# MX-RCNN
# Copyright (c) 2016 by Contributors
# Licence under The Apache 2.0 License
# https://github.com/ijkguo/mx-rcnn/
# --------------------------------------------------------

import _init_paths

import cv2
import argparse
import os
import sys
import time
import logging
from config.config import config, update_config

def parse_args():
    # print('test----parse_args----in----')
    parser = argparse.ArgumentParser(description='Test a R-FCN network')
    # general
    # 通常测试一个网络都是使用cfg文件yaml
    parser.add_argument('--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    # rcnn
    # rcnn相应参数，包括vis可视化，ignore_cache忽视缓存，thresh有效检测的阈值，shuffle可视化中随机数据
    parser.add_argument('--vis', help='turn on visualization', action='store_true')
    parser.add_argument('--ignore_cache', help='ignore cached results boxes', action='store_true')
    parser.add_argument('--thresh', help='valid detection threshold', default=1e-3, type=float)
    parser.add_argument('--shuffle', help='shuffle data on visualization', action='store_true')
    args = parser.parse_args()
    # print('test----parse_args----out----')
    return args

args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(curr_path, '../external/mxnet', config.MXNET_VERSION))

import mxnet as mx
from function.test_rcnn import test_rcnn
from utils.create_logger import create_logger


def main():
    # ctx为gpu(...)，其中配置项在yaml配置文件中
    ctx = [mx.gpu(int(i)) for i in config.gpus.split(',')]
    # print('ctx:', ctx)
    print args

    # config.output_path在yaml文件中定义，cfg为对应的yaml文件路径
    logger, final_output_path = create_logger(config.output_path, args.cfg, config.dataset.test_image_set)
    # print('1')
    # print final_output_path
    # print('1')
    # config.dataset.dataset=ImageNetVID
    test_rcnn(config, config.dataset.dataset, config.dataset.test_image_set, config.dataset.root_path, config.dataset.dataset_path, config.dataset.motion_iou_path,
              ctx, os.path.join(final_output_path, '..', '_'.join([iset for iset in config.dataset.image_set.split('+')]), config.TRAIN.model_prefix), config.TEST.test_epoch,
              args.vis, args.ignore_cache, args.shuffle, config.TEST.HAS_RPN, config.dataset.proposal, args.thresh, logger=logger, output_path=final_output_path,
              enable_detailed_eval=config.dataset.enable_detailed_eval)

if __name__ == '__main__':
    main()