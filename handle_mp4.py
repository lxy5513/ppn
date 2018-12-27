import ipdb;  pdb = ipdb.set_trace
import sys
import argparse
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

import configparser
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import os
import time

import chainer
from chainercv.utils import non_maximum_suppression
import cv2
import numpy as np
from PIL import ImageDraw, Image

from predict import COLOR_MAP
from predict import estimate, draw_humans, create_model
from utils import parse_size

import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.ini', 'UTF-8')

    model = create_model(config)


    def video_handle(video_file, video_output):
        global FPS_list
        # Video reader
        cam = cv2.VideoCapture(video_file)
        input_fps = cam.get(cv2.CAP_PROP_FPS)
        ret_val, input_image = cam.read()
        video_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))

        ending_frame = video_length

        # Video writer
        frame_rate_ratio = 1
        output_fps = input_fps / frame_rate_ratio
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # 后面两者是写入的图片大小
        out = cv2.VideoWriter(video_output,fourcc, output_fps, (672, 672))

        i = 0 # default is 0
        while(cam.isOpened()) and ret_val == True and i < ending_frame:
            if i%frame_rate_ratio == 0:

                tic = time.time()
                mask = None

                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = cv2.resize(input_image, model.insize)
                humans = estimate(model, input_image.transpose(2, 0, 1).astype(np.float32))
                pilImg = Image.fromarray(input_image)
                pilImg = draw_humans(
                    model.keypoint_names,
                    model.edges,
                    pilImg,
                    humans,
                    mask=mask.rotate(degree) if mask else None
                )
                img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
                msg = 'GPU ON' if chainer.backends.cuda.available else 'GPU OFF'
                msg += ' ' + config.get('model_param', 'model_name')
                FPS = round(1.0/(time.time() - tic), 2)
                FPS_list.append(FPS)
                cv2.putText(img_with_humans, 'FPS: % f' % (1.0 / (time.time() - tic)),
                            (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                img_with_humans = cv2.resize(img_with_humans, (3 * model.insize[0], 3 * model.insize[1]))
                #  cv2.imshow('Pose Proposal Network' + msg, img_with_humans)

                print('Processing frame: {}/{}'.format(i, video_length/frame_rate_ratio))
                toc = time.time()
                print ('processing time is %.5f' % (toc - tic))

                out.write(img_with_humans)
            # 每次在这里变化的
            ret_val, input_image = cam.read()
            i += 1
        # compute average FPS
        average_fps = sum(FPS_list)/len(FPS_list)
        print('total {} frame, {} frame per second\n\n\n'.format(len(FPS_list), round(average_fps, 1)) )


    # Video input path
    video_path = 'videos/test'

    # Output location
    video = 'test'
    output_path = 'videos/outputs/'
    output_format = '.mp4'
    video_output = output_path + video + output_format

    FPS_list = []
    files = next(os.walk(video_path))[-1]
    for item in files:
        video_file = os.path.join(video_path, item)
        name = item.split('.')[0]
        video_output = output_path + name + output_format
        video_handle(video_file, video_output)

