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

import numpy as np
from PIL import ImageDraw, Image
import utils_coreML as utils

import matplotlib
matplotlib.use('Agg')


## get the feature by model farword computation
def get_feature(model, image):
    start = time.time()
    processed_image = utils.process_image(image)
    resp, conf, x, y, w, h, e = utils.predict(model, np.expand_dims(processed_image, axis=0))
    resp = np.squeeze(resp, axis=0)
    conf = np.squeeze(conf, axis=0)
    x = np.squeeze(x, axis=0)
    y = np.squeeze(y, axis=0)
    w = np.squeeze(w, axis=0)
    h = np.squeeze(h, axis=0)
    e = np.squeeze(e, axis=0)
    print('inference time {:.5f}'.format(time.time() - start))
    return resp, conf, x, y, w, h, e

if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read('config.ini', 'UTF-8')

    model = utils.get_model()
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
                input_image = cv2.resize(input_image, (224, 224))
                #  humans = estimate(model, input_image.transpose(2, 0, 1).astype(np.float32))

                ## 获取keypoints and joints
                feature_map = get_feature(model, input_image.transpose(2, 0, 1).astype(np.float32))
                humans = utils.get_humans_by_feature(feature_map)

                ## 画出图像
                pilImg = Image.fromarray(input_image)
                pilImg = utils.draw_humans(
                    pil_image=pilImg,
                    humans=humans
                )
                img_with_humans = cv2.cvtColor(np.asarray(pilImg), cv2.COLOR_RGB2BGR)
                FPS = round(1.0/(time.time() - tic), 2)
                FPS_list.append(FPS)
                cv2.putText(img_with_humans, 'FPS: % f' % (1.0 / (time.time() - tic)),
                            (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                img_with_humans = cv2.resize(img_with_humans, (3 * 224, 3 * 224))

                print('Processing frame: {}/{}'.format(i, video_length/frame_rate_ratio))
                toc = time.time()
                print ('processing time is -----------> %.5f' % (toc - tic))

                out.write(img_with_humans)
            # 每次在这里变化的
            ret_val, input_image = cam.read()
            i += 1
        # compute average FPS
        average_fps = sum(FPS_list)/len(FPS_list)
        print('total {} frame, {} frame per second\n\n\n'.format(len(FPS_list), round(average_fps, 1)) )


    # Video input path
    video_path = './../videos/test'

    # Output location
    video = 'test'
    output_path = './../videos/outputs/'
    output_format = '.mp4'
    video_output = output_path + video + output_format

    FPS_list = []
    files = next(os.walk(video_path))[-1]
    for item in files:
        video_file = os.path.join(video_path, item)
        name = item.split('.')[0]
        video_output = output_path + name + output_format
        video_handle(video_file, video_output)
        break

