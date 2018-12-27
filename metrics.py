import ipdb;pdb=ipdb.set_trace
import argparse
import configparser
from collections import defaultdict
import itertools
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import copy
import os
import random
import time

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

import chainer
if chainer.backends.cuda.available:
    import cupy as xp
else:
    xp = np

import chainercv.transforms as transforms
from chainercv.utils import non_maximum_suppression
from chainercv.visualizations import vis_bbox
from PIL import ImageDraw, Image

from coco_dataset import get_coco_dataset
from mpii_dataset import get_mpii_dataset
from model import PoseProposalNet
from train import create_model
from network_resnet import ResNet50
from utils import parse_size
import ipdb;pdb=ipdb.set_trace

#  my wrapper
import log
mylog = log.Logger()
import cv2


COLOR_MAP = {}
DIRECTED_GRAPHS = [[]]
DATA_MODULE = None

def evaluation(config, list_):
    dataset_type = config.get('dataset', 'type')
    # gt_key points
    gt_kps_list = list_[0]
    # humans
    humans_list = list_[1]
    # gt bboxs
    gt_bboxs_list = list_[2]

    # is_visible
    is_visible_list = list_[3]



    # prediction bboxes list
    pred_bboxs_list = []

    kps_names = ['head_top', 'upper_neck', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']

    # 将humans转化成gt_kps的形式
    pred_kps_list = []
    for humans in humans_list:
        pred_bboxs = []
        pred_kps = []
        # humans maybe have several person
        for person in humans:
            item_pred = [] # 表示每一个人的14个关键点
            pred_bboxs.append(person[0]) # 预测的头部/身体边框
            for num in range(1,15):# 代表14个关键点
                if num in person:
                    y = (person[num][0] + person[num][2]) / 2
                    x = (person[num][1] + person[num][3]) / 2
                    item_pred.append([y, x])
                else:
                    item_pred.append([0,0])
            pred_kps.append(item_pred) # 表示一张图片上的所有人的关键点
        pred_kps_list.append(pred_kps)
        pred_bboxs_list.append(pred_bboxs)


    # 如果是mpii,乘以头部长度的0.5
    if dataset_type == 'mpii':
        factor = 0.5 * 0.6
    else:
        # 如果是coco，乘以全身长度的0.2
        factor = 0.2 * 0.6

    # pred_boxs_list and gt_boxs_list's center point
    pred_cp_list = []
    gt_cp_list = []
    for item in pred_bboxs_list:
        pred_cp = []
        for per_people in item:
            ymin, xmin, ymax, xmax = per_people
            cp_x = xmin + (xmax-xmin)/2
            cp_y = ymin + (ymax-ymin)/2
            pred_cp.append((cp_x,cp_y))
        pred_cp_list.append(pred_cp)
    for item in gt_bboxs_list:
        gt_cp = []
        for per_people in item:
            rx, ry, rw, rh = per_people
            cp_x = rx+rw/2
            cp_y = ry+rh/2
            gt_cp.append((cp_x,cp_y))
        gt_cp_list.append(gt_cp)

    # 用来判断结果
    results = {}

    for k in range(len(gt_cp_list)):
        pred_cp, gt_cp, pred_kps, gt_kps, gt_boxs,is_visible = pred_cp_list[k], gt_cp_list[k], pred_kps_list[k], gt_kps_list[k], gt_bboxs_list[k], is_visible_list[k]

        # 标记是否有被处理
        pred_index = np.zeros(len(pred_kps))
        gt_index = np.zeros(len(gt_kps))
        for p_index in range(len(pred_cp)):
            p_cp = pred_cp[p_index]
            dists = [np.linalg.norm(np.array(p_cp) - np.array(g_cp)) for g_cp in gt_cp]
            if dists == []:
                continue

            g_index = dists.index(min(dists))

            # 找到配对的人
            pred_kps_item = pred_kps[p_index]
            gt_kps_item = gt_kps[g_index]
            is_visible_item = is_visible[g_index]

            # 成功匹配的标记为1
            pred_index[p_index] = 1
            gt_index[g_index] = 1

            # 比较的标准
            gt_box = gt_boxs[g_index]
            h,w = gt_box[2:]
            length = np.sqrt((pow(h,2) + pow(w, 2))) * factor


            for i in range(len(kps_names)):
                name = kps_names[i]

                if not results.get(name):
                    results[name] = []

                is_v = is_visible_item[i]

                # 不检查不可见的（除了头部）
                if is_v == 0 and i!=0 :
                    continue

                pred_point = pred_kps_item[i]
                gt_point = gt_kps_item[i]
                # compute pred gt point distance
                distance_pt = np.linalg.norm(np.array(pred_point)-np.array(gt_point))


                if distance_pt < length:
                    results[name].append(1)
                else:
                    results[name].append(0)

            # 单人检测， 只检测一个人的
            break



    # accuracy
    total_results = []
    head = []
    shoulder = []
    ankle = []
    elbow = []
    wrist = []
    hip = []
    knee = []
    for item in kps_names:
        total_results += results[item]
        if 'knee' in item:
            knee += results[item]
        elif 'shoulder' in item:
            shoulder += results[name]
        elif 'ankle' in item:
            ankle += results[item]
        elif 'elbow' in item:
            elbow += results[item]
        elif 'wrist' in item:
            wrist += results[item]
        elif 'hip' in item:
            hip += results[item]
        else:
            head += results[item]


    print(len(head), len(hip))
    pck = np.sum(total_results)/len(total_results)
    pck_head = np.sum(head)/len(head)
    pck_shoulder = np.sum(shoulder)/len(shoulder)
    pck_ankle = np.sum(ankle)/len(ankle)
    pck_elbow = np.sum(elbow)/len(elbow)
    pck_wrist = np.sum(wrist)/len(wrist)
    pck_hip = np.sum(hip)/len(hip)
    pck_knee = np.sum(knee)/len(knee)

    pck, pck_head, pck_shoulder, pck_ankle, pck_elbow, pck_wrist, pck_hip, pck_knee = round(pck,2), round(pck_head,2), round(pck_shoulder, 2), round(pck_ankle, 2), round(pck_elbow, 2), round(pck_wrist, 2), round(pck_hip, 2), round(pck_knee, 2)

    mylog.info('the total pck is: {}'.format(pck))
    mylog.info('head: {}\tshoulder: {}\tankle: {}\telbow: {}\twrist: {}\thip: {}\n\n'.format(pck_head, pck_shoulder, pck_ankle, pck_elbow, pck_wrist, pck_hip))




def get_feature(model, image):
    start = time.time()
    image = xp.asarray(image)
    processed_image = model.feature_layer.prepare(image)
    resp, conf, x, y, w, h, e = model.predict(xp.expand_dims(processed_image, axis=0))
    resp = chainer.backends.cuda.to_cpu(resp.array)
    conf = chainer.backends.cuda.to_cpu(conf.array)
    w = chainer.backends.cuda.to_cpu(w.array)
    h = chainer.backends.cuda.to_cpu(h.array)
    x = chainer.backends.cuda.to_cpu(x.array)
    y = chainer.backends.cuda.to_cpu(y.array)
    e = chainer.backends.cuda.to_cpu(e.array)

    resp = np.squeeze(resp, axis=0)
    conf = np.squeeze(conf, axis=0)
    x = np.squeeze(x, axis=0)
    y = np.squeeze(y, axis=0)
    w = np.squeeze(w, axis=0)
    h = np.squeeze(h, axis=0)
    e = np.squeeze(e, axis=0)
    #  logger.info('inference time {:.5f}'.format(time.time() - start))
    return resp, conf, x, y, w, h, e


def estimate(model, image, thresh):
    feature_map = get_feature(model, image)
    detection_thresh = thresh
    return get_humans_by_feature(model, feature_map, detection_thresh)


def get_humans_by_feature(model, feature_map, detection_thresh):
    # resp, conf x, y, w, h's shape = (15, 7, 7) represent each grid have 15 proposals
    # e's shape = (14, 5, 5, 7, 7)
    resp, conf, x, y, w, h, e = feature_map
    start = time.time()
    delta = resp * conf
    K = len(model.keypoint_names)
    outW, outH = model.outsize # 7 7
    ROOT_NODE = 0  # instance
    start = time.time()
    # 还原
    rx, ry = model.restore_xy(x, y)
    rw, rh = model.restore_size(w, h)
    ymin, ymax = ry - rh / 2, ry + rh / 2
    xmin, xmax = rx - rw / 2, rx + rw / 2
    bbox = np.array([ymin, xmin, ymax, xmax]) # rectangle
    bbox = bbox.transpose(1, 2, 3, 0)
    root_bbox = bbox[ROOT_NODE]
    score = delta[ROOT_NODE] # score.shape=(7, 7)
    candidate = np.where(score > detection_thresh) # return two array, note these 满足条件的横坐标、纵坐标
    score = score[candidate]
    root_bbox = root_bbox[candidate]
    selected = non_maximum_suppression(
        bbox=root_bbox, thresh=0.3, score=score)
    root_bbox = root_bbox[selected]
    #  logger.info('detect instance {:.5f}'.format(time.time() - start))
    start = time.time()

    humans = []
    e = e.transpose(0, 3, 4, 1, 2)
    ei = 0  # index of edges which contains ROOT_NODE as begin
    # alchemy_on_humans
    for hxw in zip(candidate[0][selected], candidate[1][selected]):
        human = {ROOT_NODE: bbox[(ROOT_NODE, hxw[0], hxw[1])]}  # initial
        for graph in DIRECTED_GRAPHS:
            eis, ts = graph
            i_h, i_w = hxw
            for ei, t in zip(eis, ts):
                index = (ei, i_h, i_w)  # must be tuple
                u_ind = np.unravel_index(np.argmax(e[index]), e[index].shape)
                j_h = i_h + u_ind[0] - model.local_grid_size[1] // 2
                j_w = i_w + u_ind[1] - model.local_grid_size[0] // 2
                if j_h < 0 or j_w < 0 or j_h >= outH or j_w >= outW:
                    break
                if delta[t, j_h, j_w] < detection_thresh:
                    break
                human[t] = bbox[(t, j_h, j_w)]
                i_h, i_w = j_h, j_w

        humans.append(human)
    #  logger.info('alchemy time {:.5f}'.format(time.time() - start))
    #  logger.info('num humans = {}'.format(len(humans)))
    return humans


def draw_humans(keypoint_names, edges, pil_image, humans, mask=None):
    """
    this is what happens when you use alchemy on humans...
    note that image should be pil object
    """
    start = time.time()
    drawer = imagedraw.draw(pil_image)
    for human in humans:
        for k, b in human.items():
            if mask:
                fill = (255, 255, 255) if k == 0 else none
            else:
                fill = none
            ymin, xmin, ymax, xmax = b
            if k == 0:
                # adjust size
                t = 1
                xmin = int(xmin * t + xmax * (1 - t))
                xmax = int(xmin * (1 - t) + xmax * t)
                ymin = int(ymin * t + ymax * (1 - t))
                ymax = int(ymin * (1 - t) + ymax * t)
                if mask:
                    resized = mask.resize(((xmax - xmin), (ymax - ymin)))
                    pil_image.paste(resized, (xmin, ymin), mask=resized)
                else:
                    pass
                    ## 注释了下面两个，不画边框了
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=color_map[keypoint_names[k]])
            else:
                pass
                #  drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                 #  fill=fill,
                                 #  outline=color_map[keypoint_names[k]])
        for s, t in edges:
            if s in human and t in human:
                by = (human[s][0] + human[s][2]) / 2
                bx = (human[s][1] + human[s][3]) / 2
                ey = (human[t][0] + human[t][2]) / 2
                ex = (human[t][1] + human[t][3]) / 2

                drawer.line([bx, by, ex, ey],
                            fill=color_map[keypoint_names[s]], width=3)

    logger.info('draw humans {: .5f}'.format(time.time() - start))
    return pil_image


def create_model(config, model_name=''):
    global DIRECTED_GRAPHS, COLOR_MAP

    dataset_type = config.get('dataset', 'type')

    if dataset_type == 'mpii':
        import mpii_dataset as x_dataset
    elif dataset_type == 'coco':
        import coco_dataset as x_dataset
    else:
        raise Exception('Unknown dataset {}'.format(dataset_type))

    KEYPOINT_NAMES = x_dataset.KEYPOINT_NAMES
    EDGES = x_dataset.EDGES
    DIRECTED_GRAPHS = x_dataset.DIRECTED_GRAPHS
    COLOR_MAP = x_dataset.COLOR_MAP

    model = PoseProposalNet(
        model_name=config.get('model_param', 'model_name'),
        insize=parse_size(config.get('model_param', 'insize')),
        keypoint_names=KEYPOINT_NAMES,
        edges=np.array(EDGES),
        local_grid_size=parse_size(config.get('model_param', 'local_grid_size')),
        parts_scale=parse_size(config.get(dataset_type, 'parts_scale')),
        instance_scale=parse_size(config.get(dataset_type, 'instance_scale')),
        width_multiplier=config.getfloat('model_param', 'width_multiplier'),
    )

    result_dir = config.get('result', 'dir')

    if model_name:
        chainer.serializers.load_npz(
            os.path.join(result_dir, 'bestmodel.npz'),
            model
        )
    else:
        chainer.serializers.load_npz(
            os.path.join(result_dir, 'bestmodel.npz'),
            model
        )

    logger.info('cuda enable {}'.format(chainer.backends.cuda.available))
    logger.info('ideep enable {}'.format(chainer.backends.intel64.is_ideep_available()))
    if chainer.backends.cuda.available:
        logger.info('gpu mode')
        print('----------------------GPU------------')
        model.to_gpu(device=0)
    elif chainer.backends.intel64.is_ideep_available():
        print('----------------------Ideep------------')
        logger.info('USE cpu Indel64 mode')
        model.to_intel64()
    return model


def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--modelname", help="model full name", default='',dest='modelName')
    parser.add_argument("-n", "--testnum", help="the number of test image", type=int, default=1000, dest='test_num')
    args = parser.parse_args()
    modelName = args.modelName

    config = configparser.ConfigParser()
    config.read('config.ini', 'UTF-8')
    dataset_type = config.get('dataset', 'type')
    logger.info('loading {}'.format(dataset_type))
    if dataset_type == 'mpii':
        _, test_set = get_mpii_dataset(
            insize=parse_size(config.get('model_param', 'insize')),
            image_root=config.get(dataset_type, 'images'),
            annotations=config.get(dataset_type, 'annotations'),
            train_size=config.getfloat(dataset_type, 'train_size'),
            min_num_keypoints=config.getint(dataset_type, 'min_num_keypoints'),
            seed=config.getint('training_param', 'seed'),
        )
    elif dataset_type == 'coco':
        test_set = get_coco_dataset(
            insize=parse_size(config.get('model_param', 'insize')),
            image_root=config.get(dataset_type, 'val_images'),
            annotations=config.get(dataset_type, 'val_annotations'),
            min_num_keypoints=config.getint(dataset_type, 'min_num_keypoints'),
        )
    else:
        raise Exception('Unknown dataset {}'.format(dataset_type))

    model = create_model(config, modelName)

    ## 生成用于计算pck_object的gt_KPs、 gt_bboxs, human(pred_KPs, pred_bboxs) is _visible
    pck_object = [[], [], [], []]

    modelName = modelName if modelName else 'trained/bestmodel.npz'
    test_num = args.test_num
    print('model name: {}\t test image number: {}'.format(modelName, test_num))
    # 测试多张图片
    for i in tqdm(range(test_num)):
        idx = random.choice(range(len(test_set)))
        image = test_set.get_example(idx)['image']
        gt_kps = test_set.get_example(idx)['keypoints']
        gt_bboxs = test_set.get_example(idx)['bbox'] # (left down point, w, h)
        is_visible = test_set.get_example(idx)['is_visible'] #

        # include pred_KPs, pred_bbox
        humans = estimate(model,
                        image.astype(np.float32), 0.15)
        pck_object[0].append(gt_kps)
        pck_object[1].append(humans)
        pck_object[2].append(gt_bboxs)
        pck_object[3].append(is_visible)
    mylog.info('model name: {}\t test image number: {}'.format(modelName, test_num))
    evaluation(config, pck_object)


if __name__ == '__main__':
    main()
