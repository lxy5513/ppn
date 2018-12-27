import numpy as np
import mxnet as mx
import mxnet.contrib.onnx as onnx_mxnet
import time
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image
import PIL
import cv2
import os
import ipdb;pdb = ipdb.set_trace
from collections import namedtuple


keypoint_names = ['instance', 'head_top', 'upper_neck', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle']
edges = np.array([[ 0,  2],
       [ 2,  1],
       [ 2,  3],
       [ 2,  4],
       [ 2,  9],
       [ 2, 10],
       [ 3,  5],
       [ 5,  7],
       [ 4,  6],
       [ 6,  8],
       [ 9, 11],
       [11, 13],
       [10, 12],
       [12, 14]])


def get_model(ctx=-1, model_path='./bestmodel.onnx'):
    onnx_file = model_path
    sym, arg, aux = onnx_mxnet.import_model(onnx_file)

    # ### Get input data names
    data_names = [graph_input for graph_input in sym.list_inputs() if graph_input not in arg and graph_input not in aux]
    Batch = namedtuple('Batch', ['data'])

    test_image = np.random.rand(1,3, 224, 224).astype('float32')
    if ctx == -1:
        context = mx.cpu()
    else:
        context = mx.gpu(ctx)
    mod = mx.mod.Module(symbol=sym, data_names=data_names, context=context, label_names=None)
    mod.bind(for_training=False, data_shapes=[(data_names[0], test_image.shape)], label_shapes=None)
    mod.set_params(arg_params=arg, aux_params=aux, allow_missing=True, allow_extra=True)
    return mod


def non_maximum_suppression(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        order = score.argsort()[::-1]
        bbox = bbox[order]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        tl = np.maximum(b[:2], bbox[selec, :2])
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


def restore_xy(x, y):
    gridW, gridH = 32, 32
    outW, outH = 7, 7
    X, Y = np.meshgrid(np.arange(outW, dtype=np.float32), np.arange(outH, dtype=np.float32))
    return (x + X) * gridW, (y + Y) * gridH

def restore_size(w, h):
    inW, inH = 224, 224
    return inW * w, inH * h

def get_humans_by_feature(feature_map, detection_thresh=0.15):
    DIRECTED_GRAPHS = [[[0, 1], [2, 1]], [[0, 2, 6, 7], [2, 3, 5, 7]], [[0, 3, 8, 9], [2, 4, 6, 8]], [[0, 4, 10, 11], [2, 9, 11, 13]], [[0, 5, 12, 13], [2, 10, 12, 14]]]

    resp, conf, x, y, w, h, e = feature_map
    resp, conf, x, y, w, h, e = resp.asnumpy(), conf.asnumpy(), x.asnumpy(),y.asnumpy() ,w.asnumpy(), h.asnumpy(), e.asnumpy()
    start = time.time()
    delta = resp * conf
    K = 15
    outW, outH = 7, 7
    ROOT_NODE = 0  # instance
    start = time.time()
    rx, ry = restore_xy(x, y)
    rw, rh = restore_size(w, h)
    ymin, ymax = ry - rh / 2, ry + rh / 2
    xmin, xmax = rx - rw / 2, rx + rw / 2
    bbox = np.array([ymin, xmin, ymax, xmax])
    bbox = bbox.transpose(1, 2, 3, 0)
    root_bbox = bbox[ROOT_NODE]
    score = delta[ROOT_NODE]
    candidate = np.where(score > detection_thresh)
    score = score[candidate]
    root_bbox = root_bbox[candidate]
    selected = non_maximum_suppression(
        bbox=root_bbox, thresh=0.3, score=score)
    root_bbox = root_bbox[selected]
    print('detect instance {:.5f}'.format(time.time() - start))
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
                j_h = i_h + u_ind[0] - 5 // 2
                j_w = i_w + u_ind[1] - 5 // 2
                if j_h < 0 or j_w < 0 or j_h >= outH or j_w >= outW:
                    break
                if delta[t, j_h, j_w] < detection_thresh:
                    break
                human[t] = bbox[(t, j_h, j_w)]
                i_h, i_w = j_h, j_w

        humans.append(human)
    print('alchemy time {:.5f}'.format(time.time() - start))
    print('num humans = {}'.format(len(humans)))
    return humans


def process_image(image):
    image = image.copy()
    # CHW -> HWC
    image = image.transpose((1, 2, 0))
    # RGB -> BGR
    image = image[:, :, ::-1]
    # NOTE: in the original paper they subtract a fixed mean image,
    #       however, in order to support arbitrary size we instead use the
    #       mean pixel (rather than mean image) as with VGG team. The mean
    #       value used in ResNet is slightly different from that of VGG16.
    image -= np.array([103.063, 115.903, 123.152])
    # HWC -> CHW
    image = image.transpose((2, 0, 1))
    return image



def resize(img, size, interpolation=PIL.Image.BILINEAR):
    img = img.transpose((1, 2, 0))
    if interpolation == PIL.Image.NEAREST:
        cv_interpolation = cv2.INTER_NEAREST
    elif interpolation == PIL.Image.BILINEAR:
        cv_interpolation = cv2.INTER_LINEAR
    elif interpolation == PIL.Image.BICUBIC:
        cv_interpolation = cv2.INTER_CUBIC
    elif interpolation == PIL.Image.LANCZOS:
        cv_interpolation = cv2.INTER_LANCZOS4
    H, W = size
    img = cv2.resize(img, dsize=(W, H), interpolation=cv_interpolation)

    # If input is a grayscale image, cv2 returns a two-dimentional array.
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
    return img.transpose((2, 0, 1))



def predict(model, image):
    K = 15
    B, _, _, _ = image.shape
    Batch = namedtuple('Batch', ['data'])
    #  image = mx.nd.zeros(image, ctx=mx.gpu())
    model.forward(Batch([mx.nd.array(image)]))
    feature_map = model.get_outputs()[0]
    resp = feature_map[:, 0 * K:1 * K, :, :]
    conf = feature_map[:, 1 * K:2 * K, :, :]
    x = feature_map[:, 2 * K:3 * K, :, :]
    y = feature_map[:, 3 * K:4 * K, :, :]
    w = feature_map[:, 4 * K:5 * K, :, :]
    h = feature_map[:, 5 * K:6 * K, :, :]
    e = feature_map[:, 6 * K:, :, :].reshape((
        B,
        14,
        5, 5,
        7, 7
    ))
    return resp, conf, x, y, w, h, e


def draw_humans(pil_image, humans, mask=None):
    """
    This is what happens when you use alchemy on humans...
    note that image should be PIL object
    """
    COLOR_MAP = {'instance': (225, 225, 225), 'head_top': (255, 0, 0), 'upper_neck': (255, 85, 0), 'r_shoulder': (255, 170, 0), 'r_elbow': (255, 255, 0), 'r_wrist': (170, 255, 0), 'l_shoulder': (85, 255, 0), 'l_elbow': (0, 127, 0), 'l_wrist': (0, 255, 85), 'r_hip': (0, 170, 170), 'r_knee': (0, 255, 255), 'r_ankle': (0, 170, 255), 'l_hip': (0, 85, 255), 'l_knee': (0, 0, 255), 'l_ankle': (85, 0, 255), 'r_eye': (170, 0, 255), 'l_eye': (255, 0, 255), 'r_ear': (255, 0, 170), 'l_ear': (255, 0, 85)}
    start = time.time()
    drawer = ImageDraw.Draw(pil_image)
    for human in humans:
        for k, b in human.items():
            if mask:
                fill = (255, 255, 255) if k == 0 else None
            else:
                fill = None
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
                    ## 注释了下面两个，不画边框了
                    drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                     fill=fill,
                                     outline=COLOR_MAP[keypoint_names[k]])
            else:
                pass
                #  drawer.rectangle(xy=[xmin, ymin, xmax, ymax],
                                 #  fill=fill,
                                 #  outline=COLOR_MAP[keypoint_names[k]])
        for s, t in edges:
            if s in human and t in human:
                by = (human[s][0] + human[s][2]) / 2
                bx = (human[s][1] + human[s][3]) / 2
                ey = (human[t][0] + human[t][2]) / 2
                ex = (human[t][1] + human[t][3]) / 2

                drawer.line([bx, by, ex, ey],
                            fill=COLOR_MAP[keypoint_names[s]], width=3)

    print('draw humans {: .5f}'.format(time.time() - start))
    return pil_image

