import math
import numpy as np
import chainer
from chainer.backends.cuda import get_array_module
from chainer import reporter
import chainer.functions as F
import chainer.links as L
from chainer import initializers
import ipdb;pdb=ipdb.set_trace

major, _, _ = chainer.__version__.split(".")
MAJOR = int(major)
if MAJOR >= 5:
    from chainer import static_graph
else:
    def static_graph(func):
        """
        dummy decorator to keep compatibility between Chainer v5 and v4
        """

        def wrap(self, *args, **kwargs):
            return func(self, *args, **kwargs)
        return wrap

EPSILON = 1e-6


def area(bbox):
    _, _, w, h = bbox
    return w * h


def intersection(bbox0, bbox1):
    x0, y0, w0, h0 = bbox0
    x1, y1, w1, h1 = bbox1

    w = F.relu(F.minimum(x0 + w0 / 2, x1 + w1 / 2) - F.maximum(x0 - w0 / 2, x1 - w1 / 2))
    h = F.relu(F.minimum(y0 + h0 / 2, y1 + h1 / 2) - F.maximum(y0 - h0 / 2, y1 - h1 / 2))

    return w * h


# bbox -> [center_x, center_y, w, h]
def iou(bbox0, bbox1):
    area0 = area(bbox0)
    area1 = area(bbox1)
    intersect = intersection(bbox0, bbox1)

    return intersect / (area0 + area1 - intersect + EPSILON)


def get_network(model, **kwargs):
    if model == 'mv2':
        from network_mobilenetv2 import MobilenetV2
        return MobilenetV2(**kwargs)
    elif model == 'resnet50':
        from network_resnet import ResNet50
        return ResNet50(**kwargs)
    elif model == 'resnet18':
        from network_resnet import ResNet
        return ResNet(n_layers=18)
    elif model == 'resnet34':
        from network_resnet import ResNet
        return ResNet(n_layers=34)
    else:
        raise Exception('Invalid model name')


class PoseProposalNet(chainer.link.Chain):

    def __init__(self,
                 model_name,
                 insize,
                 keypoint_names,
                 edges,# limbs, keypoints connection
                 local_grid_size,
                 parts_scale,
                 instance_scale,
                 width_multiplier=1.0,
                 lambda_resp=0.25,
                 lambda_iou=1.0,
                 lambda_coor=5.0,
                 lambda_size=5.0,
                 lambda_limb=0.5,
                 dtype=np.float32):
        super(PoseProposalNet, self).__init__()
        self.model_name = model_name
        self.insize = insize
        self.keypoint_names = keypoint_names
        self.edges = edges
        self.local_grid_size = local_grid_size
        self.dtype = dtype
        self.lambda_resp = lambda_resp
        self.lambda_iou = lambda_iou
        self.lambda_coor = lambda_coor
        self.lambda_size = lambda_size
        self.lambda_limb = lambda_limb
        self.parts_scale = np.array(parts_scale) # side length of square for kps
        self.instance_scale = np.array(instance_scale)
        with self.init_scope():
            # 基础的网络 feature size narrow 32 multiple channels=512
            self.feature_layer = get_network(model_name, dtype=dtype, width_multiplier=width_multiplier)
            ksize = self.feature_layer.last_ksize # 1
            self.lastconv = L.Convolution2D(None,
                                            6 * len(self.keypoint_names) +
                                            self.local_grid_size[0] * self.local_grid_size[1] * len(self.edges),
                                            ksize=ksize, stride=1, pad=ksize // 2,
                                            initialW=initializers.HeNormal(1 / np.sqrt(2), dtype))

        self.outsize = self.get_outsize() # the number of gird  fixed or not?  --- (7, 7) #p1 No fixed, changed by insize and local_grid_size
        inW, inH = self.insize # image input shape[1:]
        outW, outH = self.outsize
        self.gridsize = (int(inW / outW), int(inH / outH)) #32 32

    def get_outsize(self):
        inp = np.zeros((2, 3, self.insize[1], self.insize[0]), dtype=np.float32)
        out = self.forward(inp) # Reduce the length and width of the image by 32 times
        _, _, h, w = out.shape
        return w, h

    def restore_xy(self, x, y):
        xp = get_array_module(x) #numpy or cupy
        gridW, gridH = self.gridsize
        outW, outH = self.outsize #num of grid (W, H)
        X, Y = xp.meshgrid(xp.arange(outW, dtype=xp.float32), xp.arange(outH, dtype=xp.float32)) #生成网格
        return (x + X) * gridW, (y + Y) * gridH

    def restore_size(self, w, h):
        inW, inH = self.insize
        return inW * w, inH * h

    def encode(self, in_data):
        image = in_data['image']
        keypoints = in_data['keypoints']
        bbox = in_data['bbox']
        is_labeled = in_data['is_labeled']
        dataset_type = in_data['dataset_type']
        inW, inH = self.insize
        outW, outH = self.outsize
        gridW, gridH = self.gridsize
        K = len(self.keypoint_names)
        # 实际的点位置上有没有kp
        delta = np.zeros((K, outH, outW), dtype=np.float32) #ex: (15, 7, 7)
        tx = np.zeros((K, outH, outW), dtype=np.float32)
        ty = np.zeros((K, outH, outW), dtype=np.float32)
        tw = np.zeros((K, outH, outW), dtype=np.float32)
        th = np.zeros((K, outH, outW), dtype=np.float32)
        # H x W x H' x W' x |L|
        te = np.zeros((
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW), dtype=np.float32)

        # Set delta^i_k
        for (x, y, w, h), points, labeled in zip(bbox, keypoints, is_labeled):
            if dataset_type == 'mpii': # bbox = head rectangle
                partsW, partsH = self.parts_scale * math.sqrt(w * w + h * h)
                instanceW, instanceH = self.instance_scale * math.sqrt(w * w + h * h)
            elif dataset_type == 'coco':
                partsW, partsH = self.parts_scale * math.sqrt(w * w + h * h)
                instanceW, instanceH = w, h
            else:
                raise ValueError("must be 'mpii' or 'coco' but actual {}".format(dataset_type))
            cy = y + h / 2
            cx = x + w / 2
            points = [[cy, cx]] + list(points)
            labeled = [True] + list(labeled)
            for k, (yx, l) in enumerate(zip(points, labeled)):
                if not l:
                    continue
                cy = yx[0] / gridH # why? guarantee cy < outH
                cx = yx[1] / gridW
                ix, iy = int(cx), int(cy)
                sizeW = instanceW if k == 0 else partsW
                sizeH = instanceH if k == 0 else partsH
                if 0 <= iy < outH and 0 <= ix < outW:
                    delta[k, iy, ix] = 1
                    tx[k, iy, ix] = cx - ix
                    ty[k, iy, ix] = cy - iy
                    tw[k, iy, ix] = sizeW / inW # 归一化
                    th[k, iy, ix] = sizeH / inH

            for ei, (s, t) in enumerate(self.edges):
                if not labeled[s]:
                    continue
                if not labeled[t]:
                    continue
                src_yx = points[s]
                tar_yx = points[t]
                iyx = (int(src_yx[0] / gridH), int(src_yx[1] / gridW))
                jyx = (int(tar_yx[0] / gridH) - iyx[0] + self.local_grid_size[1] // 2,
                       int(tar_yx[1] / gridW) - iyx[1] + self.local_grid_size[0] // 2)

                if iyx[0] < 0 or iyx[1] < 0 or iyx[0] >= outH or iyx[1] >= outW:
                    continue
                if jyx[0] < 0 or jyx[1] < 0 or jyx[0] >= self.local_grid_size[1] or jyx[1] >= self.local_grid_size[0]:
                    continue

                te[ei, jyx[0], jyx[1], iyx[0], iyx[1]] = 1

        # define max(delta^i_k1, delta^j_k2) which is used for loss_limb
        max_delta_ij = np.ones((len(self.edges),
                                outH, outW,
                                self.local_grid_size[1], self.local_grid_size[0]), dtype=np.float32)
        or_delta = np.zeros((len(self.edges), outH, outW), dtype=np.float32)
        for ei, (s, t) in enumerate(self.edges):
            or_delta[ei] = np.minimum(delta[s] + delta[t], 1)
        mask = F.max_pooling_2d(np.expand_dims(or_delta, axis=0),
                                ksize=(self.local_grid_size[1], self.local_grid_size[0]),
                                stride=1,
                                pad=(self.local_grid_size[1] // 2, self.local_grid_size[0] // 2))
        mask = np.squeeze(mask.array, axis=0)
        for index, _ in np.ndenumerate(mask):
            max_delta_ij[index] *= mask[index]
        max_delta_ij = max_delta_ij.transpose(0, 3, 4, 1, 2)

        # preprocess image
        image = self.feature_layer.prepare(image)

        return image, delta, max_delta_ij, tx, ty, tw, th, te

    def _forward(self, x):
        h = F.cast(x, self.dtype)
        h = self.feature_layer(h)
        # self.last_activation = F.sigmoid
        h = self.feature_layer.last_activation(self.lastconv(h))
        #  x.shape, h.shape = (None, 3, 224, 224), (None, 440, 7, 7)
        return h

    @static_graph
    def static_forward(self, x):
        return self._forward(x)

    def forward(self, x):
        """
        this provide interface of forwarding
        chainer 5.0.0 gives us static_graph to increase of speed of training
        but for some reason this does train i.e. loss does not decrease at all.
        We do not trust it for now on training. On the other hand. the speed of
        inference increases very well.Also note that if we use ideep option,
        the output result between `static_forward` and `_forward` will be different.
        """
        if chainer.config.train:
            return self._forward(x)
        else:
            # represent chainer version
            if MAJOR >= 5 and chainer.backends.cuda.available:
                return self.static_forward(x)
            else:
                return self._forward(x)


    def __call__(self, image, delta, max_delta_ij, tx, ty, tw, th, te): # receive gt value
    # delta --> gt kps
    # max_delta_ij --> gt limbs
        K = len(self.keypoint_names)
        # B batch
        B, _, _, _ = image.shape
        outW, outH = self.outsize

        # eg: (None, 440, 7, 7) --- 440 = 15*60 + 5*5*14     (len(edges)=14)
        feature_map = self.forward(image)
        resp = feature_map[:, 0 * K:1 * K, :, :]
        conf = feature_map[:, 1 * K:2 * K, :, :]
        x = feature_map[:, 2 * K:3 * K, :, :] # x.shape = [None, K, 7, 7]
        y = feature_map[:, 3 * K:4 * K, :, :]
        w = feature_map[:, 4 * K:5 * K, :, :]
        h = feature_map[:, 5 * K:6 * K, :, :]
        e = feature_map[:, 6 * K:, :, :].reshape((
            B,
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW
        ))

        (rx, ry), (rw, rh) = self.restore_xy(x, y), self.restore_size(w, h)
        (rtx, rty), (rtw, rth) = self.restore_xy(tx, ty), self.restore_size(tw, th)
        ious = iou((rx, ry, rw, rh), (rtx, rty, rtw, rth))

        # add weight where can't find keypoint
        xp = get_array_module(max_delta_ij)
        zero_place = xp.zeros(max_delta_ij.shape).astype(self.dtype)


        # used for generate onnx model
        try:
            max_delta_ij = max_delta_ij.array
            delta = delta.array
        except:
            pass

        zero_place[max_delta_ij < 0.5] = 0.0005
        weight_ij = xp.minimum(max_delta_ij + zero_place, 1.0)

        xp = get_array_module(delta)
        # add weight where can't find keypoint
        zero_place = xp.zeros(delta.shape).astype(self.dtype)
        zero_place[delta < 0.5] = 0.0005
        weight = xp.minimum(delta + zero_place, 1.0)

        half = xp.zeros(delta.shape).astype(self.dtype)
        half[delta < 0.5] = 0.5

        loss_resp = F.sum(F.square(resp - delta), axis=tuple(range(1, resp.ndim)))
        loss_iou = F.sum(delta * F.square(conf - ious), axis=tuple(range(1, conf.ndim)))
        loss_coor = F.sum(weight * (F.square(x - tx - half) + F.square(y - ty - half)), axis=tuple(range(1, x.ndim)))
        loss_size = F.sum(weight * (F.square(F.sqrt(w + EPSILON) - F.sqrt(tw + EPSILON)) +
                                    F.square(F.sqrt(h + EPSILON) - F.sqrt(th + EPSILON))),
                          axis=tuple(range(1, w.ndim)))
        loss_limb = F.sum(weight_ij * F.square(e - te), axis=tuple(range(1, e.ndim)))

        loss_resp = F.mean(loss_resp)
        loss_iou = F.mean(loss_iou)
        loss_coor = F.mean(loss_coor)
        loss_size = F.mean(loss_size)
        loss_limb = F.mean(loss_limb)

        loss = self.lambda_resp * loss_resp + \
            self.lambda_iou * loss_iou + \
            self.lambda_coor * loss_coor + \
            self.lambda_size * loss_size + \
            self.lambda_limb * loss_limb

        reporter.report({
            'loss': loss,
            'loss_resp': loss_resp,
            'loss_iou': loss_iou,
            'loss_coor': loss_coor,
            'loss_size': loss_size,
            'loss_limb': loss_limb
        }, self)

        return loss

    def predict(self, image):
        K = len(self.keypoint_names)
        B, _, _, _ = image.shape
        outW, outH = self.outsize

        with chainer.using_config('train', False),\
                chainer.function.no_backprop_mode():
            feature_map = self.forward(image)

        resp = feature_map[:, 0 * K:1 * K, :, :]
        conf = feature_map[:, 1 * K:2 * K, :, :]
        x = feature_map[:, 2 * K:3 * K, :, :]
        y = feature_map[:, 3 * K:4 * K, :, :]
        w = feature_map[:, 4 * K:5 * K, :, :]
        h = feature_map[:, 5 * K:6 * K, :, :]
        e = feature_map[:, 6 * K:, :, :].reshape((
            B,
            len(self.edges),
            self.local_grid_size[1], self.local_grid_size[0],
            outH, outW
        ))

        return resp, conf, x, y, w, h, e
