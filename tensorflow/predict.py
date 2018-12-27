import utils_coreML as utils
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import ImageDraw, Image
import PIL
import cv2
import os
import ipdb;pdb = ipdb.set_trace
from collections import namedtuple


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

def preprocess_image(path):
    f = Image.open(path)
    img = f.convert('RGB')
    img = np.asarray(img, dtype=np.float32)
    img = img.transpose((2, 0, 1))
    image = utils.resize(img, (224, 224))
    image = image.astype(np.float32)
    return image

def main(i=2):
    # 读取图片数据
    img_file = '/Users/liuxingyu/Pictures/pose'
    imgs = os.walk(img_file)
    imgs = list(imgs)[0][-1]
    img_path = os.path.join(img_file, imgs[i])
    img_path = '/Users/liuxingyu/Pictures/pose/056459408.jpg'

    image = preprocess_image(img_path)

    model = utils.get_model()
    ## 获取keypoints and joints
    feature_map = get_feature(model, image)

    humans = utils.get_humans_by_feature(feature_map)

    ## 画出图像
    pil_image = Image.fromarray(image.transpose(1, 2, 0).astype(np.uint8))
    pil_image = utils.draw_humans(
        pil_image=pil_image,
        humans=humans
    )
    # save
    pil_image.save('results/{}'.format(imgs[i]))

if __name__ == '__main__':
    main()
