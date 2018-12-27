#!/usr/bin/env python
# coding: utf-8



import onnx
import onnx_coreml
import coremltools 


#  model = onnx.load('../bestmodel.onnx') 
#  cml = onnx_coreml.convert(model)
#  print(type(cml))
#  cml.save('coreML.model')

import ipdb 
ipdb.set_trace()
# load model 
cml = coremltools.models.MLModel('./coreML.mlmodel') 




# ## Preprocess image


import numpy as np 
from IPython.display import display 
from PIL import Image
import PIL
import cv2

# 读取图片数据
img_file = '/Users/liuxingyu/Pictures/pose'
import os
imgs = os.walk(img_file)
imgs = list(imgs)[0][-1]


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

## 图片预处理
img1 = os.path.join(img_file, imgs[2]) 
dis_img = Image.open(img1)
# display(dis_img) ### display image
f = Image.open(img1)
img = f.convert('RGB')
img = np.asarray(img, dtype=np.float32) 
img = img.transpose((2, 0, 1))
image = resize(img, (224, 224))



inputs = cml.input_description
type(inputs)
# cml.output_description



from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


input_name = cml.get_spec().description.input[-1].name
output_name = cml.get_spec().description.output[0].name
input_name 
output_name

import ipdb;ipdb.set_trace()
mg_out = cml.predict({input_name: image})[output_name]

