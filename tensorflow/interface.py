import onnx
from onnx_tf.backend import prepare
import warnings
import numpy as np
import ipdb; pdb = ipdb.set_trace

warnings.filterwarnings('ignore')
#  mxmodel = onnx.load('ppn.onnx')
chmodel = onnx.load('bestmodel.onnx')


ch_tf = prepare(chmodel)
model = ch_tf
#  input_last = {model.inputs[-1]:np.array([1,3,224,224])}
inputs = [ i for i in model.inputs] 
inputs[-1] = np.zeros(shape=(1,3,224,224), dtype=np.float32)

outputs = model.outputs[0]
model.run(inputs, outputs=outputs)
