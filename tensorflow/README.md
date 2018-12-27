## 未完成 

无论是由chainer还是有mxnet生成的onnx模型，都不可以在tensorflow中使用。  
原因可能是tf是静态图结构，原模型有些多余的inputs必须输入到模型， (在mxnet、coreML中可以忽略) 
所以产生错误，在tf官方文档中，对于import也是experiment阶段。  

