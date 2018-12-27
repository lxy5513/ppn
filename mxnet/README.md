## use onnx model by onnx 


#### 如何保存module

```python
# 模型前缀, 自定义
model_prefix = 'mx_ppn'
checkpoint = mx.callback.do_checkpoint(model_prefix)
```



```python
arg_params, aux_params = module.get_params()
module.set_params(arg_params, aux_params)
# 保存模型参数与symbol   epoch 要自己写
callback(epoch, module.symbol, arg_params, aux_params)
```




#### 不仅加载参数，同时加载 Symbol

```python
# epoch 写和保存是一样的

sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, epoch)


# assign the loaded parameters to the module
mod.set_params(arg_params, aux_params)


```

