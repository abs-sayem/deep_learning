## **Making new Layers and Models via Subclassing**
**Author:** [Abs Sayem](https://github.com/abs-sayem)<br>
**Date Created:** 2022/08/27<br>
**Last Modified:** 2022/09/14<br>
**Description:** Guide to making `Layers` and `Model` objects via Subclassing

#### **Setup**
```
import tensorflow as tf
from tensorflow import keras
```

#### **The `Layer` Class: combination of state(weights) and computation**
###### **A layer encapsulates both a state(layer's weight) and a transformation from inputs to outputs. Note that- The weights `w` and `b` can be automatically tracked by the layer being set as layer attributes. We can also add weights to a layer using the `add_weights()` method.**
#### **Layers can have Non-Trainable Weights**
###### **Besides trainable weights, we can add non-trainable weights to a layer as well. These weights are meant not to be taken into account during backpropagation, when we are training the layer. It's part of `layer.weights`, but it gets categorized as a non-trainable weight.**
#### **Unknown Input: Deferring(delay/postpone) weight creation until the shape of the inputs is known**
###### **Our `Linear` layer above took an `input_dim` arguments that was used to compute the shape of the weights `w` and `b` in `__init__()`.<br>In many cases, we may not know in advance the size of the inputs and we would like to lazily create weights when that value becomes known, some time after instantiating the layer.<br>In the Keras API, it recommends creating layer weights in the `build(self, inputs_shape)` method of the layer.<br>The `__call__()` method of our layer will automatically run `build()` the first time it is called. In this way we can create a lazy layer which is easier to use.<br>Implementing `build()` separately as shown above nicely seperates creating weights only once from using weights in every call. Layer implementers are allowed to defer weight creation to the first `__call__()`, but need to take care that, later calls use the same weights. In addition, since `__call__()` is likely to be executed for the first time inside a `tf.function`, any variable creation that takes place in `__call__()` should be wrapped in a `tf.init_scope`.**
#### **Layers are Recursively Composable(Writeable)**
###### **If we assign a layer instance as an attribute of another layer, the outer layer will start tracking the weights created by the inner layer. Keras recommend creating such sublayers in the `__init__()` method and leave it to the first `__call__()` to trigger building their weights.**
#### **The `add_loss()` Method**
###### **While writing the call method, we can create loss tensors. That loss tensors, we will want to use later while writing our training loop. This is doable by calling `self.add_loss(value)`.<br>These losses(including those created by any inner layer) can be retrived via `layer.losses`. This property is reset at the start of every `__call__()` to the top-level layer, so that `layer,losses` always contains the loss values created during the last forward pass. Also, the `loss` property also contains regularization losses created for the weights of any inner layer.<br>These losses are meant to be taken when writing training loops. These losses also work seemlessly with `fit()` (they get automatically summed and added to the main loss, if any).**
#### **The `add_metric()` Method**
###### **Like `add_loss()`, there also has the `add_metric()` method- used for tracking the moving average of a quantity during training.<br>Consider a layer: a `logistic endpoint` layer - takes predictions and targets as input, computes the loss tracked via `add_loss()`, and then computes an accuracy scalar, which is tracks via `add_metric()`.**
#### **We can Optionally Enable Serialization on our Layers**
###### **If we need our custom layers to be serializable as part of a `Functional Model`, we can optionally implement a `get_onfig()` method.<br>** `[NB]` **The `__init__()` method of the base `Layer` class takes some keywords arguments, in particular a `name` and a `dtypes`. It's good practice to pass these arguments to the parent class in `__Init__()` and to include them in the layer config. If we need more flexibility when deserializing the layer from its config, we can also override the `from_config()` class method.**
#### **Privileged `training` Argument in the `call()` Method**
###### **Some layers, in particular the `BatchNormalization` layer and the `Dropout` layer, have different behaviors during training and inference. For such layers, it is the best practice to expose `training` (boolean) argument in the `call()` method.<br>By exposing this argument in `call()`, we enable the built-in training and evaluation loops(e.g. `fit()`) to correctly use the layer in training and inference.**
#### **Privileged `mask` Argument in the `call()` Method**
###### **The other privileged argument supported by `call()` is the `mask` method.<br>A mask is a boolean tensor used to skip certain input timesteps when processing timeseries data. We will find it in all Keras RNN layers.<br>Keras will automatically pass the correct `mask` arguments to `__call__()` for layers that support it, when a mask is generated by a prior layer. Mask-generating layers are the `Embedding` layer configured with `mask_zero=True` and the `Masking` layer.**
#### **The `Model` Class**
###### **in general, we will use the `Layer` class to define the inner computation blocks, and will use the `Model` class to define the outer model. For instance, in a ResNet50 model, we would have several ResNet blocks subclassing `Layer`, and a single `Model` enclosing the entire ResNet network.**
###### **The `Model` class has the same API as `Layer`, with the following differences:**
* *It exposes built-in training, evaluation and prediction loops(`model.fit()`, `model.evaluate()`, `model.predict()`).*
* *It exposes the list of its inner layers, via the `model.layers` property.*
* *It exposes saving and serialization APIs(save(), save_weights()...)*
###### **Meanwhile, the `Layer` class corresponds to what we refer to in the literature as a "layer"(as in `convolutional layer`, or `recurrent layer`) or as a "block"(as in `DNN`). And the `Model` class corresponds to what is referred to in the literature as a "model"(as in `deep learning model`) or as a "network"(as in `DNN`)**
#### **Putting all Together: an End-to-End Example**
###### **Here's what we've learned so far:**
* *A `Layer` encapsulate a state(created in `__init__()` or `build()`) and some computation(defined in `call()`).*
* *Layers can be recursively nested to create new, bigger computation blocks.*
* *Layers can create and track losses(typically regularization losses) as well as metrics, via `add_loss()` and `add_metric()`.*
* *The outer container, the thing we want to train, is a `Model`. A `Model` is just like a `Layer`, but with added training and serialization utilities.*
###### **We can put all of these things together into an end-to-end example: we can implement a Variational AutoEncoder(VAE) and train it on MNIST utilities.<br>Our VAE will be a subclass of `Model`, built as a nested composition of layers that subclass `Layer`. It will feature a regularization loss(KL divergence).**
#### **The `Functional API`: Beyond Object-Oriented Development**
###### **We can also build models using the [Functional API](https://github.com/abs-sayem/deep_learning/blob/main/keras/functional_api/readme.md). Importantly, choosing one style or onother, from object-oriented or functional-api, doesn't prevent us from leveraging components written in the other style: we can always mix and match.**