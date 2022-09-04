## **Making new Layers and Models via Subclassing**
**Author:** [Abs Sayem](https://github.com/abs-sayem)<br>
**Date Created:** 2022/08/27<br>
**Last Modified:** 2022/09/01<br>
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