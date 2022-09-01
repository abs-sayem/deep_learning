## **Making new Layers and Models via Subclassing**
**Author:** [Abs Sayem](https://github.com/abs-sayem)<br>
**Date Created:** 2022/08/27<br>
**Last Modified:** 2022/08/31<br>
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