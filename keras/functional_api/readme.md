## **The Functional API**
**Author:** [Abs Sayem](https://github.com/abs-sayem)<br>
**Date Created:** 2022/08/02<br>
**Last Modified:** 2022/08/03<br>
**Description:** Guide to the Functional API

### **Setup**
```
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

### **Introduction**
###### **The **Keras Functionl API** is a way to create models that are more flexible than the Sequential API. The Functional API can handle models with non-linear topology, shared layers, and multiple inputs or outputs.** <br>
###### **The main idea is that a deep learning model is usually a directed acyclic graph (DAG) of layers. So the functional API is a way to build ***graps of layers***.**

Consider the following model:
```
(input: 784-dimensional vectors)
        !
[Dense (64 units, relu activation)]
        !
[Dense (64 units, relu activation)]
        !
[Dense (10 units, softmax activation)]
        !
(output: logits of a probability distribution over 10 classes)
```
This is a basic graph with three layers. To build this model using the functional API, start by creating an input node:
```
inputs = keras.Input(shape=(784,))
# If, for example we have an image input with a shape of (32,32,3), we would use
img_inputs = keras.Input(shape(32,32,3))
# We can check out the shape and data type
inputs.shape
inputs.dtype
```
We can create a new node in the graph of layers by calling a layer on this input object: The "layer call" action is like drawing an arrow from "inputs" to the layer we created.
```
dense = layers.Dense(64, activation='relu')
x = dense(inputs)   # Here, we are passing the inputs to the "dense" layer, and got "x" as the output.
```
Let's add a few more layers to the graph of layers:
```
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)
```
Now, we can create a `model` by specifying its inputs and outputs in the graph of layers:
```
model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
# Let's check out the model summary
model.summary()
```
We can also plot the model as graph with input and output shapes
```
keras.utils.plot_model(model, "mnist_model_info.png", show_shapes=True)
```