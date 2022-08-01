## **The Sequential Model**
##### **Setup**
```
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```
##### **When to Use a Sequential Model**

A *sequential* model is appropriate for a **plain stack of layers** where each layer has **exactly one input tensor and one output tensor**.

The following `sequential` model:
```
# Define a Sequential model with 3 layers
model = keras.Sequential([
    layers.Dense(2, activation='relu', name='layer1'),
    layers.Dense(3, activation='relu', name='layer2'),
    layers.Dense(2, name='layer3'),
])
# Call the model on a test input
x = tf.ones((3,3))
y = model(x)
```
is equivalent to this function:
```
# Create 3 layers
layer1 = layers.Dense(2, activation='relu', name='layer1'),
layer2 = layers.Dense(3, activation='relu', name='layer2'),
layer3 = layers.Dense(2, name='layer3'),

# Call layers on a test input
x = tf.ones((3,3))
y = layer3(layer2(layer1(x)))
```
A Sequential model is not appropriate when:
* model has multiple inputs or multiple outputs
* any of the layers has multiple input or multiple outputs
* we nedd to do layer sharing
* we want non-linear topology(e.g. a residual connection, a multi-branch model)
