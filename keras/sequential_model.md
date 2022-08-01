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

#### **Creating a Sequential Model**

We can create a Sequential model by passing a list of layers to the Sequential constructor:
```
model = keras.Sequential([
    layers.Dense(2, activation='relu'),
    layers.Dense(3, activation='relu'),
    layers.Dense(4),
])
```
We can also create a Sequential model incrementally via the `add()` method:
```
model = keras.Sequential()
model.add(layers.Dense(2, activation='relu')),
model.add(layers.Dense(3, activation='relu')),
model.add(layers.Dense(2),
```
Also note that- the Sequential constructor accepts a `name` argument, just like any layer or model in Keras. This is useful to annotate TensorBoard grphs with semantically meaningfull names:
```
model = keras.Sequential(name='my_sequential_model')
model.add(layers.Dense(2, activation='relu', name='layer1')),
model.add(layers.Dense(3, activation='relu', name='layer2')),
model.add(layers.Dense(4, name='layer3'),
```
#### **Specifying the Input Shape in Advance**

Generally, all layers in Keras need to know the shape of their inputs in order to be able to create their weights. So, when we create a layer like this, initially it has no weights:
```
layers = layers.Dense(2)
layers.weights  # Empty
```
It creates its weights the first time it is called on an input, since the shape of the weights depends on the shape of the inputs. Naturally, when we instantiate a Sequential model without an input, it isn't built and calling `model.weights` results an error. The weights are created when the model first sees some input data.

**Giving Input Shape:** 