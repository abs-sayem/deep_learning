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
layer1 = layers.Dense(2, activation='relu', name='layer1')
layer2 = layers.Dense(3, activation='relu', name='layer2')
layer3 = layers.Dense(2, name='layer3')

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
model.add(layers.Dense(2, activation='relu'))
model.add(layers.Dense(3, activation='relu'))
model.add(layers.Dense(2))
```
Also note that- the Sequential constructor accepts a `name` argument, just like any layer or model in Keras. This is useful to annotate TensorBoard grphs with semantically meaningfull names:
```
model = keras.Sequential(name='my_sequential_model')
model.add(layers.Dense(2, activation='relu', name='layer1'))
model.add(layers.Dense(3, activation='relu', name='layer2'))
model.add(layers.Dense(4, name='layer3'))
```
#### **Specifying the Input Shape in Advance**

Generally, all layers in Keras need to know the shape of their inputs in order to be able to create their weights. So, when we create a layer like this, initially it has no weights:
```
layers = layers.Dense(2)
layers.weights  # Empty
```
It creates its weights the first time it is called on an input, since the shape of the weights depends on the shape of the inputs. Naturally, when we instantiate a Sequential model without an input, it isn't built and calling `model.weights` results an error. The weights are created when the model first sees some input data.

**Giving Input Shape:**
##### **Using `input` Object:**
We should start our model by passing an `input` object to the model, so that it knows its input shape from the beginning:
```
model = keras.Sequential()
model.add(keras.Input(shape=(4,)))
model.add(layers.Dense(2, activation='relu'))
```
##### **Using `input_shape` Argument:**
A simple alternative is to just pass an `input_shape` argument to the first layer:
```
model = keras.Sequential()
model.add(layers.Dense(2, activation='relu', input_shape=(4,)))
```
*Models built with a predefined input shape like this always have weghts(even before seeing any data) and always have a defined output shape. In general, it's recommended best practice to always specify the input shape of a Sequential model in advance if you know what it is.*

#### What to do after we have a model
Once our model architecture is ready-
* train the model, evaluate it and run inference
* save the model to local disk and store it
* speed up model training by leveraging multiple GPUs

#### A Common Debugging Workflow (A CNN Persfective)
When building a new Sequential architecture, it's useful to incrementally stack layers with `add()` and frequently print model summaries. For instance, this enables us to monitor how a stack of `conv2D` and `MaxPooling2D` layers is downsampling image feature maps:
```
model = keras.Sequential()
model.add(keras.Input(shape=(250, 250, 3)))                     # 250x250 RGB images
model.add(layers.Conv2D(32, 5, strides=2, activation='relu'))   # (123, 123, 32)
model.add(layers.Conv2D(32, 3, activation='relu'))              # (121, 121, 32)
model.add(layers.MaxPooling2D(3))                               # (40, 40, 32)

# Can we guess what the current output shape is at this point?
# Let's just print it-
model.summary()     # The shape is: (40, 40, 32) downsampled 250 to 40

# We can keep downsampling ...
model.add(layers.Conv2D(32, 3, activation='relu'))  # (38, 38, 32)
model.add(layers.Conv2D(32, 3, activation='relu'))  # (36, 36, 32)
model.add(layers.MaxPooling2D(3))                   # (12, 12, 32)
model.add(layers.Conv2D(32, 3, activation='relu'))  # (10, 10, 32)
model.add(layers.Conv2D(32, 3, activation='relu'))  # (8, 8, 32)
model.add(layers.MaxPooling2D(3))                   # (4, 4, 32)

# And now let's see the output shape
model.summary()     # Output Shape: (4, 4, 32)

# Now we can apply Global MaxPooling
model.add(layers.GlobalMaxPooling2D())

# Finally, add a classification layer
model.add(layers.Dense(10))
```

#### Feature Extraction with a Sequential Model
Once a Sequential model has been build, it behaves liake a **Functial API Model**. This means- evaery layer has an `input` and `output` attribute This attribute can be used- to quickly creating a model that extracts the outputs of all intermediate layers in a Sequential model:
```
# Build a Sequential Model
initial_model = keras.Sequential([
    keras.Input(shape=(250, 250, 3)),
    layers.Conv2D(32, 5, strides=2, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(32, 3, activation='relu'),
])

# Create a Feature Extractor
feature_extractor = keras.Model(
    inputs = initial_model.input,
    outputs = [layer.output for layer in initial_model.layers],
)

# Call the feature extractor on test input
x = tf.ones((1, 250, 250, 3))       # 1=???
features = feature_extractor(x)
```
Here is a similar example that only extract features from one layer:
```
# Build a Sequential Model
initial_model = keras.Sequential([
    keras.Input(shape=(250, 250, 3)),
    layers.Conv2D(32, 5, strides=2, activation='relu'),
    layers.Conv2D(32, 3, activation='relu', name='intermediate_layer'),
    layers.Conv2D(32, 3, activation='relu'),
])

# Create a Feature Extractor
feature_extractor = keras.Model(
    inputs = initial_model.input,
    outputs = initial_model.get_layer(name='intermediate_layer').output,
)

# Call the feature extractor on test input
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
```

#### **Transfer Learning with a Sequential Model**
Transfer learning consists of freezing the bottom layers in a model and only training the top layers. Let's have a look two common transfer learning blueprint involving Sequential models.

First, let's assume that we have a Sequential model and we want to freeze all layers except the last one. In this case, we would simply iterate over `model.layers` and set `layer.trainable=False` on each layer, except the last layer:
```
model = kares.Sequential([
    keras.Input(shape=(784)),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10),
])

# More likely, we would wnat to first load pre-trained weights
model.load_weights(...)

# Freeze all layers except the last one
for layer in model.layers[:-1]:
    layer.trainable = False

# Recompile and Train: It will only update the weights of the last layer
model.compile(...)
model.fit(...)
```
Another common blueprint is to use a Sequential model to stack **a pre-trained model** and some freshly initialized classification layers:
````
# Load a convolutional base with pre-trained weights
base_model = keras.applications.Xception(
    weights = 'imagenet',
    include_top = False,
    pooling = 'avg'
)

# Freeze the base model
base_model,trainable = False

# Use a Sequential model to add a trainable classifier on top
model = keras.Sequential([
    base_model,
    layers.Dense(1000),
])

# Compile and Train
model.compile(...)
model.fit(...)
```