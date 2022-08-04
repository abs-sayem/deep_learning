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
###### **The **Keras Functionl API** is a way to create models that are more flexible than the Sequential API. The Functional API can handle models with non-linear topology, shared layers, and multiple inputs or outputs.**
###### **The main idea is that a deep learning model is usually a directed acyclic graph (DAG) of layers. So the functional API is a way to build ***graphs of layers***.**

###### **Consider the following model:**
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
###### **This is a basic graph with three layers. To build this model using the functional API, start by creating an input node:**
```
inputs = keras.Input(shape=(784,))
# If, for example we have an image input with a shape of (32,32,3), we would use
img_inputs = keras.Input(shape=(32,32,3))
# We can check out the shape and data type
inputs.shape
inputs.dtype
```
###### **We can create a new node in the graph of layers by calling a layer on this input object: The "layer call" action is like drawing an arrow from "inputs" to the layer we created.**
```
dense = layers.Dense(64, activation='relu')
x = dense(inputs)   # Here, we are passing the inputs to the "dense" layer, and got "x" as the output.
```
###### **Let's add a few more layers to the graph of layers:**
```
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)
```
###### **Now, we can create a `model` by specifying its inputs and outputs in the graph of layers:**
```
model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
# Let's check out the model summary
model.summary()
```
###### **We can also plot the model as graph with input and output shapes**
```
keras.utils.plot_model(model, "mnist_model_info.png", show_shapes=True)
```

### **Training Evaluation and Inference**
###### **Training, Evaluation and Inference work exactly in the same way for models built using functional api as for `sequential` models.<br>The `model` class offers a built-in training loop(the `fit()` method) and a built-in evaluation loop(the `evaluate()` method). We can easily [customize these loops](https://keras.io/guides/customizing_what_happens_in_fit/).**
###### **Let's load the MNIST image data, reshape it inro vectors, fit the model on the data and then evaluate the model on the test data:**
```
# Load the data and split then into training and testing
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape the data (Normalization)
x_train = x_train.reshape(60000, 784).astype("float32")/255
x_test = x_test.reshape(10000, 784).astype("float32")/255

# Compile the model
model.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = keras.optimizers.RMSprop(),
        metrics = ["accuracy"]
)

# Fit the model to data
history = model.fit(x_train, y_train, batch_size=64, epochs=2, verbose=1, validation_split=0.2)

# Evaluate the model on test data
test_scores = model.evaluate(x_test, y_test, verbose=1)
print("Test Loss:", test_scores[0])
print("Test Accuracy:", test_scores[1])
```
*For more details, see the [training and evaluation](https://keras.io/guides/making_new_layers_and_models_via_subclassing/) guide.*

### **Save and Serialize**
###### **The standard way to save a model is to call `model.save()` to save the entire model as a single file. We can later create the same model from this file even if the model is no longer available.<br>This saved files includes- model architecture, model weight values(that are learned during training), model training config, optimizers and its state.**
```
# Save and then delete the model
model.save("model_name")
del model

# Recreate the same model from the file:
model = keras.models.load_model("model_name")
```
*For more details, read the model [serialization and saving](https://keras.io/guides/serialization_and_saving/) guide.*

### **Use the same Graph of Layers to define Multiple Models**
###### **In the functional API, models are created by specifying their inputs and outputs in a graph of layers. That means- a single graph of layers can be used to generate multiple models.<br>In the example below, we use the same stack of layers to instantiate two models: an `encoder` model that turns image inputs into 16-dimensional vectors, and an end-to-end `autoencoder` model for training.**
```
# Define encoder input output
encoder_input = keras.Input(shape=(28, 28, 1), name='img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

# Define decoder output
x = layers.Reshape(4, 4, 1)(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

# Create Encoder
encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()

# Create Autoencoder
autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
autoencoder.summary()
```
*Here, the decoding architecture is symmetrical to the encoding architecture, so the output shape is the same as the input shape(28, 28, 1).<br>The reverse of a `Conv2d` layer is `Conv2DTranspose` layer and the reverse of a `MaxPooling2D` layer is `UpSampling2D` layer.*

### **All Models are Callable, just like Layers**
###### **We can treat a model as a layer by calling it on an `Input` or on the output of another layer. When we call a model we aren't just reusing the architecture of the model, we also reuses its weights.<br>Here, we create an encoder model, a decoder model and chains to them in twp calls to obtain the autoencoder model.**
```
# Define encoder input output
encoder_input = keras.Input(shape=(28, 28, 1), name='original_img')
x = layers.Conv2D(16, 3, activation='relu')(encoder_input)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.Conv2D(16, 3, activation='relu')(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

# Define decoder input output
decoder_input = keras.Input(shape=(16,), name='encoded_img')
x = layers.Reshape(4, 4, 1)(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
x = layers.Conv2DTranspose(32, 3, activation='relu')(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation='relu')(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation='relu')(x)

# Create Encoder
encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()

# Create Decoder
autoencoder = keras.Model(decoder_input, decoder_output, name='decoder')
autoencoder.summary()

# Create Autoencoder
autoencoder_input = keras.Input(shape=(28, 28, 1), name='img')
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model(autoencoder_input, decoded_img, name='autoencoder')
autoencoder.summary()
```
###### **As we see, the model can be nested: a model can contain sub-models. A common case for model nesting is `ensembling`. Here is how to ensemble a set of models into a single model that averages their predictions:**
```
def getModel():
    inputs = keras.Input(shape=(128,))
    outputs = layers.Dense(1)(inputs)
    return(keras.Model(inputs, outputs))

model1 = getModel()
model2 = getModel()
model3 = getModel()

inputs = keras.Input(shape=(128,))
y1 = model1(inputs)
y2 = model2(inputs)
y3 = model3(inputs)
outputs = layers.average([y1, y2, y3])
ensemble_model = keras.Model(inputs=inputs, outputs=outputs)
```