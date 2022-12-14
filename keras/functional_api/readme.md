## **The Functional API**
**Author:** [Abs Sayem](https://github.com/abs-sayem)<br>
**Date Created:** 2022/08/02<br>
**Last Modified:** 2022/08/27<br>
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

### **Training Evaluation and Inference**
###### **Training, Evaluation and Inference work exactly in the same way for models built using functional api as for `sequential` models.<br>The `model` class offers a built-in training loop(the `fit()` method) and a built-in evaluation loop(the `evaluate()` method). We can easily [customize these loops](https://keras.io/guides/customizing_what_happens_in_fit/).**

### **Save and Serialize**
###### **The standard way to save a model is to call `model.save()` to save the entire model as a single file. We can later create the same model from this file even if the model is no longer available.<br>This saved files includes- model architecture, model weight values(that are learned during training), model training config, optimizers and its state.**

### **Use the same Graph of Layers to define Multiple Models**
###### **In the functional API, models are created by specifying their inputs and outputs in a graph of layers. That means- a single graph of layers can be used to generate multiple models.<br>In the example below, we use the same stack of layers to instantiate two models: an `encoder` model that turns image inputs into 16-dimensional vectors, and an end-to-end `autoencoder` model for training.**

### **All Models are Callable, just like Layers**
###### **We can treat a model as a layer by calling it on an `Input` or on the output of another layer. When we call a model we aren't just reusing the architecture of the model, we also reuses its weights.<br>Here, we create an encoder model, a decoder model and chains to them in twp calls to obtain the autoencoder model.**

### **Manipulate Complex Graph Topologies**
**Models with multiple inputs and outputs**
###### **The functional api makes it easy to manipulate inputs and outputs but Seqiential API does not.<br>For example, if we build a system for ranking customer issue tickets by priority and routing them to the correct department, then the model will have three input-**
* the title of the ticket(text input)
* the text body of the ticket(text input) and
* any tags added by the user(categorical input)
###### **The model will have two outputs-**
* the priority score between 0 and 1 (scalar sigmoid output) and
* the department that should handle the ticket (softmax output over the set of departments).

**A ResNet Model**
###### **In addition to models with multiple inputs and outputs, the functional API makes it easy to manipulate non-linear connectivity topologies -- there are models with layers that are not connected sequentially, which the `Sequential` API cannot handle.<br>A common use case for this is residual connections.**

### **Shared Layers**
###### **Shared layers are layer instances that are reused multiple times in the same model -- they learn features that correspond to multiple paths in the graph-of-layers.<br>Shared layers are often used to encode input from similar spaces(say, two different pieces of text that feature similar vocabulary). They enable sharing of information across these diferent inputs, and make it possible to train such a model on less data.**
###### **To share a layer in the functional API, call the same layer instance multiple times.**

### **Extract and Reuse Nodes in the Graph-of-Layers**
###### **Because the graph of layers we are manipulating is a static data structure, it can be accessed and inspected. And this is how we are able to plot functional models as images.<br>This also means that we can access the activations of intermediate layers and reuse them elsewhere -- which is very usefull for feature extraction.**

### **Extend the API Using Custom Layers**

###### **`tf.keras` includes a wide range of built-in layers, for example:**
* Convolutional Layers: `Conv1D`, `Conv2D`, `Conv3D`, `Conv2DTranspose`
* Pooling Layers: `MaxPooling1D`, `MaxPooling2D`, `MaxPooling3D`, `AverageMaxpooling1D`
* RNN Layers: `GRU`, `LSTM`, `ConvLSTM2D`
* `BatchNormalization`, `Dropout`, `Embedding` etc.
###### **But we also easily extend the API by creating our own layers by subclassing the `layer` class and implement:**
* `call` method- that spacifies the computation done by the layer
* `build` method- that creates the weights of the layer

### **When to Use the Functional API**
###### **Should we use the Keras functional API to create a new model or just subclass the `Model` class directly?- In general, the functional API is higher-level, easier and safer, and has a number of features that subclassed models do not support.**
###### **Model subclassing provides greater flexibility when building models that are not easily expressible as directed acyclic graphs of layers. For example, we could not implement a Tree-RNN with the functional API and would have to subclass `Model` directly.<br>For in-depth look at the differencrs between the functional API and model subclassing, read [What are Symbolic and Imperative APIs in Tensorflow2.0](https://blog.tensorflow.org/2019/01/what-are-symbolic-and-imperative-apis.html).<br>**
#### **Functional API Strengths**
###### **There are some properties that are also true for Sequential models but are not true for subclassed models:**
* **Less Berbose:** There is no `super(MyClass, self).__init__(...)`, no `def call(self, ...)` etc.
* **Model Validation while Defining its Connectivity Graph:** In the functional API, the input specification (shape, dtype) is created in advance(using `input`). Every time we call a layer, the layer checks that the specification passed to it matches its assumptions, and it will raise a helpful error message if not. This guarantees that any model we can build with the functional API will run.
* **A Functional Model is Plottable and Inspectable:** We can plot the model as a graph and can easily access intermediate nodes in this graph.
* **A Functional API can be Serialized or Cloned:** Because a functional model is a data structure rather than a piece of code, it is safely serializable and can be saved as a single file that allows us to recreate the exact same model without having access to any of the original code.<br>To serialize a subclassed model, it is necessary for the implementer to specify a `get_config()` and `from_config()` method at the model level.
#### **Functional API Weakness**
* **Doesn't Support Dynamic Architectures:** The functional API treats models as DAGs of layers. This is true for most of the architectures, but not for all -- for example, recursive networks ot Tree RNNs do not follow this assumption and connot be implemented in the functional API.

### **Mix-and-Match API Styles**
###### **Choosing between the functional API and Model subclassing isn't a decision. All models in the `tf.keras` API can interect with each other, whether they're `Sequential` models, functional models, or subclassed models.<br>We can always use a `Functional` or `Sequential` model as part of a subclassed model or layer.**
###### **We can use any subclassed layer or model in the functional API as long as it implements a `call` method that follows one of the following patterns:**
* `call(self, inputs, **kwargs)` -- *Where `inputs` is a tensor or a nested structures of tensors(a list of tensors), and where `**kwargs` are non-tensor arguments(non-inputs).*
* `call(self, inputs, training=None, **kwargs)` -- *Where `training` is a boolean indicating whether the layer should behave in training mode and inference mode.*
* `call(self, inputs, mask=None, **kwargs)` -- *Where `mask` is a boolean mask tensor(useful for RNNs, for instance).*
* `call(self, inputs, training=None, mask=None, **kwargs)` -- *Of course, we can have both masking and training-specific behavior at the same time.*
###### **Additionally, if we implement the `get_config` method on our custom layer or model, the functional models we create will still be serializable and cloneable.**