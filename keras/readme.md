### **Keras and Tensorflow-2**
###### **Keras is a deep learning API, running on top of the ***Tensorflow***. It is written in Python. Keras is the high-level API of Tensorflow 2: an approachable, highly-productive interface for solving machine learning problems with a focus on modern deep learning.**

#### **Contact with Keras**
###### **The core data structures of Keras are ***layers*** and ***models***. The simplest type of model is `sequential model`, a linear stack of layers.**

Here is the sequential model---
```
from tensorflow.keras,models import Sequential

model = Sequential()
```
Stacking layers is as easy as ***.add()***:
```
from tensorflow.keras.layers import Dense

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
```
Once the model is ready, we can configure its learning process with ***.compile()***:
```
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```
We can further configure the optimizer. The Keras philosophy is to keep things simple also allowing users to be fully in control when need bia subclassing.
```
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True))
```
Now, we can literate the model on training data in batches:
```
# x_train and y_train are numpy array
model.fit(x_train, y_train, epochs=5, batch_size=32)
```
Evaluate the test loss and metrics in one line:
```
loss_and_metrics = model.evaluade(x_test, y_test, batch_size=128)
```
Generate prediction on new data:
```
classes = model.predict(x_test, batch_size=128)
```
**Incremental Learning:** *In much the same way, we were able to train and evaluate a simple neural network above in a few lines, we can use Keras to quickly develop new training procedures or exotic model architectures which requires incremental learning at each step. Here's a low-level training loop example, combining Keras functionality with the Tensorflow ***GradientTape***.*
```
import tensorflow as tf

# Prepare an optimizer
optimizer = tf.keras.optimizers.Adam()
# Prepare a loss function
loss_fn = tf.keras.losses.k1_divergence

# Iterate over the batches of a dataset
for inputs, targets in dataset:
    # Open a GradientTape
    with tf.GradientTape() as tape:
        # Forward pass
        predictions = model(inputs)
        # Compute the loss value for this batch
        loss_value = loss_fn(targets, predictions)
    
    # Get gradients of loss wrt the weights
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```
#### Installation and Compatibility
Keras comes packaged with Tensorflow 2 as `tensorflow.keras`. To start using Keras, simply install Tensorflow 2 using the command `pip install tensorflow`.

Keras/Tensorflow are compatible with:
* Python 3.7 or above
* Windows 7 or later