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