### **The Functional API**
**Author:** Abs Sayem<br>
**Date Created:** 2022/08/02<br>
**Last Modified:** 2022/08/03<br>
**Description:** Guide to the Functional API

#### **Setup**
```
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```

#### **Introduction**
The **Keras Functionl API** is a way to create models that are more flexible than the Sequential API. The Functional API can handle models with non-linear topology, shared layers, and multiple inputs or outputs.

The main idea is that a deep learning model is usually a directed acyclic graph (DAG) of layers. So the functional API is a way to build *graps of layers*.