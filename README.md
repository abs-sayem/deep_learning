### **Deep Learning**

###### **Spliting Dataset**
* Suppose we have a dataset where we seperate `features` and `target`. We will split this dataset into train and test set.
```python
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets (test size = 20%)
x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
```
* `test_size`: portion of total dataset that will use for validation while training
* `random_state:`
    * specifying random state ensures that the random split is the same for every time.
    * `42` is a random value, it can be any value.
    * if not specified, the split will be different for every time.

###### **Something Else**