---
title: "Issues with sparse softmax cross entropy in Keras"
layout: post
---


```python
import keras as k
import numpy as np
import pandas as pd
import tensorflow as tf
```

# Experimenting with sparse cross entropy

I have a problem to fit a sequence-sequence model using the sparse cross entropy loss. It is not training fast enough compared to the normal categorical_cross_entropy. I want to see if I can reproduce this issue.

First we create some dummy data


```python
X = np.array([[1,2,3,4,5], [0,1,2,3,4]]).reshape(2,5)
Y = k.utils.to_categorical(X, 6)
```

Then we define a basic model which ties the weight from the embedding in the output layer


```python
input = k.layers.Input((None, ))
embedding = k.layers.Embedding(6, 10)
lstm_1 = k.layers.LSTM(10, return_sequences=True)
```


```python
embedding_input = embedding(input)
lstm_1 = lstm_1(embedding_input)
```


```python
lambda_layer = k.layers.Lambda(lambda x: k.backend.dot(
        x, k.backend.transpose(embedding.embeddings)))
```


```python
lambd = lambda_layer(lstm_1)
```


```python
softmax = k.layers.Activation('softmax')(lambd)
arg_max = k.layers.Lambda(lambda x: k.backend.argmax(x, axis=2))(softmax)
```


```python
model = k.Model(inputs = input, outputs=lambd)
model_sparse = k.Model(inputs = input, outputs=lambd)
```

Now we want to compile our model using, first the categorical_crossentropy loss to make sure everything runs fine. We want to make sure we are tracking accuracy as well, we need to implement this function ourselves...


```python
def sparse_loss(target, output):
    # Reshape into (batch_size, sequence_length)
    output_shape = output.get_shape()
    targets = tf.cast(tf.reshape(target, [-1]), 'int64')
    logits = tf.reshape(output, [-1, int(output_shape[-1])])
    print('logits ',logits.get_shape())
    res = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets,
        logits=logits)
    if len(output_shape) >= 3:
        # if our output includes timestep dimension
        # or spatial dimensions we need to reshape
        res = tf.reduce_sum(res)
        return(res)
    else:
        return(res)
def normal_loss(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    return(tf.reduce_sum(loss))
```


```python
model.compile(k.optimizers.SGD(lr=1), normal_loss,
             target_tensors=[tf.placeholder(dtype='int32', shape=(None, None))])
model_sparse.compile(k.optimizers.SGD(lr=1), sparse_loss,
                    target_tensors=[tf.placeholder(dtype='int32', shape=(None, None))])
# model_sparse.compile(k.optimizers.SGD(lr=1), sparse_loss)

```

    logits  (?, 6)



```python
print(model.evaluate(X, X))
print(model_sparse.evaluate(X, X))
```

    2/2 [==============================] - 0s 48ms/step
    17.917783737182617
    2/2 [==============================] - 0s 113ms/step
    17.917783737182617



```python
model_sparse.fit(X,X, epochs=10)
```

    Epoch 1/10
    2/2 [==============================] - 1s 388ms/step - loss: 17.9178
    Epoch 2/10
    2/2 [==============================] - 0s 4ms/step - loss: 17.9026
    Epoch 3/10
    2/2 [==============================] - 0s 7ms/step - loss: 17.8617
    Epoch 4/10
    2/2 [==============================] - 0s 5ms/step - loss: 17.7170
    Epoch 5/10
    2/2 [==============================] - 0s 8ms/step - loss: 17.2932
    Epoch 6/10
    2/2 [==============================] - 0s 8ms/step - loss: 16.3683
    Epoch 7/10
    2/2 [==============================] - 0s 9ms/step - loss: 14.0281
    Epoch 8/10
    2/2 [==============================] - 0s 5ms/step - loss: 10.6938
    Epoch 9/10
    2/2 [==============================] - 0s 6ms/step - loss: 10.7706
    Epoch 10/10
    2/2 [==============================] - 0s 10ms/step - loss: 16.4204





    <keras.callbacks.History at 0x11590c1d0>




```python
model.fit(X, X, epochs=10)
```

    Epoch 1/10
    2/2 [==============================] - 0s 248ms/step - loss: 15.0457
    Epoch 2/10
    2/2 [==============================] - 0s 6ms/step - loss: 11.0899
    Epoch 3/10
    2/2 [==============================] - 0s 7ms/step - loss: 6.4245
    Epoch 4/10
    2/2 [==============================] - 0s 8ms/step - loss: 4.6941
    Epoch 5/10
    2/2 [==============================] - 0s 7ms/step - loss: 3.1935
    Epoch 6/10
    2/2 [==============================] - 0s 7ms/step - loss: 2.8392
    Epoch 7/10
    2/2 [==============================] - 0s 8ms/step - loss: 1.1097
    Epoch 8/10
    2/2 [==============================] - 0s 8ms/step - loss: 0.6671
    Epoch 9/10
    2/2 [==============================] - 0s 9ms/step - loss: 0.5167
    Epoch 10/10
    2/2 [==============================] - 0s 12ms/step - loss: 0.4167





    <keras.callbacks.History at 0x116a14ba8>


