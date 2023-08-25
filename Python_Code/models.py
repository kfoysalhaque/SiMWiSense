"""
    Copyright (C) 2023 Khandaker Foysal Haque
    contact: haque.k@northeastern.edu
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def ConvLayer(input, filters, kernelsize):
    
    x = layers.Conv2D(filters,kernelsize,padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    return x

def EmbeddingNet(inputshape):

    input = keras.Input(shape=inputshape)
    x = ConvLayer(input,64,3)
    x = layers.MaxPool2D()(x)
    x = ConvLayer(x,64,3)
    x = layers.MaxPool2D()(x)
    x = ConvLayer(x,64,3)
    x = layers.MaxPool2D()(x)
    x = ConvLayer(x,64,3)
    output = layers.GlobalAveragePooling2D()(x)

    return keras.Model(inputs=input,outputs=output)

def Decoder(inputshape,nclasses,softmax=True):
    
    input = keras.Input(shape=inputshape)
    output = layers.Dense(nclasses)(input)
    if softmax:
        output = layers.Softmax()(output)
    return keras.Model(inputs=input,outputs=output)


class Embed_Dec(keras.Model):

    def __init__(self,inputshape, nclasses):
        super (Embed_Dec,self).__init__()
        self.embedding = EmbeddingNet(inputshape)
        self.decoder = Decoder(self.embedding.layers[-1].output.shape[1:],nclasses)

    def compile(self,optimizer,loss):
        super(Embed_Dec,self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.loss_metric = keras.metrics.CategoricalCrossentropy(name="loss")
        self.acc_metric = keras.metrics.CategoricalAccuracy(name="accuracy")

    @property
    def metrics(self):
        return[self.loss_metric, self.acc_metric]
        
    # custom fit
    def train_step(self, data):
        input, targ = data
        with tf.GradientTape() as tape: 
            embed_output = self.embedding(input,training=True)
            prediction = self.decoder(embed_output,training=True)
            loss_value = self.loss(y_true=targ,y_pred=prediction)
        
        variables = self.trainable_variables
        grad = tape.gradient(loss_value,variables)
        self.optimizer.apply_gradients(zip(grad,variables))
        self.loss_metric.update_state(targ,prediction)
        self.acc_metric.update_state(targ,prediction)
        return {"loss":self.loss_metric.result(), "accuracy":self.acc_metric.result()}

    def test_step(self, data):
        input, targ = data
        embed_output = self.embedding(input,training=False)
        prediction = self.decoder(embed_output,training=False)
        self.loss_metric.update_state(targ,prediction)
        self.acc_metric.update_state(targ,prediction)
        return {"loss":self.loss_metric.result(), "accuracy":self.acc_metric.result()}

class customCallback(keras.callbacks.Callback):
    """
    a custom callback for training Embed_Dec
    """
    def __init__(self,dir,patience):
        super(keras.callbacks.Callback, self).__init__()
        self.checkpoint_path = dir # path to store best model
        self.patience = patience # patience for early stopping
    
    def on_train_begin(self,logs=None):
        self.count = 0 # count for non-best epoch
        self.best_score = np.Inf

    def on_epoch_end(self,epoch,logs=None):
        current = logs.get("val_loss")
        if np.less(current,self.best_score):
            self.best_score = current
            self.count = 0
            self.model.embedding.save(self.checkpoint_path) # save the best model
        else:
            self.count += 1
            if self.count >= self.patience:
                self.model.stop_training = True