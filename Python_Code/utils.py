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


import tensorflow as tf
from tensorflow import keras


def proto_loss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries):

    # compute prototype for a given support set
    prototypes = tf.reduce_mean(tf.reshape(x_latent,[num_classes, num_support, -1]), axis=1)
    # compute euclidean distance
    tiled_proto = tf.tile(tf.expand_dims(prototypes, axis=0), (num_classes*num_queries, 1, 1))
    tiled_queries = tf.tile(tf.expand_dims(q_latent, axis=1), (1, num_classes, 1))
    distances = tf.reduce_mean(tf.square(tiled_proto - tiled_queries), axis=2)
    # compute the loss for protonet
    log_probs = tf.reshape(tf.nn.log_softmax(-distances), [num_classes, num_queries, -1])
    loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(labels_onehot, log_probs), axis=-1), [-1]))
    # compute accuracy
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(log_probs, axis=-1), tf.argmax(labels_onehot, axis=-1)), tf.float32))
    
    return loss, acc


def meta_train_step_proto(embed, clf, optimizer, x, q, labels, num_classes, num_support, num_queries):
    
    labels = tf.reshape(labels,[num_classes,num_queries])
    labels = keras.utils.to_categorical(labels, num_classes=num_classes)
    with tf.GradientTape() as tape:
        x_latent = embed(x,training=False)
        x_latent = clf(x_latent,training=True)
        q_latent = embed(q, training=False)
        q_latent = clf(q_latent,training=True)
        loss, acc = proto_loss(x_latent,q_latent,labels,num_classes,num_support,num_queries)
    gradients = tape.gradient(loss,clf.trainable_variables)
    optimizer.apply_gradients(zip(gradients,clf.trainable_variables))

    return loss, acc


def meta_test_step_proto(embed, clf, x, q, labels, num_classes, num_support, num_queries):

    labels = tf.reshape(labels,[num_classes,num_queries])
    labels = keras.utils.to_categorical(labels, num_classes=num_classes)
    x_latent = embed(x,training=False)
    x_latent = clf(x_latent,training=True)
    q_latent = embed(q, training=False)
    q_latent = clf(q_latent,training=True)
    loss, acc = proto_loss(x_latent,q_latent,labels,num_classes,num_support,num_queries)
    
    return loss, acc


def meta_train_step_softmax(embed, clf, optimizer, x, xlabels, q, qlabels,nclasses):
    loss_fn = keras.losses.CategoricalCrossentropy()
    acc_fn = keras.metrics.CategoricalAccuracy()
    xlabels = keras.utils.to_categorical(xlabels, num_classes=nclasses)
    qlabels = keras.utils.to_categorical(qlabels, num_classes=nclasses)
    with tf.GradientTape() as tape:
        x_latent = embed(x,training=False)
        x_predict = clf(x_latent,training=True)
        q_latent = embed(q, training=False)
        q_predict = clf(q_latent,training=True)
        x_loss = loss_fn(xlabels,x_predict)
        q_loss = loss_fn(qlabels,q_predict)
        acc_fn.update_state(qlabels,q_predict)
    gradients = tape.gradient(x_loss,clf.trainable_variables)
    optimizer.apply_gradients(zip(gradients,clf.trainable_variables))

    return q_loss, acc_fn.result()


def meta_test_step_softmax(embed, clf, q, labels, num_classes):
    acc_fn = keras.metrics.CategoricalAccuracy()
    labels = keras.utils.to_categorical(labels, num_classes=num_classes)
    latent = embed(q,training=False)
    predict = clf(latent,training=True)
    acc_fn.update_state(labels,predict)
    
    return acc_fn.result(), predict