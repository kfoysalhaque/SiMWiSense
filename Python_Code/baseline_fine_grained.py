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

import argparse
import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from dataGenerator import DataGenerator_Coarse, DataGenerator
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

# Constants
num_classes = 20
window_size = 50
epoch = 3

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_env', help='training Scenario')
    parser.add_argument('train_station', help='train station')
    parser.add_argument('test_env', help='testing Scenario')
    parser.add_argument('test_station', help='test station')
    parser.add_argument('model_save', help='Name of the model')
    parser.add_argument('NoOfSubcarrier', help='No Of Subcarrier')
    args = parser.parse_args()
    return args

# Data Paths
def get_data_paths(args):
    train_env = args.train_env
    train_station = args.train_station
    test_env = args.test_env
    test_station = args.test_station

    Bw = "80MHz"
    num_mon = "3mo"
    slots="Slots"
    model_save= args.model_save
    Train_dir= 'Train'
    Test_dir= 'Test'
    NoOfSubcarrier = int(args.NoOfSubcarrier)
    input_shape=(window_size, NoOfSubcarrier, 2)


    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    data_path = "../Data/fine_grained"
    train_dir = os.path.join(data_path, train_env, Bw, num_mon, train_station, slots, Train_dir )
    test_dir = os.path.join(data_path, test_env, Bw, num_mon, test_station, slots, Test_dir )
    model_dir = os.path.join(data_path, train_env, Bw, num_mon, train_station, model_save )

    return train_dir, test_dir, model_dir, NoOfSubcarrier

# Model Definition
def get_baseline_model(slice_size, classes, NoOfSubcarrier, fc1, fc2):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (3, 3), padding='same', strides=2, input_shape=(slice_size, NoOfSubcarrier, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.MaxPooling2D(pool_size=(2, 1)))
    model.add(layers.Flatten())
    model.add(layers.Dense(classes, activation='softmax'))

    model.summary()

    return model

def train_model(model, train_gen, val_gen, model_dir):
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=15, verbose=1, factor=0.5, min_lr=0.0001)
    checkpoint = ModelCheckpoint(model_dir, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, verbose=1)

    history = model.fit(
        x=train_gen,
        epochs=epoch,
        validation_data=val_gen,
        callbacks=[learning_rate_reduction, checkpoint, earlystopping],
        verbose=1
    )

    return history

def plot_accuracy(history, train_dir):
    import matplotlib.pyplot as plt

    plt.plot(history.history["accuracy"], label="Training acc")
    plt.plot(history.history["val_accuracy"], label="Validation acc")
    plt.legend()
    plt.savefig(os.path.join(train_dir, 'train_val_accuracy.png'), dpi=300)
    plt.show()



def evaluate_and_plot_confusion_matrix(model, test_gen, labels, train_dir):
    Y = test_gen.labels[test_gen.indexes]
    Y_true = np.zeros(len(Y))

    for i, e in enumerate(Y):
        Y_true[i] = labels.index(e)

    Y_pred = model.predict(test_gen)
    Y_pred = np.argmax(Y_pred, axis=1)

    cm = confusion_matrix(Y_true[:len(Y_pred)], Y_pred, normalize='true')
    plt.figure(figsize=(32, 32))
    ax = sns.heatmap(cm, cmap=plt.cm.Greens, annot=True, fmt='.1f', square=True, xticklabels=labels, yticklabels=labels)
    ax.set_ylabel('Actual', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.savefig(os.path.join(train_dir, 'confusion_matrix.png'), dpi=300)
    plt.show()


def main():
    args = parse_arguments()
    train_dir, test_dir, model_dir, NoOfSubcarrier = get_data_paths(args)
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T']


    # Model Definition
    model = get_baseline_model(window_size, len(labels), NoOfSubcarrier, fc1=256, fc2=128)

    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=15, verbose=1, factor=0.5, min_lr=0.0001)
    checkpoint = ModelCheckpoint(model_dir, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.05, patience=10, verbose=1)


    tr_csv = os.path.join(train_dir, 'train_set.csv')
    val_csv = os.path.join(train_dir, 'val_set.csv')
    test_csv = os.path.join(test_dir, 'test_set.csv')

    train_gen = DataGenerator(train_dir,tr_csv, NoOfSubcarrier, len(labels), (window_size, NoOfSubcarrier, 2), batchsize=64)
    val_gen = DataGenerator(train_dir,val_csv,NoOfSubcarrier, len(labels), (window_size, NoOfSubcarrier, 2), batchsize=64)
    test_gen = DataGenerator(test_dir,test_csv, NoOfSubcarrier, len(labels), (window_size, NoOfSubcarrier, 2),batchsize=64, shuffle=False)


    # Compiling the Model
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="categorical_crossentropy", metrics=["accuracy"])

    # Training the Model
    history = train_model(model, train_gen, val_gen, model_dir)


    # Plotting Accuracy
    plot_accuracy(history, train_dir)

    # Evaluating Model
    print("The validation accuracy is :", history.history['val_accuracy'])
    print("The training accuracy is :", history.history['accuracy'])
    print("The validation loss is :", history.history['val_loss'])
    print("The training loss is :", history.history['loss'])

    model = load_model(model_dir)
    final_loss, final_accuracy = model.evaluate(test_gen)
    print('Test Loss: {}, Test Accuracy: {}'.format(final_loss, final_accuracy))

    evaluate_and_plot_confusion_matrix(model, test_gen, labels, train_dir)

if __name__ == '__main__':
    main()
