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

from atexit import _ncallbacks
import os
from tensorflow import keras
from models import *
from metalearn import FReE_Learning
from dataGenerator import DataGenerator, FewshotDataGen, DataGenerator_Coarse, FewshotDataGen_Coarse

def model_training(inputshape,nclasses,data_dir,model_dir,tr_csv,val_csv,learningrate,batchsize,patience,epochs, NoOfSubcarrier):
    # define model
    model = Embed_Dec(inputshape,nclasses)
    opt = keras.optimizers.Adam(learningrate)
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(opt,loss)
    # define dataset
    train_gen = DataGenerator(data_dir,tr_csv, NoOfSubcarrier,nclasses,inputshape,batchsize)
    #print(np.shape(train_gen))
    val_gen = DataGenerator(data_dir, val_csv, NoOfSubcarrier, nclasses,inputshape,batchsize)
    # training
    history = model.fit(
        x=train_gen,
        epochs=epochs,
        validation_data = val_gen,
        callbacks = customCallback(model_dir,patience))
    
    print("training complete, model saved at: ", model_dir)
    

def model_testing(model_dir,train_dir,test_dir,meta_tr_csv,meta_test_csv,inputshape,learningrate,nways,kshots,nqueries,epochs,episodes,iterations, NoOfSubcarrier):
    # define model
    model = FReE_Learning(model_dir,inputshape,learningrate)
    # define dataset
    meta_train_set = FewshotDataGen(train_dir,meta_tr_csv,inputshape,nways, NoOfSubcarrier)
    meta_test_set = FewshotDataGen(test_dir,meta_test_csv,inputshape,nways, NoOfSubcarrier)
    # finetune
    classifiers = ["crossentropy", "knn"]
    for clf in classifiers:
        print("finetuning decoder with: ", clf)
        model.retrain(meta_train_set,nways,kshots,nqueries,epochs,episodes,classifier=clf)
        print("finetuning complete, start testing")
        model.test(meta_train_set,meta_test_set,nways,kshots,nqueries,iterations)



def model_training_coarse(inputshape,nclasses,data_dir,model_dir,tr_csv,val_csv,learningrate,batchsize,patience,epochs, NoOfSubcarrier):
    # define model
    model = Embed_Dec(inputshape,nclasses)
    opt = keras.optimizers.Adam(learningrate)
    loss = keras.losses.CategoricalCrossentropy()
    model.compile(opt,loss)
    # define dataset
    train_gen = DataGenerator_Coarse(data_dir,tr_csv, NoOfSubcarrier,nclasses,inputshape,batchsize)
    #print(np.shape(train_gen))
    val_gen = DataGenerator_Coarse(data_dir, val_csv, NoOfSubcarrier, nclasses,inputshape,batchsize)
    # training
    history = model.fit(
        x=train_gen,
        epochs=epochs,
        validation_data = val_gen,
        callbacks = customCallback(model_dir,patience))
    
    print("training complete, model saved at: ", model_dir)




def model_testing_coarse(model_dir,train_dir,test_dir,meta_tr_csv,meta_test_csv,inputshape,learningrate,nways,kshots,nqueries,epochs,episodes,iterations, NoOfSubcarrier):
    # define model
    model = FReE_Learning(model_dir,inputshape,learningrate)
    # define dataset
    meta_train_set = FewshotDataGen_Coarse(train_dir,meta_tr_csv,inputshape,nways, NoOfSubcarrier)
    meta_test_set = FewshotDataGen_Coarse(test_dir,meta_test_csv,inputshape,nways, NoOfSubcarrier)
    # finetune
    classifiers = ["crossentropy", "knn"]
    for clf in classifiers:
        print("finetuning decoder with: ", clf)
        model.retrain(meta_train_set,nways,kshots,nqueries,epochs,episodes,classifier=clf)
        print("finetuning complete, start testing")
        model.test(meta_train_set,meta_test_set,nways,kshots,nqueries,iterations)