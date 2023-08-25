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
from tensorflow import keras
import pandas as pd
from multiprocessing import Pool, cpu_count
import scipy.io as spio
import os
from multiprocessing import Pool, cpu_count
from itertools import repeat
windowsize=window_size=50

def read_mat(dir,file, NoOfSubcarrier):    
    
    data = spio.loadmat(os.path.join(dir, file))
    if file[0] == 'A':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 0
    elif file[0] == 'B':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 1
    elif file[0] == 'C':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 2
    elif file[0] == 'D':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 3
    elif file[0] == 'E':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 4
    elif file[0] == 'F':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 5
    elif file[0] == 'G':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 6
    elif file[0] == 'H':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 7
    elif file[0] == 'I':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 8
    elif file[0] == 'J':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 9
    elif file[0] == 'K':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 10
    elif file[0] == 'L':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 11
    elif file[0] == 'M':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 12
    elif file[0] == 'N':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 13
    elif file[0] == 'O':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 14
    elif file[0] == 'P':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 15
    elif file[0] == 'Q':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 16
    elif file[0] == 'R':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 17
    elif file[0] == 'S':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 18
    elif file[0] == 'T':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 19

    csi_real = np.real(csi_data_cplx)
    #print(np.shape(csi_real))
    csi_imag = np.imag(csi_data_cplx)
    csi_real = csi_real.reshape(windowsize,-1,1)
    csi_imag = csi_imag.reshape(windowsize,-1,1)
    csi_data = np.concatenate([csi_real,csi_imag],axis=2)

    
    return csi_data, label

def read_mat_coarse(dir,file, NoOfSubcarrier):    
    
    data = spio.loadmat(os.path.join(dir, file))
    if file[0] == 'A':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 0
    elif file[0] == 'B':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 1
    elif file[0] == 'C':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 2
    elif file[0] == 'D':
        csi_data_cplx = data['csi_mon'][:,0:NoOfSubcarrier]
        label = 3

    csi_real = np.real(csi_data_cplx)
    #print(np.shape(csi_real))
    csi_imag = np.imag(csi_data_cplx)
    csi_real = csi_real.reshape(windowsize,-1,1)
    csi_imag = csi_imag.reshape(windowsize,-1,1)
    csi_data = np.concatenate([csi_real,csi_imag],axis=2)

    
    return csi_data, label



class DataGenerator(keras.utils.Sequence):
    
    def __init__(self, dataset_path, dataset_csv, NoOfSubcarrier, num_classes, chunk_shape, batchsize, shuffle=True, to_categorical=True):

        df = pd.read_csv(dataset_csv)
        self.dataset_path = dataset_path
        self.batchsize = batchsize
        self.datalist = df["filename"]
        self.labels = df["label"]
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.NoOfSubcarrier = NoOfSubcarrier
        self.windowsize = chunk_shape[0]
        self.length = chunk_shape[1]
        self.height = chunk_shape[2]
        self.to_categorical = to_categorical
        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()
        return
    def __len__(self):
        """Denote the number of batches"""
        return int(np.floor(len(self.labels) / self.batchsize))
    def __getitem__(self, idx):
        """Generate one batch of data"""
        indexes = self.indexes[idx*self.batchsize:(idx+1)*self.batchsize]
        X, y = self.__load_batch(indexes)
        return X, y
    def on_epoch_end(self):
        """Update indexes after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __load_batch(self, indexes):
        """Read new batch of data """
        batch_data = np.empty((self.batchsize,self.windowsize,self.length,self.height))
        batch_label = np.empty(self.batchsize,dtype=int)
        for i, k in enumerate(indexes):
            batch_data[i], batch_label[i] = read_mat(self.dataset_path,self.datalist[k], self.NoOfSubcarrier)
        if self.to_categorical:
            batch_label = keras.utils.to_categorical(batch_label, num_classes=self.num_classes)
        return batch_data, batch_label

class FewshotDataGen:

    def __init__(self,dataset_path,dataset_csv,chunk_shape, num_classes, NoOfSubcarrier):
        
        self.df = pd.read_csv(dataset_csv)
        self.dataset_path = dataset_path
        self.labellist = [subfolder[0] for subfolder in os.listdir(dataset_path)
                                        if os.path.isdir(os.path.join(dataset_path,subfolder))]
        self.num_classes = num_classes

        self.NoOfSubcarrier = NoOfSubcarrier
        self.windowsize = chunk_shape[0]
        self.length = chunk_shape[1]
        self.height = chunk_shape[2]

    def load_batch(self,nways,kshots,nqueries=None):
        """Read new batch of data """
        self.ways = nways
        self.shots = kshots
        self.queries = nqueries

        support_data = np.zeros(shape=(self.ways*self.shots,self.windowsize,self.length,self.height))
        support_label = np.zeros(self.ways*self.shots,dtype=int)
        support_mini_batch=[]
        if self.queries is not None:
            query_data = np.zeros(shape=(self.ways*self.queries,self.windowsize,self.length,self.height))
            query_label = np.zeros(self.ways*self.queries,dtype=int)
            query_mini_batch=[]
        
        for label in self.labellist:
            support_mini_batch.append(self.df.loc[self.df['label']==label].sample(n=self.shots,replace=True))
            if self.queries is not None:
                query_mini_batch.append(self.df.loc[self.df['label']==label].sample(n=self.queries,replace=True))
        support_mini_batch = pd.concat(support_mini_batch)
        for j, l in enumerate(support_mini_batch['filename']):
            support_data[j], support_label[j] = read_mat(self.dataset_path,l, self.NoOfSubcarrier)
 
        if self.queries is not None:
            query_mini_batch = pd.concat(query_mini_batch)
            for i, k in enumerate(query_mini_batch['filename']):
                query_data[i], query_label[i] = read_mat(self.dataset_path,k, self.NoOfSubcarrier)
            return support_data, support_label, query_data, query_label

        return support_data,support_label


class DataGenerator_Coarse(keras.utils.Sequence):
    
    def __init__(self, dataset_path, dataset_csv, NoOfSubcarrier, num_classes, chunk_shape, batchsize, shuffle=True, to_categorical=True):

        df = pd.read_csv(dataset_csv)
        self.dataset_path = dataset_path
        self.batchsize = batchsize
        self.datalist = df["filename"]
        self.labels = df["label"]
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.NoOfSubcarrier = NoOfSubcarrier
        self.windowsize = chunk_shape[0]
        self.length = chunk_shape[1]
        self.height = chunk_shape[2]
        self.to_categorical = to_categorical
        self.indexes = np.arange(len(self.labels))
        self.on_epoch_end()
        return
    def __len__(self):
        """Denote the number of batches"""
        return int(np.floor(len(self.labels) / self.batchsize))
    def __getitem__(self, idx):
        """Generate one batch of data"""
        indexes = self.indexes[idx*self.batchsize:(idx+1)*self.batchsize]
        X, y = self.__load_batch(indexes)
        return X, y
    def on_epoch_end(self):
        """Update indexes after each epoch"""
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    def __load_batch(self, indexes):
        """Read new batch of data """
        batch_data = np.empty((self.batchsize,self.windowsize,self.length,self.height))
        batch_label = np.empty(self.batchsize,dtype=int)
        for i, k in enumerate(indexes):
            batch_data[i], batch_label[i] = read_mat_coarse(self.dataset_path,self.datalist[k], self.NoOfSubcarrier)
        if self.to_categorical:
            batch_label = keras.utils.to_categorical(batch_label, num_classes=self.num_classes)
        return batch_data, batch_label




class FewshotDataGen_Coarse:

    def __init__(self,dataset_path,dataset_csv,chunk_shape, num_classes, NoOfSubcarrier):
        
        self.df = pd.read_csv(dataset_csv)
        self.dataset_path = dataset_path
        self.labellist = [subfolder[0] for subfolder in os.listdir(dataset_path)
                                        if os.path.isdir(os.path.join(dataset_path,subfolder))]
        self.num_classes = num_classes

        self.NoOfSubcarrier = NoOfSubcarrier
        self.windowsize = chunk_shape[0]
        self.length = chunk_shape[1]
        self.height = chunk_shape[2]

    def load_batch(self,nways,kshots,nqueries=None):
        """Read new batch of data """
        self.ways = nways
        self.shots = kshots
        self.queries = nqueries

        support_data = np.zeros(shape=(self.ways*self.shots,self.windowsize,self.length,self.height))
        support_label = np.zeros(self.ways*self.shots,dtype=int)
        support_mini_batch=[]
        if self.queries is not None:
            query_data = np.zeros(shape=(self.ways*self.queries,self.windowsize,self.length,self.height))
            query_label = np.zeros(self.ways*self.queries,dtype=int)
            query_mini_batch=[]
        
        for label in self.labellist:
            support_mini_batch.append(self.df.loc[self.df['label']==label].sample(n=self.shots,replace=True))
            if self.queries is not None:
                query_mini_batch.append(self.df.loc[self.df['label']==label].sample(n=self.queries,replace=True))
        support_mini_batch = pd.concat(support_mini_batch)
        for j, l in enumerate(support_mini_batch['filename']):
            support_data[j], support_label[j] = read_mat_coarse(self.dataset_path,l, self.NoOfSubcarrier)
 
        if self.queries is not None:
            query_mini_batch = pd.concat(query_mini_batch)
            for i, k in enumerate(query_mini_batch['filename']):
                query_data[i], query_label[i] = read_mat_coarse(self.dataset_path,k, self.NoOfSubcarrier)
            return support_data, support_label, query_data, query_label

        return support_data,support_label

