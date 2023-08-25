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


import os
import argparse
import tensorflow as tf
from tensorflow import keras
import numpy as np

from TrainTest import model_testing, model_training, model_testing_coarse, model_training_coarse

###### hyper parameters ######

windowsize = 50
data_proc = "processed_dataset"
# train phase
learningrate=0.01
batchsize=128
patience=20
train_epochs = 10
# meta finetune phase
meta_lr = 0.01
meta_train_epoch = 15
meta_train_episode = 100
meta_test_iterations = 1000
kshots = 10
nqueries = 10


def arguments_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('test', help='Which Test: coarse or fine_grained')
    parser.add_argument('Train_Env', help='Testing Scenario')
    parser.add_argument('train_sta', help='name of the station')
    parser.add_argument('Test_Env', help='Testing Scenario')
    parser.add_argument('test_sta', help='name of the station')
    parser.add_argument('model_save', help='Name of the model')
    parser.add_argument('NoOfSubcarrier', help='No Of Subcarrier')
    parser.add_argument('--Train', '-tr', action='store_true',help='pre-train embedding model')
    parser.add_argument('--Finetune', '-ft', action='store_true',help='finetune decoder')
    return parser.parse_args()

def main():
    args = arguments_parser()
    test = args.test
    Train_Env=args.Train_Env
    train_sta=args.train_sta
    Test_Env=args.Test_Env
    test_sta=args.test_sta
    model_save= args.model_save
    NoOfSubcarrier= int(args.NoOfSubcarrier)

    Bw = "80MHz"
    num_mon = "3mo"
    slots="Slots"

    inputshape = (windowsize, NoOfSubcarrier, 2)
    data_path = "../Data/" + test + "/"

    if test == "coarse":
        num_classes = 4
    elif test == "fine_grained":
        num_classes = 20

    
    model_dir = os.path.join(data_path, Train_Env, Bw, num_mon, train_sta, model_save )

    train_dir= os.path.join(data_path, Train_Env, Bw, num_mon, train_sta, slots, 'Train' )
    test_dir= os.path.join(data_path, Test_Env, Bw, num_mon, test_sta, slots, 'Test' )
    meta_tr_dir = os.path.join(data_path, Test_Env, Bw, num_mon, test_sta, slots, 'Train' )
    
    tr_csv = os.path.join(train_dir, 'train_set.csv')
    val_csv = os.path.join(train_dir, 'val_set.csv')
    meta_tr_csv = os.path.join(meta_tr_dir,'val_set.csv')
    test_csv = os.path.join(test_dir, 'test_set.csv')

    if args.Train:
        if test == "fine_grained":
            model_training(inputshape,num_classes,train_dir,model_dir,tr_csv,val_csv,learningrate,batchsize,patience,train_epochs, NoOfSubcarrier)
        elif test == "coarse":
            model_training_coarse(inputshape,num_classes,train_dir,model_dir,tr_csv,val_csv,learningrate,batchsize,patience,train_epochs, NoOfSubcarrier)

    if args.Finetune:
        if test == "fine_grained":
            model_testing(model_dir,meta_tr_dir,test_dir,meta_tr_csv,test_csv,inputshape,meta_lr,num_classes,kshots,nqueries,meta_train_epoch,meta_train_episode,meta_test_iterations, NoOfSubcarrier)
        elif test == "coarse":
            model_testing_coarse(model_dir,meta_tr_dir,test_dir,meta_tr_csv,test_csv,inputshape,meta_lr,num_classes,kshots,nqueries,meta_train_epoch,meta_train_episode,meta_test_iterations, NoOfSubcarrier)

if __name__ == '__main__':
    main()



