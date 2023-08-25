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


from tensorflow import keras
from utils import *
from models import Decoder
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

class FReE_Learning():
    def __init__(self, embedding, inputshape, learning_rate=0.01, optimizer='Adam'):

        self.embedding = keras.models.load_model(embedding)
        self.height, self.width, self.depth = inputshape
        self.lr_rate = learning_rate
        
        if optimizer == 'Adam':
            self.optimizer = keras.optimizers.Adam(learning_rate=self.lr_rate)
        elif optimizer == 'SGD':
            self.optimizer = keras.optimizers.SGD(learning_rate=self.lr_rate)
        else:
            raise NotImplementedError("the optimizer is not supported, please use SGD or Adam")

    def retrain(self,miniset,nways,kshots,nqueries,n_epochs,n_episodes,classifier='crossentropy',n_neighbors=5):
        
        self.classifier = classifier
        if self.classifier == 'crossentropy':
            self.decoder = Decoder(self.embedding.layers[-1].output.shape[1:],nways)
            for ep in range(n_epochs):
                for epi in range(n_episodes):
                    support,s_label,query,q_label = miniset.load_batch(nways,kshots,nqueries=nqueries)
                    loss, acc = meta_train_step_softmax(self.embedding,self.decoder,self.optimizer,support,s_label,query,q_label,nways)
                    if (epi+1) % 50 == 0:
                        print('[epoch {}/{}, episode {}/{}] => meta-training loss: {:.5f}, meta-training acc: {:.5f}'.format(ep+1,
                                                                    n_epochs,
                                                                    epi+1,
                                                                    n_episodes,
                                                                    loss,
                                                                    acc))
        elif self.classifier == 'proto':
            self.decoder = Decoder(self.embedding.layers[-1].output.shape[1:],nways,softmax=False)
            for ep in range(n_epochs):
                for epi in range(n_episodes):
                    support, _, query, labels = miniset.load_batch(nways,kshots,nqueries=nqueries)
                    loss, acc = meta_train_step_proto(self.embedding,self.decoder,self.optimizer,support,query,labels,nways,kshots,nqueries)
                    if (epi+1) % 50 == 0:
                        print('[epoch {}/{}, episode {}/{}] => meta-training loss: {:.5f}, meta-training acc: {:.5f}'.format(ep+1,
                                                                    n_epochs,
                                                                    epi+1,
                                                                    n_episodes,
                                                                    loss,
                                                                    acc))
        elif self.classifier == 'knn':
            self.decoder = KNeighborsClassifier(n_neighbors=n_neighbors)
            print("nearest neighbor classifier is chosen.")
        else:
            raise NotImplementedError("the classifier is not supported, please use crossentropy, proto or knn")
   
    def test(self,miniset,testset,nways,kshots,nqueries,iteration):

        acc_list = []
        y_true = []
        y_pred = []
        for iter in range(iteration):
            support,s_labels = miniset.load_batch(nways,kshots)
            query,q_labels = testset.load_batch(nways,nqueries)
            if self.classifier == 'proto':
                _, acc = meta_test_step_proto(self.embedding,self.decoder,support,query,q_labels,nways,kshots,nqueries)

            elif self.classifier == 'crossentropy':
                acc, q_predict = meta_test_step_softmax(self.embedding,self.decoder,query,q_labels,nways)
                q_predict = np.argmax(q_predict,axis=1)

            elif self.classifier == 'knn':
                support_latent = self.embedding(support,training=False)
                query_latent = self.embedding(query,training=False)
                self.decoder.fit(support_latent,s_labels)
                acc = self.decoder.score(query_latent,q_labels)
                q_predict = self.decoder.predict(query_latent)

            y_true.append(q_labels)
            y_pred.append(q_predict)

            acc_list.append(acc)
            if (iter+1)%50 == 0:
                print("testing ==> [batch{}/{}]".format(iter+1,iteration))
        acc_avg = sum(acc_list)/len(acc_list)
        
        y_t = np.concatenate(y_true)
        y_p = np.concatenate(y_pred)
        
        labels = range(nways)
        cm = np.around(confusion_matrix(y_t,y_p,normalize='true'),2)
        
        plt.figure(figsize=(12, 9))
        plt.title('confusion matrix of {}'.format(self.classifier), fontsize=12,fontstyle='italic',weight=800)
        ax = sns.heatmap(cm, cmap=plt.cm.Blues, fmt='g', annot=True, square=True, xticklabels=labels, yticklabels=labels)
        ax.set_ylabel('Actual Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        plt.savefig('confusion matrix of {}'.format(self.classifier),dpi=300)

        print(self.classifier," test accuracy: {:.4f}".format(acc_avg))