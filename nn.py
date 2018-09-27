#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 12:17:19 2018

@author: ivan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report,confusion_matrix,roc_curve
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
import os


ROOT_PATH = os.path.dirname(os.getcwd())
PROJECT_PATH = os.path.join(ROOT_PATH,'ml_taller2')

def load():
    data=np.genfromtxt("yeast.csv",delimiter=",", dtype=np.unicode_)

    X=data[1:,1:-1].astype(float)  
    
    y=data[1:,-1]
    le = preprocessing.LabelEncoder()
    #le.fit(y)
    y = le.fit_transform(y)
    #without scaling
    #nn(X,y)
    nnS(X,y)
    selectKBest(X,y)

def nnS(X,y):
    minmax=preprocessing.MinMaxScaler(feature_range=(0, 1))
    X=minmax.fit_transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y)
    mlp=MLPRegressor(hidden_layer_sizes=(8,14,10),solver='lbfgs',
                     activation='logistic',max_iter=10000,
                     alpha=0.01,momentum=0.5)
    #print mlp
    mlp.fit(X_train,y_train)
    predictions = mlp.predict(X_test)
    # print("Confusion matrix \n{}".format(confusion_matrix(y_test,predictions)))
    # ValueError: Classification metrics can't handle a mix of multiclass and 
    # continuous targets
    print("MLP Scores {}".format(mlp.score(X_test,y_test)))
    #cross validation
    scores=cross_val_score(mlp,X,y,cv=10)
    print("Scores {}".format(scores))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    fpr, tpr, thresholds = roc_curve(y_test, predictions, pos_label=2)
    #ROC Curve
    roc(y_test, predictions, 0, 'CYT')
    roc(y_test, predictions, 1, 'NUC')
    roc(y_test, predictions, 2, 'MIT')
    roc(y_test, predictions, 3, 'ME3')
    roc(y_test, predictions, 4, 'ME2')
    roc(y_test, predictions, 5, 'ME1')
    roc(y_test, predictions, 6, 'EXC')
    roc(y_test, predictions, 7, 'VAC')
    roc(y_test, predictions, 8, 'POX')
    roc(y_test, predictions, 9, 'ERL')
    
def selectKBest(X,y):
    selected=SelectKBest(score_func=f_regression,k=4)
    fit=selected.fit(X,y)
    np.set_printoptions(precision=5)
    print("Features scores: {}".format(fit.scores_))
    print("Features p-values: {}".format(fit.pvalues_))
    
def roc(y_test, predictions, pos_label, label):
    filename = 'nn_{}_roc_curve_.png'.format(label) 
    filepath = os.path.join(PROJECT_PATH, filename)
    fpr, tpr, thresholds = roc_curve(y_test, predictions, pos_label=pos_label)
    #ROC Curve
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic: {}'.format(label))
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(filename)
    plt.show()

def main():
    load()

if __name__=="__main__":
    main()