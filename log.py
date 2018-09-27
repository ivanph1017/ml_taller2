# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 08:39:24 2018
@author: a2203
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, model_selection, linear_model
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve
from sklearn import metrics
import scipy.stats as st
import os


ROOT_PATH = os.path.dirname(os.getcwd())
PROJECT_PATH = os.path.join(ROOT_PATH,'ml_taller2')

def get_data():
    data=np.genfromtxt("yeast.csv",delimiter=",", dtype=np.unicode_)
    
    X=data[1:,1:-1].astype(float)  
    
    y=data[1:,-1]
    le = preprocessing.LabelEncoder()
    #le.fit(y)
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3)
    return X_train, X_test, y_train, y_test

def cross_val_logisticR(X_train, X_test, y_train, y_test):
    lg_cv=linear_model.LogisticRegressionCV(cv=10, n_jobs = 10)
    lg_cv.fit(X_train,y_train)
    predictions = lg_cv.predict(X_test)
    print('Predicciones {}'.format(predictions))
    cm=confusion_matrix(y_test, predictions)
    print('Matriz de confusion')
    print(cm)
    print('Accuracy de Sklearn {}'.format(accuracy_score(y_test, predictions)))
    stds = np.std(X_train, axis=0)
    return lg_cv.intercept_[0], lg_cv.coef_[0], stds, predictions
    
    
def z_test(coef, std):
    #z score
    z = coef / std
    # p-value
    p = st.norm.cdf(z)
    print('Z score: {}'.format(z))
    print('P value: {:0.8f}'.format(p))
    return z, p

def g_test(y_test, predictions):
    actual = y_test + 1
    estimates = predictions + 1
    y_ith = estimates * np.log(estimates / actual)
    #g score
    g = 2 * np.sum(y_ith)
    # n - 1 degrees of freedom
    df = y_test.shape[0] - 1
    # p-value
    p = st.chi2.cdf(g, df)
    print('G score: {}'.format(g))
    print('P value: {:0.8f}'.format(p))
    return g, p

def roc(y_test, predictions, pos_label, label):
    filename = 'log_{}_roc_curve_.png'.format(label) 
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
    X_names = ['mcg', 'gvh','alm', 'mit', 
             'erl', 'pox', 'vac', 
             'nuc']
    print('Primera corrida\n\n')
    X_train, X_test, y_train, y_test = get_data()
    intercept, coefs, stds, predictions = cross_val_logisticR(X_train, X_test, y_train, y_test)
    print('Intercepto {}'.format(intercept))
    print('Coeficientes {}'.format(coefs))
    cols_discarded = []
    for j in range(X_train.shape[1]):
        print('\nFeature {}'.format(X_names[j]))
        z, p = z_test(coefs[j], stds[j])
        # Si el intervalo de confianza es estricto al 95%, se queda sin features
        # if p >= 0.025 and p <= 0.975
        if p >= 0.025 and p <= 0.975:
            print('Feature to discard'.format(X_names[j]))
            cols_discarded.append(j)
    print('\nModelo 1')
    g, p_g = g_test(y_test, predictions)
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
    
    # X = np.concatenate((data[:,:-2], data[:,-1].reshape((-1, 1))), axis=1)
# =============================================================================
#     cols = list(filter(lambda i: not i in cols_discarded, range(len(X_names))))
#     X_names_remaining = list(map(lambda i: X_names[i], cols))
#     X_train = np.delete(X_train, cols_discarded, 1)
#     X_test =  np.delete(X_test, cols_discarded, 1)
# 
#     print('\n\n2da corrida')
#     intercept, coefs, stds, predictions = cross_val_logisticR(X_train, X_test, y_train, y_test)
#     print('Intercepto {}'.format(intercept))
#     print('Coeficientes {}'.format(coefs))
#     for j in range(X_train.shape[1]):
#         print('\nFeature {}'.format(X_names_remaining[j]))
#         z, p = z_test(coefs[j], stds[j])
#     print('\nModelo 2')
#     g, p_g = g_test(y_test, predictions)
#     print('\n\n')
# =============================================================================
    
if __name__=="__main__":
    main()