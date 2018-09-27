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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import scipy.stats as st


def get_data():
    data = np.genfromtxt("yeast.csv",delimiter=",")
    X = data[:,1:]
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    # Formula para estandarizar
    X = (X - means) / stds
    print(X)
    y = data[:, -1].astype(int)
    print(y)
#    y = y - 1
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
        if p >= 0.2 and p <= 0.8:
            print('Feature to discard'.format(X_names[j]))
            cols_discarded.append(j)
    print('\nModelo 1')
    g, p_g = g_test(y_test, predictions)
    
    # X = np.concatenate((data[:,:-2], data[:,-1].reshape((-1, 1))), axis=1)
    cols = list(filter(lambda i: not i in cols_discarded, range(len(X_names))))
    X_names_remaining = list(map(lambda i: X_names[i], cols))
    X_train = np.delete(X_train, cols_discarded, 1)
    X_test =  np.delete(X_test, cols_discarded, 1)

    print('\n\n2da corrida')
    intercept, coefs, stds, predictions = cross_val_logisticR(X_train, X_test, y_train, y_test)
    print('Intercepto {}'.format(intercept))
    print('Coeficientes {}'.format(coefs))
    for j in range(X_train.shape[1]):
        print('\nFeature {}'.format(X_names_remaining[j]))
        z, p = z_test(coefs[j], stds[j])
    print('\nModelo 2')
    g, p_g = g_test(y_test, predictions)
    print('\n\n')
    
if __name__=="__main__":
    main()