# Normalización
Las "x" se normalizó de la siguiente forma (MinMaxScaler)
X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
X_scaled = X_std * (max - min) + min

El label "y" se normalizó (LabelEncoder) con valores numéricos: 1, 2, ..., 10

# Regresión logística - CrossValidation

Se utilizó cross validation con K = 10 en Python 3 con el módulo log.py

Predicciones [7 7 7 0 5 0 0 0 0 0 7 6 0 5 0 0 7 6 3 0 2 6 7 5 0 2 0 7 7 0 6 6 0 4 7 0 0
 7 0 5 0 7 0 2 6 7 5 0 7 6 7 0 6 0 6 7 2 5 0 0 0 0 0 7 0 7 0 5 0 7 7 0 0 7
 2 0 5 7 0 7 5 0 3 7 0 0 6 0 7 0 0 0 0 7 7 0 0 2 7 6 8 0 0 0 4 0 7 0 6 6 7
 3 6 5 0 5 6 7 0 0 0 0 6 6 0 7 0 5 6 7 6 6 5 0 0 7 5 0 2 6 7 0 0 6 0 0 0 6
 7 3 7 0 6 7 0 7 5 0 6 0 6 7 7 0 0 5 5 6 6 0 3 0 0 7 0 0 5 0 5 2 6 5 5 0 6
 7 5 7 7 6 6 0 5 7 0 7 0 0 0 7 6 0 7 7 6 3 0 7 6 6 5 7 7 6 6 5 6 5 6 4 0 0
 7 0 0 6 5 6 7 3 7 7 0 5 7 0 7 5 6 0 7 0 0 6 2 0 5 7 7 0 7 6 0 0 0 0 5 0 0
 0 4 0 7 0 6 6 7 0 0 0 0 7 7 6 0 7 0 5 0 5 0 7 0 7 5 6 7 0 0 0 0 0 6 0 6 0
 0 6 0 5 7 0 0 0 0 0 0 6 0 8 0 0 7 7 5 6 6 6 5 6 0 6 3 0 5 6 0 0 0 6 7 7 0
 5 7 3 0 6 7 7 0 0 0 0 7 7 7 7 0 5 0 5 7 7 7 6 6 7 0 6 6 0 5 7 7 0 0 7 7 0
 7 5 0 5 0 0 7 7 6 4 5 7 0 6 6 4 3 0 5 0 7 6 6 6 0 6 0 5 0 0 6 7 0 0 5 0 0
 0 0 0 6 0 0 0 6 5 7 7 0 0 0 7 7 0 0 5 5 0 7 8 0 0 7 7 0 0 6 5 7 0 0 7 5 5
 6 0]

Matriz de confusion
[[96  0  0  0  0  0  9 29  0  0]
 [ 0  2  0  0  0  0  0  0  0  0]
 [ 3  0  4  1  0  0  0  0  0  0]
 [ 1  0  0  9  0  1  1  0  0  0]
 [ 3  0  0  1  2  2  3  1  0  0]
 [10  0  0  0  0 42  0  1  0  0]
 [25  0  0  0  1  4 39  3  0  0]
 [53  0  0  0  0  5  5 77  0  0]
 [ 3  0  0  0  0  0  0  0  4  0]
 [ 3  0  0  0  0  2  0  1  0  0]]

Accuracy de Sklearn 0.6165919282511211
Intercepto -0.4013080060401378
Coeficientes [-0.45069 -1.21265  2.52929 
              -1.8671   0.02267 -0.58001  
              0.03719 -1.30758]

Feature mcg
Z score: -1.1339272705503745
P value: 0.12841253
Feature to discard

Feature gvh
Z score: -11.241518129284911
P value: 0.00000000

Feature alm
Z score: 28.798491678125345
P value: 1.00000000

Feature mit
Z score: -11.364076276735775
P value: 0.00000000

Feature erl
Z score: -2.1071137656415404
P value: 0.01755386

Feature pox
Z score: -7.635334101312592
P value: 0.00000000

Feature vac
Z score: 1.1525722041979105
P value: 0.87545699
Feature to discard

Feature nuc
Z score: -12.438778531866575
P value: 0.00000000

Modelo 1
G score: 1024.4198648846525
P value: 1.00000000

## Curvas ROC
### CYT

![Curva ROC CYT](URL_IMAGE)

### NUC

![Curva ROC NUC](URL_IMAGE)

### MIT

![Curva ROC MIT](URL_IMAGE)

### ME3

![Curva ROC ME3](URL_IMAGE)

### ME2

![Curva ROC ME2](URL_IMAGE)

### ME1

![Curva ROC ME1](URL_IMAGE)

### EXC

![Curva ROC EXC](URL_IMAGE)

### VAC

![Curva ROC VAC](URL_IMAGE)

### POX

![Curva ROC POX](URL_IMAGE)

### ERL

![Curva ROC ERL](URL_IMAGE)

# Red neuronal

Se utilizó Python 3 con el módulo nn.py

Capas de entrada: 8 (# de features)
Capas ocultas: 14
Capas de salida: 10 (# de clases del label)

Función de activación: logística
Maximas iteraciones: 10 000
Penalidad L2 (alpha): 0.01

No se indica momentum porque solo se utiliza con solver='sgd'
MLP Scores -0.252989802302646

## Cross-validation K = 10
Scores de Error [-0.70337 -0.63077 -0.74278 
        -0.18618 -1.13936 -0.75973 
        -1.1892  -1.58681 -0.67682 
        -0.26947]
Accuracy: -0.79 (+/- 0.80)

## Curvas ROC
### CYT

![Curva ROC CYT](URL_IMAGE)

### NUC

![Curva ROC NUC](URL_IMAGE)

### MIT

![Curva ROC MIT](URL_IMAGE)

### ME3

![Curva ROC ME3](URL_IMAGE)

### ME2

![Curva ROC ME2](URL_IMAGE)

### ME1

![Curva ROC ME1](URL_IMAGE)

### EXC

![Curva ROC EXC](URL_IMAGE)

### VAC

![Curva ROC VAC](URL_IMAGE)

### POX

![Curva ROC POX](URL_IMAGE)

### ERL

![Curva ROC ERL](URL_IMAGE)

## KBest Features (K = 4)
Features F-scores: [10.16273  0.62562 15.67817 
                    17.8361   4.05819 15.60139 
                    0.07494  48.50648]
Features p-values: [1.46301e-03 4.29092e-01 
                    7.86463e-05 2.55432e-05 
                    4.41380e-02 8.18702e-05
                    7.84307e-01 4.93280e-12]

Los mejores features con p-value < 0.05 para un intervalor de confianza de 95% son:
1.46301e-03: mcg
7.86463e-05: alm
2.55432e-05: mit
4.41380e-02: erl
8.18702e-05: pox
4.93280e-12: nuc

El top 4 son:
4.93280e-12: nuc
2.55432e-05: mit
7.86463e-05: alm
8.18702e-05: pox