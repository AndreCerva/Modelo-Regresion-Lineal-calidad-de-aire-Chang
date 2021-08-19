# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:20:44 2021

@author: andre
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#Recordar modelo y=bo+b1x1+b2x2...bpxp+E //y=variable dependiente, E= error, bp=valor de coeficientes
#1) paso
#Cargar nuestra base de datos a examinar
dfquality=pd.read_csv('CalidadShangai.csv')#(dataset)
dfquality=dfquality.dropna() #Elimina los valores que contengan valores nulos, útil solo cuando son muchos valores
#2) Ver la relación entre las variables de nuestra base de datos
"""Mediante el metodo corr(), graficarlo mediante mapa de calor y tomar 
una decisión según se requiera y se tenga una relación"""
MatrizCorrelacion=dfquality.corr()
sns.heatmap(MatrizCorrelacion,annot=True)
plt.rcParams['figure.figsize']=(10,10)
plt.show()
#3) separar nuestros datos de interes en otro df
#Predecir CO segun PM2.5,PM10,N02
dfinteres=dfquality[['PM2.5','PM10','NO2']]
dfCO=dfquality['CO']
#4) Dividir datos en variables de entrenamiento y test con train_test_split
#Variables independientes y luego la dependiente, tamaño de los datos de test
Xtrain,Xtest,ytrain,ytest=train_test_split(dfinteres,dfCO,test_size=0.2)
#5) Crear modelo con nuestros datos
modelolineal=LinearRegression()
modelolineal.fit(Xtrain, ytrain)
print(f'El coeficiente bp es {modelolineal.coef_}')
print(f'El valor bo {modelolineal.intercept_}')
#5) metricas
#r2 es porcentaje de variacion en la respuesta
#r2: coeficiente de determinacion, determina la calidad del modelo para replicar los resultados
#y la proporcion de la variacion de los resultados que puede explicarse por el modelo
ypredicciontrain=modelolineal.predict(Xtrain)
ypredicciontest=modelolineal.predict(Xtest)
print(f'El valor de r2 es: {r2_score(ytrain,ypredicciontrain)}')
r2test=r2_score(ytest,ypredicciontest)
print(f'El valor de r2 del test es: {r2test}')
#r2 ajustado=1-VNE*(n-(p+1))VT/(n-1)
#VNE=1-r2, n= numero de datos utilizados,, p=numero de variables independientes
r2ajustado=1-(1-r2test)*(len(ytest)-1)/(len(ytest)-Xtest.shape[1]-1)
#6) Visualizar resultados
resultados=modelolineal.predict(dfinteres)
dfresultados=pd.DataFrame({'Real':dfCO,'Predicho':resultados})
maxpredicho=dfresultados['Predicho'].max()
print(f'El valor máximo predicho es: {maxpredicho}')
maxreal=dfresultados['Real'].max()
print(f'El valor máximo real es: {maxreal}')
minpredicho=dfresultados['Predicho'].min()
print(f'El valor min predicho es: {minpredicho}')
minreal=dfresultados['Real'].min()
print(f'El valor min real es: {minpredicho}')
dfresultados.head(20).plot(kind='bar')
plt.title('Comparación entre real y predicho del modelo')
plt.xlabel('Muestra')
plt.ylabel('Cantidad')
plt.show()



