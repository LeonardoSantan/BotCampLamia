import tensorflow as tf
import numpy as np
from tensorflow import keras


#define modelo entre si
#unica camada com um unico neuronio
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
#perda e otimizador, fazendo suposição entre os números
#através da suposição irá identificar se suposição é boa ou ruim
#gerando uma nova suposição através da função de otimização 
model.compile(optimizer='sgd', loss='mean_squared_error')
#matriz de X e Y
xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0])
ys = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0])
#tente por 500 vezes esse processo
model.fit(xs,ys, epochs=500)

print(model.predict([10.0]))