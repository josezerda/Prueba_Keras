import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam 
from matplotlib import pyplot as plt



#print(tf.__version__)

def generate_gaussians_distributions(sep=1, N=500, random_state=42, normalize=True):
    np.random.seed(random_state)
    # Zeros
    X1 = np.random.multivariate_normal(sep*np.array([0.5, 0.5]), [[0.1,-0.085],[-0.085,0.1]], N//2)
    # Ones
    X2 = np.random.multivariate_normal([-0.25, -0.25], [[0.1,0],[0,0.1]], N//2)
    X = np.append(X1, X2, axis=0)
    y = np.append(np.zeros(N//2), np.ones(N//2))
    indexes = np.arange(len(y))
    np.random.shuffle(indexes)
    if normalize:
        X = (X - X.mean(axis=0))/X.std(axis=0)
    else:
        X[:, 0] = X[:, 0]
        X[:, 1] = X[:, 1]
    return X[indexes], y[indexes]


# Primero probar con N = 3000 para ver bien las distribuciones
X, y = generate_gaussians_distributions(sep=0.5, N = 500, normalize=False, random_state=41)
X_test, y_test = generate_gaussians_distributions(sep=0.5, N = 500, normalize=False, random_state=42)

#print(X)
#plt.scatter(X[y==0,0], X[y==0,1])
#plt.scatter(X[y==1,0], X[y==1,1])
#plt.show()

model = Sequential()

# Agrego una capa que tiene 2 entradas, 1 salida y una funcion de activacion sigmoidea
model.add(Dense(1, input_shape=(2,),activation='sigmoid'))

#Ahora compilamos el modelo, definimos el optimizador
#optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.001)
#model.compile(keras.optimizers.experimental.SGD(learning_rate=0.001), loss='binary_crossentropy',  metrics=['accuracy'])


optimizer = Adam(learning_rate=0.1)

print(model.layers[0].get_weights())

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

#Ahora veo como me quedo el modelo
model.summary()

history = model.fit(X, y, epochs=300, batch_size=16, validation_data=(X_test, y_test))

print(model.layers[0].get_weights())

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.show()
