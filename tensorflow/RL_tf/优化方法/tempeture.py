import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a=np.array([-40,14,32,46,59,72,100],dtype=float)

for i,c in enumerate(celsius_q):
    print("{} degrees Celsius = {} degrees Fahrenhet".format(c,fahrenheit_a[i]))

l0 = tf.keras.layers.Dense(units=4,input_shape=[1])
l1 = tf.keras.layers.Dense(units=4)
l2 = tf.keras.layers.Dense(units=1)
model=tf.keras.Sequential([l0,l1,l2])
model.compile(loss='mean_squared_error',optimizer=tf.keras.optimizers.Adam(0.1))
history=model.fit(celsius_q,fahrenheit_a,epochs=500,verbose=False)
print("Finished training the model")

import matplotlib.pyplot as plt
plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])

print(model.predict([100.0]))
print("These are the l0 layer variables: {}".format(l0.get_weights()))
print("These are the l1 layer variables: {}".format(l1.get_weights()))
print("These are the l2 layer variables: {}".format(l2.get_weights()))