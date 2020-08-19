import numpy as np
import random
import tensorflow as tf
import time

import matplotlib.pyplot as plt

from src.util.helper import param_reader

# Generates a sine function with a random amplitude and phase on the interval [-5, 5] and attempts to fit a 2-layer mlp to it.
# Add dynamic layer functionality later?

def sinewave(cf, seed, params = np.array([])):
    layer_size = cf.input_size
    x_0 = -5.0
    x_n = 5.0

    # Sample sine wave parameters
    a = random.uniform(0.1, 5.0)
    b = random.uniform(0.0, 2*np.pi)

    # Sample training data

    if cf.metatrain:
        p = 50
        x_train = np.linspace(x_0, x_n, p)

    else:
        p = 10
        x_train = np.zeros(p)
        for i in range(p):
            x_train[i] = random.uniform(x_0, x_n)
    
    y_train = a*np.sin(x_train + b)

    # Build and compile MLP
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation=cf.activation, input_shape=[1]),
        tf.keras.layers.Dense(64, activation=cf.activation),
        tf.keras.layers.Dense(1)
    ])
    
    #Other model attributes
    optimizer = tf.keras.optimizers.SGD(cf.learning_rate)
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer = optimizer, loss = loss)

    #Set MLP parameters if necessary
    if params.size == 0:
        if cf.metatrain:
            paramset = model.trainable_variables
            for i in range(len(paramset)):
                paramset[i] = paramset[i].numpy()
            params = np.concatenate(paramset, axis=None)
            # Pass initial parameters to reptile for initial meta-update
            param_reader(cf, params)
    else:
        # Shape data generated for each layer for accurate parameter assignment
        slices = np.empty(len(model.trainable_variables) - 1, dtype=int)
        slices[0] = model.trainable_variables[0].numpy().size
        shapelist = [model.trainable_variables[0].numpy().shape]
        for i in range(1,slices.size):
            holder = model.trainable_variables[i].numpy()
            slices[i] = holder.size + slices[i-1]
            shapelist.append(holder.shape)
        shapelist.append(model.trainable_variables[slices.size].numpy().shape)

        # Split parameters and make assignments
        params = np.split(params, slices)
        for i in range(len(params)):
            model.trainable_variables[i].assign(tf.convert_to_tensor(np.reshape(params[i],shapelist[i]), dtype=tf.float32))

    # Perform regression against data samples
    start_time = time.time()
    model.fit(x_train, y_train, batch_size=cf.batch_size, epochs=cf.num_of_iterations)
    end_time = time.time()
    time_elapsed = end_time - start_time
    
    # Collect parameters
    paramset = model.trainable_variables
    for i in range(len(paramset)):
        paramset[i] = paramset[i].numpy()
    params = np.concatenate(paramset, axis=None)

    # Evaluation on training set
    score = model.evaluate(x_train, y_train)

    # Meta-learning test evaluation
    # Saves plot of curves to main folder
    if cf.metatest:
        x_test = np.arange(x_0, x_n, (x_n - x_0)/100.0)
        y_test = model.predict(x_test)
        y_true = a*np.sin(x_test + b)

        plt.plot(x_test, y_true, label = 'True Curve')
        plt.plot(x_test, y_test, label = 'Trained Curve')
        plt.plot(x_train, y_train, 'g^', label = 'Sampled Points')
        plt.legend(loc="upper left")
        plt.savefig('src/ReptileDemo/Output/seed_{}_iterations_{}.png'.format(seed, cf.num_of_iterations))
        plt.clf()


    exp_name = cf.framework
    exact_score = 'N/A'
    #exact_score = 'True function is f(x) = ' + str(a) + '*sin(x + ' + str(b) + ')'
    return exp_name, score, time_elapsed, exact_score, params