import math
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def f(x, y):
    z = -1 * tf.sin(x * x) * tf.cos(3 * y * y) * tf.exp(-(x * y) * (x * y)) - tf.exp(-(x + y) * (x + y))
    return z


plt.ion()
plt.figure(figsize=(3, 2), dpi=300)
params = {'legend.fontsize': 4,
          'legend.handlelength': 4}
plt.rcParams.update(params)
plt.axis('off')

x = np.arange(-1.5, 1.5, 0.05)
y = np.arange(-1.0, 1.0, 0.05)
X, Y = np.meshgrid(x, y)
z = f(X, Y)

with tf.Session() as sess:
    Z = sess.run(z)

plt.contour(X, Y, Z)
plt.draw()

x_i = 0.5
y_i = 0.75

x_var = []
y_var = []
for i in range(7):
    x_var.append(tf.Variable(x_i, [1], dtype=tf.float32))
    y_var.append(tf.Variable(y_i, [1], dtype=tf.float32))

cost = []
for i in range(7):
    cost.append(f(x_var[i], y_var[i]))

ops = []
ops.append(tf.train.AdadeltaOptimizer(20).minimize(cost[0]))
ops.append(tf.train.AdagradOptimizer(0.05).minimize(cost[1]))
ops.append(tf.train.AdamOptimizer(0.01).minimize(cost[2]))
ops.append(tf.train.FtrlOptimizer(0.05).minimize(cost[3]))
ops.append(tf.train.GradientDescentOptimizer(0.05).minimize(cost[4]))
ops.append(tf.train.MomentumOptimizer(0.01, 0.95).minimize(cost[5]))
ops.append(tf.train.RMSPropOptimizer(0.03).minimize(cost[6]))

ops_label = ["Adadelta",
             "Adagrad",
             "Adam",
             "Ftrl",
             "GD",
             "Momentum",
             "RMSProp"]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    last_x = []
    last_y = []
    last_point = []
    for i in range(7):
        last_x.append(x_i)
        last_y.append(y_i)
        last_point.append(None)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for iter in range(300):
        for i, op in enumerate(ops):
            _, x_val, y_val = sess.run([op, x_var[i], y_var[i]])

            if last_point[i]:
                last_point[i].remove()
            last_point[i] = plt.scatter(x_val, y_val, color=colors[i], s=3, label=ops_label[i])

            if last_x[i] and last_y[i]:
                plt.plot([last_x[i], x_val], [last_y[i], y_val], color=colors[i], linewidth=0.5)
            last_x[i] = x_val
            last_y[i] = y_val

        plt.legend(last_point, ops_label)  # , prop={'size': 7})
        plt.savefig("op_figs/" + str(iter) + '.png')
        print(iter)
        plt.pause(0.001)

print("done")
