import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# cost function
def cost_func(x=None, y=None):
    '''Cost function.
    For visualizing contour plot, call f() and collect placeholder nodes for fast GPU calc.
    To incorporate variables to optimize, pass them in as argument to attach as x and y.

    Args:
        x: None if placeholder tensor is used as input. Specify x to use x as input tensor.
        y: None if placeholder tensor is used as input. Specify y to use y as input tensor.

    Returns:
        Tuple (x, y, z) where x and y are input tensors and z is output tensor.
    '''
    if not x:
        x = tf.placeholder(tf.float32, shape=[None, 1])
    if not y:
        y = tf.placeholder(tf.float32, shape=[None, 1])

    # two local minima near (0, 0)
    z = __f1(x, y)

    # 3rd local minimum at (-0.5, -0.8)
    z -= __f2(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.35, y_sig=0.35)

    # one steep gaussian trench at (0, 0)
#     z -= __f2(x, y, x_mean=0, y_mean=0, x_sig=0.2, y_sig=0.2)

    # three steep gaussian trenches
    z -= __f2(x, y, x_mean=1.0, y_mean=-0.5, x_sig=0.2, y_sig=0.2)
    z -= __f2(x, y, x_mean=-1.0, y_mean=0.5, x_sig=0.2, y_sig=0.2)
    z -= __f2(x, y, x_mean=-0.5, y_mean=-0.8, x_sig=0.2, y_sig=0.2)

    return x, y, z


# noisy hills of the cost function
def __f1(x, y):
    return -1 * tf.sin(x * x) * tf.cos(3 * y * y) * tf.exp(-(x * y) * (x * y)) - tf.exp(-(x + y) * (x + y))


# bivar gaussian hills of the cost function
def __f2(x, y, x_mean, y_mean, x_sig, y_sig):
    normalizing = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = (-1 * tf.square(x - x_mean)) / (2 * tf.square(x_sig))
    y_exp = (-1 * tf.square(y - y_mean)) / (2 * tf.square(y_sig))
    return normalizing * tf.exp(x_exp + y_exp)


# pyplot settings
plt.ion()
plt.figure(figsize=(3, 2), dpi=300)
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
params = {'legend.fontsize': 3,
          'legend.handlelength': 3}
plt.rcParams.update(params)
plt.axis('off')

# input (x, y) and output (z) nodes of cost-function graph
x, y, z = cost_func()

# visualize cost function as a contour plot
x_val = y_val = np.arange(-1.5, 1.5, 0.005, dtype=np.float32)
x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
y_val_mesh_flat = y_val_mesh.reshape([-1, 1])
with tf.Session() as sess:
    z_val_mesh_flat = sess.run(z, feed_dict={x: x_val_mesh_flat, y: y_val_mesh_flat})
z_val_mesh = z_val_mesh_flat.reshape(x_val_mesh.shape)
levels = np.arange(-10, 1, 0.05)
plt.contour(x_val_mesh, y_val_mesh, z_val_mesh, levels, alpha=.7, linewidths=0.4)
plt.draw()

# starting location for variables
x_i = 0.75
y_i = 1.0

# create variable pair (x, y) for each optimizer
x_var, y_var = [], []
for i in range(7):
    x_var.append(tf.Variable(x_i, [1], dtype=tf.float32))
    y_var.append(tf.Variable(y_i, [1], dtype=tf.float32))

# create separate graph for each variable pairs
cost = []
for i in range(7):
    cost.append(cost_func(x_var[i], y_var[i])[2])

# define method of gradient descent for each graph
ops_param = [['Adadelta', 50],
             ['Adagrad', 0.10],
             ['Adam', 0.05],
             ['Ftrl', 0.5],
             ['GD', 0.05],
             ['Momentum', 0.01],
             ['RMSProp', 0.02]]

ops = []
ops.append(tf.train.AdadeltaOptimizer(ops_param[0][1]).minimize(cost[0]))
ops.append(tf.train.AdagradOptimizer(ops_param[1][1]).minimize(cost[1]))
ops.append(tf.train.AdamOptimizer(ops_param[2][1]).minimize(cost[2]))
ops.append(tf.train.FtrlOptimizer(ops_param[3][1]).minimize(cost[3]))
ops.append(tf.train.GradientDescentOptimizer(ops_param[4][1]).minimize(cost[4]))
ops.append(tf.train.MomentumOptimizer(ops_param[5][1], momentum=0.95).minimize(cost[5]))
ops.append(tf.train.RMSPropOptimizer(ops_param[6][1]).minimize(cost[6]))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # use last location to draw a line to the current location
    last_x, last_y = [], []
    plot_cache = []
    for i in range(7):
        last_x.append(x_i)
        last_y.append(y_i)
        plot_cache.append(None)

    # available colors for each label
    colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

    # loop each step of the optimization algorithm
    steps = 1000
    for iter in range(steps):
        for i, op in enumerate(ops):
            # run a step of optimization and collect new x and y variable values
            _, x_val, y_val = sess.run([op, x_var[i], y_var[i]])

            # move dot to the current value
            if plot_cache[i]:
                plot_cache[i].remove()
            plot_cache[i] = plt.scatter(x_val, y_val, color=colors[i], s=3, label=ops_param[i][0])

            # draw a line from the previous value
            if last_x[i] and last_y[i]:
                plt.plot([last_x[i], x_val], [last_y[i], y_val], color=colors[i], linewidth=0.5)
            last_x[i] = x_val
            last_y[i] = y_val

        plt.legend(plot_cache, ops_param)
        plt.savefig('figures/' + str(iter) + '.png')
        print('iteration: {}'.format(iter))
        plt.pause(0.001)

print("done")
