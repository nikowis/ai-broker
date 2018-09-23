import tensorflow as tf


# Number of stocks in training data

def construct_model(X_train):
    n_stocks = X_train.shape[1]

    # Neurons
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128

    # Session
    net = tf.InteractiveSession()

    # Placeholder
    X_plchldr = tf.placeholder(dtype=tf.float32, shape=[None, n_stocks])
    Y_plchldr = tf.placeholder(dtype=tf.float32, shape=[None])

    # Initializers
    sigma = 1
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Hidden weights
    W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

    # Output weights
    W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
    bias_out = tf.Variable(bias_initializer([1]))

    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X_plchldr, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    # Output layer (transpose!)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y_plchldr))

    # Optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)

    # Init
    net.run(tf.global_variables_initializer())

    return net, opt, mse, out, X_plchldr, Y_plchldr
