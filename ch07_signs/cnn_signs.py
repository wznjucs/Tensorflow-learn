import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

from datetime import datetime
import time

np.random.seed(1)



def load_dataset():
    
    def load_raw_data():
        train_dataset = h5py.File('datasets/train_signs.h5', "r")
        train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

        test_dataset = h5py.File('datasets/test_signs.h5', "r")
        test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

        classes = np.array(test_dataset["list_classes"][:]) # the list of classes

        train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
        test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

        return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    
    def convert_to_one_hot(Y, out_C):
        Y = np.eye(out_C)[Y.reshape(-1)].T
        return Y
    
    def pre_process(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig):
        X_train = X_train_orig / 255.
        X_test = X_test_orig / 255.
        Y_train = convert_to_one_hot(Y_train_orig, 6).T
        Y_test = convert_to_one_hot(Y_test_orig, 6).T
        return X_train, Y_train, X_test, Y_test

    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_raw_data()
    X_train, Y_train, X_test, Y_test = pre_process(X_train_orig, Y_train_orig, X_test_orig, Y_test_orig)
#     print ("X_train shape: " + str(X_train.shape))
#     print ("Y_train shape: " + str(Y_train.shape))
#     print ("X_test shape: " + str(X_test.shape))
#     print ("Y_test shape: " + str(Y_test.shape))
    return X_train, Y_train, X_test, Y_test, classes
	
	
	
# GRADED FUNCTION: forward_propagation

def forward_propagation(X):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
    """
    
    with tf.name_scope("parameters"):
        parameters = {
            "conv1": {
                "w" : tf.get_variable(name="conv1_w", shape=[4,4,3,8], initializer=tf.contrib.layers.xavier_initializer()),
                "b" : tf.Variable(tf.constant(0.0, shape=[8]), name="conv1_b"),
                "strides": [1,1,1,1],
                "padding": "SAME",
            },
            "max_pool1":{
                "ksize" : [1,8,8,1],
                "strides": [1,8,8,1],
                "padding": "SAME",
            },
            "conv2": {
                "w" : tf.get_variable(name="conv2_w", shape=[2,2,8,16], initializer=tf.contrib.layers.xavier_initializer()),
                "b" : tf.Variable(tf.constant(0.0, shape=[16]), name="conv2_b"),
                "strides": [1,1,1,1],
                "padding": "SAME",
            },
            "max_pool2": {
                 "ksize" : [1,4,4,1],
                "strides": [1,4,4,1],
                "padding": "SAME",
            },
            
        }
    
    def conv2d(X, conv_params):
        Z = tf.nn.conv2d(X, filter=conv_params["w"], strides=conv_params["strides"], padding=conv_params["padding"])
        conv = tf.nn.relu(tf.nn.bias_add(Z, conv_params["b"]))
        return conv
    
    def max_pool(X, pool_params):
        return tf.nn.max_pool(X, ksize=pool_params["ksize"], strides=pool_params["strides"], padding=pool_params["padding"])
    

    conv1 = conv2d(X, parameters["conv1"])
    
    pool1 = max_pool(conv1, parameters["max_pool1"])

    conv2 = conv2d(pool1, parameters["conv2"])
    
    pool2 = max_pool(conv2, parameters["max_pool2"])
#     print(pool2)
    
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
#     print(pool_shape[0], nodes)
    P2 = tf.reshape(pool2, [-1, nodes])
#     print(P2)
    P2 = tf.contrib.layers.flatten(pool2)
#     print(P2)
    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None" 
    Z3 = tf.contrib.layers.fully_connected(P2, 6, activation_fn=None)
#     print(Z3)
    return Z3
	
	
# GRADED FUNCTION: model

def model(learning_rate=0.015, num_epochs=105, batch_size=128, print_cost=True):
    """
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)

    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    # Clears the default graph stack and resets the global default graph.
    # to be able to rerun the model without overwriting tf variables
    ops.reset_default_graph()

    # to keep results consistent (tensorflow seed)
    SEED = 3
    tf.set_random_seed(seed=SEED)

    # Loading the data (signs)
    X_train, Y_train, X_test, Y_test, classes = load_dataset()

    # Create Placeholders of the correct shape
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Pred = forward_propagation(X)

    # Cost function: Add cost function to tensorflow graph
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Pred, labels=Y))

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    # Calculate the correct predictions and compute accuracy
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
    # To keep track of the cost
    costs = []

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):
            seed = 0
            epoch_cost = 0.
            num_batch = int(m / batch_size)  # number of batches of size batch_size in the train set
            seed = seed + 1
            start_time = time.time()
            
            total_batch = random_mini_batches(X_train, Y_train, batch_size, seed)

            for one_batch in total_batch:
                # Select a minibatch
                (batch_xs, batch_ys) = one_batch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                sess.run(optimizer, feed_dict={X:batch_xs, Y: batch_ys})
                batch_cost_val = sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})
                epoch_cost += batch_cost_val / num_batch
            
            costs.append(epoch_cost)
            train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
            test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Epoch:%03d/%03d  Cost:%.6f  Train Accuracy:%.6f  Test Accuracy:%.6f" 
                      % (epoch+1, num_epochs, epoch_cost, train_accuracy, test_accuracy))
                
                duration = time.time() - start_time
                # print('%s: epoch %03d, duration = %.3f' % (datetime.now(), epoch + 1, duration))
            
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        return train_accuracy, test_accuracy

model(learning_rate=0.02, num_epochs=105, batch_size=128, print_cost=True)
