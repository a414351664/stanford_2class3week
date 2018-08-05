# encoding:utf-8
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.python.framework import ops
from tf_utils import *

model_path = "./model/"


# 数据处理，将X, Y变成(nx, m)的形式Y(classes, m)
def data_deal():
    x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes = load_dataset()
    # x_train : (1080, 64, 64, 3)   test 有12０张
    # y_train:(1, 1080)
    # index = 0
    # plt.imshow(x_train_orig[index])
    # plt.show()
    # print('y = ', np.squeeze(y_train_orig[:, index]))
    x_train_flatten = x_train_orig.reshape(x_train_orig.shape[0], -1).T
    x_test_flatten = x_test_orig.reshape(x_test_orig.shape[0], -1).T
    x_train = x_train_flatten / 255.
    x_test = x_test_flatten / 255.
    # x_train (nx, m)
    y_train = convert_to_one_hot(y_train_orig, len(classes))
    y_test = convert_to_one_hot(y_test_orig, len(classes))
    # 用上面的表达式　得到的是数组，用这个表达式，得到的是张量　y_train = tf.one_hot(y_train_orig.reshape(-1), len(classes), axis=0)
    return x_train, x_test, y_train, y_test


def initial_parameters(layer_dim):
    parameters = {}
    for i in range(1, len(layer_dim)):
        parameters["W" + str(i)] = tf.get_variable(name=("W" + str(i)), shape=[layer_dim[i], layer_dim[i - 1]],
                                                   initializer=tf.contrib.layers.xavier_initializer(seed=1))
        parameters["b" + str(i)] = tf.get_variable(name=("b" + str(i)), shape=[layer_dim[i], 1],
                                                   initializer=tf.zeros_initializer())
    return parameters


def compute_cost(Z3, Y):
    # 将其变为(m, nx)的形式
    Z3 = tf.transpose(Z3)
    Y = tf.transpose(Y)
    t = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z3)
    loss = tf.reduce_mean(t)
    return loss


def model(x_train, x_test, y_train, y_test, learning_rate=0.0001, num_epochs=1500, batch_size=32):
    ops.get_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x, m) = x_train.shape
    n_y = y_train.shape[0]
    costs = []

    # define the X, Y
    X = tf.placeholder(tf.float32, shape=[n_x, None], name='X')
    Y = tf.placeholder(tf.float32, shape=[n_y, None], name='Y')

    # initial the parameters
    layer_dim = (x_train.shape[0], 25, 12, y_train.shape[0])
    parameters = initial_parameters(layer_dim)

    # forward_computer
    logits = forward_propagation(X, parameters)
    cost = compute_cost(logits, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # initial the all parameters
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        # save model
        # saver = tf.train.Saver()
        # ckpt = tf.train.get_checkpoint_state(model_path)
        # if ckpt is not None:
        #     path = ckpt.model_checkpoint_path
        #     print('loading pre-trained model from %s.....' % path)
        #     saver.restore(sess, path)

        sess.run(init)
        for epoch in range(num_epochs):
            # 用于计算每一次迭代的cost,平均每个batch的cost
            epoch_cost = 0
            # 计算有多少个ｂａｔｃｈ
            num_batches = int(m / batch_size)
            # 获得ｂａｔｃｈｅｓ
            batches = random_mini_batches(x_train, y_train, batch_size, seed)
            for batch in batches:
                x_batch, y_batch = batch
                _, batch_cost = sess.run([optimizer, cost], feed_dict={X: x_batch, Y: y_batch})

                epoch_cost += batch_cost
            epoch_cost /= num_batches
            if epoch % 100 == 0:
                print("epoch, ", epoch, "cost is ", epoch_cost)
            if epoch % 5 == 0:
                costs.append(epoch_cost)
            # if epoch % (num_epochs / 10) == 0 and epoch != 0:
            #     # sess.run(tf.assign(net.global_step, epoch))
            #     saver.save(sess, model_path + 'points', global_step=epoch)

        # plot the costs
        plt.plot(np.squeeze(costs))
        plt.xlabel("iteration (per tens)")
        plt.ylabel("cost")
        plt.title("Learning_rate = " + str(learning_rate))
        plt.show()

        # saver the parameters
        parameters = sess.run(parameters)

        # computer correct_pre
        correct_pre = tf.equal(tf.argmax(logits), tf.argmax(Y))

        # accuracy 将bool转为float
        accuracy = tf.reduce_mean(tf.cast(correct_pre, 'float'))

        print('Train :', accuracy.eval({X: x_train, Y: y_train}))
        print('Test :', accuracy.eval({X: x_test, Y: y_test}))
        return parameters


def main():
    x_train, x_test, y_train, y_test = data_deal()
    parameters = model(x_train, x_test, y_train, y_test)
    pass


if __name__ == '__main__':
    main()
