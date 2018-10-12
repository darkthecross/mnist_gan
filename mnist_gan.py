import keras
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
import cv2


def build_g(in_tensor):
    fc1 = tf.layers.dense(
        in_tensor, 64, activation=tf.nn.leaky_relu, name='g_fc1')
    fc2 = tf.layers.dense(in_tensor, 32 * 32,
                          activation=tf.nn.leaky_relu, name='g_fc2')
    rs1 = tf.reshape(fc2, (-1, 32, 32, 1))
    conv1 = tf.layers.conv2d(
        rs1, 64, 3, activation=tf.nn.leaky_relu, name='g_conv1')
    conv2 = tf.layers.conv2d(
        conv1, 1, 3, activation=tf.nn.leaky_relu, name='g_conv2')
    return tf.sigmoid(conv2)


def build_d(in_tensor):
    conv1 = tf.layers.conv2d(
        in_tensor, 64, 3, activation=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE, name='d_conv1')
    conv2 = tf.layers.conv2d(
        conv1, 16, 3, activation=tf.nn.leaky_relu, reuse=tf.AUTO_REUSE, name='d_conv2')
    flt = tf.layers.flatten(conv2, name='d_flt')
    fc1 = tf.layers.dense(flt, 128, activation=tf.nn.leaky_relu,
                          reuse=tf.AUTO_REUSE, name='d_fc1')
    fc2 = tf.layers.dense(fc1, 10, activation=tf.nn.leaky_relu,
                          reuse=tf.AUTO_REUSE, name='d_fc2')
    return tf.sigmoid(fc2)


def get_batch(x_t, batch_size):
    xx = np.zeros((batch_size, 28, 28, 1), dtype=np.float32)
    idx = np.random.choice(x_t.shape[0], batch_size)
    for i in range(batch_size):
        xx[i, :, :, :] = x_t[idx[i], :, :, :]
    return xx


def main():
    img_rows, img_cols = 28, 28
    batch_size = 128
    num_classes = 10
    epochs = 12
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    unique, counts = np.unique(y_train, return_counts=True)
    count_dict = dict(zip(unique, counts))
    x_nines = np.zeros((count_dict[9], 28, 28, 1), dtype=np.float32)
    count_nines = 0
    for i in range(y_train.shape[0]):
        if y_train[i] == 9:
            x_nines[count_nines, :, :, :] = x_train[i, :, :, :]
            count_nines += 1

    z = tf.placeholder(tf.float32, shape=(None, 10))
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, 1))
    r = tf.placeholder(tf.float32, shape=(None, 10))

    gz = build_g(z)
    dz = build_d(gz)
    dx = build_d(x)

    g_loss = -tf.reduce_mean(tf.log(dz))
    d_loss = -tf.reduce_mean(tf.log(dx) + tf.log(1.-dz))

    tvars = tf.trainable_variables()
    g_vars = [var for var in tvars if 'g_' in var.name]
    # print(g_vars)
    g_trainer = tf.train.AdamOptimizer(
        0.0001).minimize(g_loss, var_list=g_vars)

    d_vars = [var for var in tvars if 'd_' in var.name]
    # print(d_vars)
    d_trainer = tf.train.AdamOptimizer(
        0.0001).minimize(d_loss, var_list=d_vars)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(2000):
            # training for D
            rand_z = np.random.rand(128, 10)
            batch_x = get_batch(x_nines, 128)

            feed_dict = {x: batch_x, z: rand_z}

            _, _, gl, dl = sess.run([g_trainer, d_trainer, g_loss, d_loss],
                                    feed_dict=feed_dict)

            if i % 30 == 0 or i == 1:
                print('Step %i: Generator Loss: %f, Discriminator Loss: %f' %
                      (i, gl, dl))
                rand_z = np.random.rand(1, 10)
                img = sess.run(gz, feed_dict={z: rand_z})
                res_img = img[0, :, :, :]
                show_img = cv2.resize(res_img, (84, 84))
                show_img[:, :] = show_img[:, :] * 255
                show_img = show_img.astype(np.uint8)
                # print(res_img)
                cv2.imshow('im', show_img)
                cv2.imwrite('img/iter'+str(i)+'.png', show_img)
                cv2.waitKey(1)


if __name__ == '__main__':
    main()
