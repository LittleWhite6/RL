import tensorflow as tf

#计算图
a = tf.Variable([[1., 2., 3., 4.], [5., 6., 7., 8.]])
a = tf.reduce_mean(a, axis = 0)
a = tf.reshape(a, shape=(1, 4))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(a))