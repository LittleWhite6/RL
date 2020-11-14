import tensorflow as tf
with tf.compat.v1.variable_scope('layer1',reuse=tf.compat.v1.AUTO_REUSE):
    a1 = tf.Variable(tf.constant(1.0, shape=[1]),name="a")
    a2 = tf.Variable(tf.constant(1.0, shape=[1]),name="a")
    a3 = tf.compat.v1.get_variable("a", shape=[1], initializer=tf.constant_initializer(1.0))
    a4 = tf.compat.v1.get_variable("a", shape=[1], initializer=tf.constant_initializer(1.0))
    print(a1)
    print(a2)
    print(a1==a2)
    print(a3)
    print(a4)
    print(a3==a4)