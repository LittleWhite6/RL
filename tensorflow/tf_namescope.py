import tensorflow as tf
with tf.name_scope('123'):
    '''
    with tf.name_scope('456'):
        with tf.compat.v1.variable_scope('789'):
            a = tf.Variable(1, name='a')
            print(a.name)
            b = tf.compat.v1.get_variable('b', 1)
            print(b.name)
    with tf.name_scope('456'):
        with tf.compat.v1.variable_scope('789'):
            c = tf.Variable(1, name='c')
            print(c.name)
            d = tf.Variable(1, name='d')
            print(d.name)
    
    with tf.name_scope('456'):
        with tf.compat.v1.variable_scope('789'):
            d=tf.Variable(1,name='d')
            print(d.name)
        with tf.compat.v1.variable_scope('789'):
            e=tf.compat.v1.get_variable('e',1)
            print(e.name)
    '''
    with tf.name_scope('123'):
        with tf.name_scope(None):
            c=tf.Variable(1,name='f')
            print(c.name)