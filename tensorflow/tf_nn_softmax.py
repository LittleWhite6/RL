''' tf.nn.softmax(
    logits,
    axis=None,
    name=None)
该方法用来做归一化，logits是一个张量，数据类型必须是half, float32, float64
softmax = e^logits /  Σe^logits 
'''

import tensorflow as tf

x = tf.constant([[3., 1., -3.]])
tf.global_variables_initializer()
with tf.Session() as sess:
    print(sess.run(tf.nn.softmax(x)))
