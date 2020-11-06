import tensorflow as tf
import numpy as np
with tf.name_scope('Normal'):
    x=tf.compat.v1.placeholder(tf.float32,[],'x')
    a=tf.Variable(0.0,'a')
    c=tf.random.uniform([],0.0,1.0)  #tf.random_uniform <=> tf.random.uniform
    op=a.assign(c) 
    c2=tf.random_uniform([],0.0,1.0)
    op2=a.assign(c2)
    with tf.control_dependencies([op]):
        f=a*x
        with tf.control_dependencies([op2]):
            g=tf.gradients(f,x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([f, g], feed_dict={x: 1}))

#stop_gradient方法用于告诉计算图，方法中的变量将不参与梯度计算
with tf.name_scope('normal'):
    x = tf.placeholder(tf.float32, [], 'a')
    a1 = tf.random_uniform([], 0.0, 1.0)
    a2 = tf.random_uniform([], 0.0, 1.0)
    with tf.control_dependencies([op]):
        f1 = a1*x
        f2 = a2*x
        #下面这句话是关键
        f=f2+tf.stop_gradient(f1-f2)
        with tf.control_dependencies([op2]):
            g=tf.gradients(f,x)

#注册梯度函数
@tf.RegisterGradient("mult_grad")
def _mult_grad(op,grad):
    c2=np.random.uniform(0.0,1.0)
    return op.inputs[1]*grad,c2*grad
g=tf.compat.v1.get_default_graph()
x=tf.placeholder(tf.float32,[])
a=tf.random_uniform([],0.0,1.0)
with g.gradient_override_map({"Mul":"mult_grad"}):
    f=tf.multiply(a,x)
    g=tf.gradients(f,x)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([f,g],feed_dict={x:1}))