'''
tf.placeholder(dtype,shape=None,name=None)
该函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符传入数据。
'''
import tensorflow as tf
import numpy as np


print("example_1")
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)
with tf.Session() as sess:
    print(sess.run(output,feed_dict={input1:[3.],input2:[4.]})) 
#feed_dict是字典形式.   整数后面加个点， 如 7.  ，表示的是浮点数，省略了.后面的0