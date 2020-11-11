import tensorflow as tf

matrix1 = tf.constant([[1,3]])    #1行2列
matrix2 = tf.constant([[2],[4]])  #2行1列
product = tf.matmul(matrix1,matrix2) #matrix multiply np.dot(m1,m2)

#method 1
'''sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()'''

#method2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
