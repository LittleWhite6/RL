import tensorflow as tf

'''
a=tf.Variable(1,name='a')   #创建一个变量
g=tf.compat.v1.get_default_graph()  #获取默认计算图
print (g.get_operations())  #将计算图中的操作输出

a=tf.compat.v1.get_variable('a',1)
g=tf.compat.v1.get_default_graph()
print(g.get_operations())
'''

a=tf.Variable('a',1)
b=tf.Variable('b',2)
c=tf.add(a,b,'c')
g=tf.compat.v1.get_default_graph()
print(g.get_operations()[-1])
print(g.get_tensor_by_name('c:0'))  #利用Graph的方法进行查找