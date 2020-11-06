import tensorflow as tf
a = tf.Variable(1.0, name='a')
loss = a-1
opt=tf.train.GradientDescentOptimizer(0.1)
g=tf.compat.v1.get_default_graph()  
'''
tensorflow的计算一般分为俩个部分：第一阶段定义计算图中的所有计算，第二阶段执行计算。
tensorflow中的每一个计算都是计算图上的一个节点，节点之间的边描述了计算之间的关系。
不同计算图上的张量和运算都不会共享,tf.get_default_grapg()可以用来获取当前的计算图(compat.v1.)
'''
print(g.get_operations())   #op,可以叫做节点或操作，输入和输出都是tensor张量
grad=opt.compute_gradients(loss)
print(g.get_operations())
train_op=opt.apply_gradients(grad)
print(g.get_operations())
tf.summary.FileWriter("logs",g).close()
print(g.get_collection('variables'))    #显示计算图中的变量
print(g.get_collection('train_variables'))  #显示可以训练的变量
#Tensorflow使用Session来执行定义好的计算图
