'''tf.clip_by_value(1-y,1e-10,1.0)
功能：可以将一个张量中的数值限制在一个范围之内。（可以避免一些运算错误:可以保证在进行log运算时，不会出现log0这样的错误或者大于1的概率）
参数：（1）1-y：input数据（2）1e-10、1.0是对数据的限制。
当1-y小于1e-10时，输出1e-10；
当1-y大于1e-10小于1.0时，输出原值；
当1-y大于1.0时，输出1.0；
'''
import tensorflow as tf
v=tf.constant([1.0,2.0,3.0],[4.0,5.0,6.0])
print(tf.clip_by_value(v,2.5,4.5).eval())
#输出[[2.5,2.5,3.],[4.,4.5,4.5]]

'''eval(expression, globals=None, locals=None)  
官方文档中的解释是，将字符串str当成有效的表达式来求值并返回计算结果。
globals和locals参数是可选的，如果提供了globals参数，那么它必须是dictionary类型；
如果提供了locals参数，那么它可以是任意的map对象。'''