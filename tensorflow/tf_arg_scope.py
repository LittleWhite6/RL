#Tensorflow的contrib文件夹中包含arg_scope，使用这个功能只需要定义一个关于参数的上下文空间，并在其中定义默认参数的赋值
import tensorflow as tf
import tf.contrib.framework.arg_scope as arg_scope
#top1=tf.nn.conv2d(bottom,filter=[3,3,3,16],strides=[1,1,1,1],padding='SAME',data_format='NHWC)
with arg_scope([tf.nn.conv2d],strides=[1,1,1,1],padding='SAME',data_format='NHWC'):
    top1=tf.nn.conv2d(bottom1,filter=[3,3,3,16])
    bottom2=tf.nn.relu(top1)
    top2=tf.nn.conv2d(bottom2,filter=[3,3,16,32])
    bottom3=tf.nn.relu(top2)

'''
实现参数共享的功能并不复杂，用到了contexlib库和装饰器这样的设计模式。思路上整个框架需要完成以下俩个主要工作：
1）定义时传入一些设定好的共享参数，并将其保存，必要时还可以实现参数的嵌套覆盖等机制，这部分工作主要由contexlib完成。
2）在函数被调用时，将设定的默认参数传入函数中。这部分工作由装饰器完成。
'''

#contexlib部分
from contextlib import contextmanager
import time

@contextmanager
def timer(prefix):
    start_time = time.time()
    yield()
    duration=time.time()-start_time
    print(prefix+str(duration))

#装饰器部分
#import time
def deco(fuc):
    start_time=time.time()
    func()
    duration=time.time()-start_time
    print(prefix+str(duration))
    return func
#定义完成后，只要在函数前面加上与上面定义的函数名同名的注解
@deco
def my_fuc():
    #code


#example
@add_arg_scope
def func1(*args,**kwargs):
    return (args,kwargs)
with arg_scope((func1),a=1,b=None,c=[1]):
    args,kwargs=func1(0)
    print(args)
    print(kwargs)