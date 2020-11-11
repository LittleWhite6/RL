import tensorflow as tf

'''
assign(ref,value,validate_shape=None,use_locking=None,name=None)： 
通过将value赋给ref来更新ref. 此操作输出在赋值后保留新值ref的张量. 这使得更易于链接需要使用重置值得操作.
ARGS:
    ref: 一个可变的张量，来自变量节点，节点可能未初始化
    value: 张量.必须具有与ref相同的类型，是要分配给变量的值
    validate_shape: 一个可选的bool,默认为ture. 如果为true，则操作将验证value的形状是否与分配给的张量的形状相匹配； 如果为false,ref将对值的形状进行引用
    use_locking: 一个可选的bool. 默认为True. 如果为True, 则分配将受锁保护；否则，该行为是未定义的，但可能会显示较少的争用.
    name: 操作的名称
'''

def test_1():
    a = tf.Variable([10, 20])
    b = tf.assign(a, [20, 30])
    c = a + [10, 20]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("test_1 run a : ", sess.run(a))  # => [10 20]
        # => [10 20]+[10 20] = [20 40] 因为b没有被run所以a还是[10 20]
        print("test_1 run c : ", sess.run(c))
        # => ref:a = [20 30] 运行b，对a进行assign
        print("test_1 run b : ", sess.run(b))
        # => [20 30] 因为b被run过了，所以a为[20 30]
        print("test_1 run a again : ", sess.run(a))
        # => [20 30] + [10 20] = [30 50] 因为b被run过了，所以a为[20,30], 那么c就是[30 50]
        print("test_1 run c again : ", sess.run(c))


def test_2():
    a = tf.Variable([10, 20])
    b = tf.assign(a, [20, 30])
    c = b + [10, 20]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(a))  # => [10 20]
        print(sess.run(c))  # => [30 50] 运行c的时候，由于c中含有b，所以b也被运行了
        print(sess.run(a))  # => [20 30]

def main():
    test_1()
    test_2()

if __name__ == '__main__()':
    main()

main()