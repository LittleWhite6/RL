import tensorflow as tf

'''
def test():
    state = tf.Variable(0, name='counter')
    print(state.name)
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for __ in range(3):
            sess.run(update)
            print(sess.run(state))
            # assign未被执行，ref值不更新，assign只能给变量赋值，常量报错
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