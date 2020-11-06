import random

'''
random.random()
随机浮点数：0<= n < 1.0
'''
a = random.random()
print(a)


'''
random.uniform(a,b)
用于生成一个指定范围内的随机浮点数，两个参数其中一个是上限，一个是下限。如果a > b，则生成的随机数n: b <= n <= a。如果 a <b， 则 a <= n <= b。
'''
print(random.uniform(1, 10))
print(random.uniform(10, 1))


'''
random.randint(a,b)
用于生成一个指定范围内的整数，其中参数a是下限，b是上限.
'''
print(random.randint(1, 10))


'''
random.randrange([start],stop[,step])
从指定范围内，按指定基数递增的集合中获取一个随机数
'''
print(random.randrange(10, 30, 2))


'''
random.choice(sequence)从序列中获取一个随机元素
参数sequence表示一个有序类型。sequence在python中不是一种特定的类型，而是泛指一系列的类型。list,tuple,字符串都属于sequence。
'''
lst = ['python', 'C', 'C++', 'java']
str1 = {'I love Python'}
print(random.choice(lst))
print(random.choice(str1))


'''
random.shuffle(x[,random])
用于将一个列表中的元素打乱，即将列表内的元素随机排列
'''
p = {'A', 'B', 'C', 'D', 'E'}
random.shuffle(p)
print(p)


'''
random.sample(sequence,k)
从指定序列中随机获取指定长度的片段并随机排列.
注意：sample函数不会修改原有序列
'''
lst = [1, 2, 3, 4, 5]
print(random.sample(lst, 4))
print(lst)