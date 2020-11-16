list1 = ['Google', 'Runoob', 1997, 2000]

list2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'a', 'b', 'c', 'd', 'e', 'f']

print(list2[0], list2[1])
print(list2[-1], list2[-2])
# list(m:n)表示从列表第m个元素起到第n个元素之前
print(list2[1:3])
print(list2[1:-2])
# 表示输出从列表第3个元素到最后
print(list2[3:])


list1[1] = "Apple"
print(list1[1])
# del语句删除列表元素
del list1[1]
print(list1)
# 列表求长度
print(len(list1))
# 列表拼接
print(list1+list2)
print(list1*4)
print(1997 in list1)
for x in list1:
    print(x, end=" ")
print()


# 嵌套列表
a = ['a', 'b', 'c']
n = [1, 2, 3]
x = [a, n]
print(x)
print(x[0])
print(x[0][1])


# Python列表函数&方法
print("Python列表函数&方法：")

# max(list)返回列表元素最大值,min(list)返回列表元素最小值
print(max(a), min(a))

# list(seq)将元组转换为列表
aTuple = (123, 'Google', 'Runoob')
print(aTuple, list(aTuple))

# list.append(obj)在列表末尾添加新的对象
a.append('New')
print(a)

#list.count(obj)统计某个元素在列表中出现的次数
a.append('a')
print(a.count('a'))

#list.extend(seq)在列表末尾一次性追加另一个序列中的多个值（用新列表扩展原来的列表）
a.extend(n)
print(a)

#list.index(obj)从列表中找出某个值第一个匹配项的索引位置
print(a.index('a'))

#list.insert(index,obj)将对象插入列表
a.insert(2,4)
print(a)

#list.pop([index=-1])移除列表中的一个元素（默认最后一个元素），并且返回该元素的值
print(a.pop(2),a)

#list.remove(obj)移除列表中某个值的第一个匹配项
a.remove('a')
print(a)

#list.reverse()反向列表中元素
a.reverse()

#list.sort(key=None,reverse=False)对原列表进行排序
# key -- 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序
# reverse -- 排序规则，reverse=True降序，reverse=False升序（默认）
vowels=['e','a','u','o','i']
vowels.sort(reverse=True)
print(vowels)
def takeSecond(elem):
    return elem[1]
random=[(2,2),(3,4),(4,1),(1,3)]
random.sort(key=takeSecond)
print("排序列表:",random)

#list.clear()清空列表，类似于del a[:]
a.clear()
print(a)

#list.copy()复制列表
a=random.copy()
print("复制random给a列表：",a)

#append与extend区别
x = [1, 2, 3]
x.append([4, 5])
print (x)
x = [1, 2, 3]
x.extend([4, 5])
print (x)