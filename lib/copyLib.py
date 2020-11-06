import copy
origin = [1, 2, [3, 4]]
cop1 = copy.copy(origin)
# 浅拷贝：只拷贝内存中对象的地址引用。
# 对浅拷贝或拷贝源任意一个进行修改，另一个随之修改
cop2 = copy.deepcopy(origin)
# 深拷贝：不仅仅拷贝可变对象的地址引用，还拷贝可变对象在内存中的数据。
# 修改任意一个对象，另一个拷贝对象不受影响
print(cop1 == cop2)
print(cop1 is cop2)
origin[2][0] = "hey!"
print(origin)
print(cop1)
print(cop2)

'''
定制copy行为：
在自定义类时可通过__copy()__与__deepcopy()__来实现浅拷贝与深拷贝
浅拷贝copy没有参数
深拷贝deepcopy具有参数memo dictionary。对象需要深拷贝的属性值都通过参数传递给方法
'''
import functools

@functools.total_ordering
#函数修饰符@的作用是为现有函数增加额外的功能，常用于插入日志，性能测试，事务处理等
class Myclass:
    def __init__(self,name):
        self.name=name
    def __copy__(self):
        return Myclass(self.name)
    def __deepcopy__(self,memo):
        return Myclass(copy.deepcopy(self.name,memo))

a=Myclass("Name")
sc=copy.copy(a)
dc=copy.deepcopy(a)