#!/usr/bin/python3

class MyClass:
    """一个简单的类实例"""
    i = 12345

    def f(self):
        return "Hello World"


# 实例化类
x = MyClass()
# 访问类的属性和方法
print("MyClass类的属性i为：", x.i)
print("MyClass类的方法f输出为：", x.f())


"""
类有一个名为__init__()的特殊方法(构造方法)，该方法在类实例化时会自动调用，像下面这样:
def __init__(self):
    self.data = []
__init__()方法可以有参数，参数通过方法传递到类的实例化操作上。
"""


class Complex:
    def __init__(self, realpart, imagpart):
        self.r = realpart
        self.i = imagpart


x = Complex(3.0, -4.5)
print(x.r, x.i)


# self代表类的实例，而非类. (self不是python关键字，我们把他换成runoob也是可以正常执行的)
class Test:
    def prt(self):
        print(self)
        print(self.__class__)


t = Test()
t.prt()


# 在类的内部，使用def关键字来定义一个方法，与一般函数定义不同，类方法必须包含参数self，且为第一个参数，self代表的是类的实例
class people:
    # 定义基本属性
    name = ''
    age = 0
    # 定义私有属性，私有属性在类外部无法直接进行访问
    __weight = 0
    # 定义构造方法

    def __init__(self, n, a, w):
        self.name = n
        self.age = a
        self.__weight = w

    def speak(self):
        print("{0}说：我{1}岁".format(self.name, self.age))


# 实例化类
p = people('runoob', 10, 30)
p.speak()


'''
Python支持类的继承，派生类的定义如下所示：
class DerivedClassName(BaseClassName1):
    <statement-1>
    .
    .
    .
    <statement-N>
BaseClassName(示例中的基类名)必须与派生类定义在一个作用域内。
除了类，还可以用表达式，基类定义在另一个模块中时这一点非常有用：
    class DerivedClassName(modname.BaseClassName):
'''


class student(people):
    grade = ''

    def __init__(self, n, a, w, g):
        # 调用父类的构造函数
        people.__init__(self, n, a, w)
        self.grade = g
    # 覆盖父类的方法

    def speak(self):
        print("{0} 说：我{1}岁了，我在读{2}年纪".format(self.name, self.age, self.grade))


s = student('ken', 10, 60, 3)
s.speak()


'''
Python同样有限的支持多继承形成。多继承的类定义形如下例：
class DerivedClassName(Base1, Base2, Base3):
    <statemen-1>
    .
    .
    .
    <statemen-N>
需要注意圆括号中父类的顺序，若是父类中有相同的方法名，而在子类使用时未指定，python从左至右搜索，即方法在子类中未找到时，从左到右查找父类中是否包含方法
'''


class speaker():
    topic = ''
    name = ''

    def __init__(self, n, t):
        self.name = n
        self.topic = t

    def speak(self):
        print("我叫 {} ,我是一个演说家，我演讲的主题是{}".format(self.name, self.topic))
# 多重继承


class sample(speaker, student):
    a = ''

    def __init__(self, n, a, w, g, t):
        student.__init__(self, n, a, w, g)
        speaker.__init__(self, n, t)


test = sample("Tim", 25, 80, 4, "Python")
test.speak()  # 方法名同，默认调用的是在括号中排在前面的父类的方法

# 方法重写


class Parent:
    def myMethod(self):
        print('调用父类方法')


class Child(Parent):
    def myMethod(self):
        print('调用子类方法')


c = Child()  # 子类实例
c.myMethod()  # 子类调用重写方法
super(Child, c).myMethod  # 用子类对象调用父类已被覆盖的方法
# super()函数是用于调用父类(超类)的一个方法


'''
类属性与方法：
    1)类的私有属性:
    __private_attrs: 俩个下划线开头，声明该属性为私有，不能在类的外部被使用或直接访问。在类内部的方法中使用时self.__private_attrs
    2)类的方法：
    def name(self): self的名字并不是规定死的，也可以使用this,最好是self
    3)类的私有方法：
    __private_method: 俩个下划线开头，声明该方法为私有方法，只能在类的内部调用，不能在类的外部调用。self.__private_methods。
'''


class JustCounter:
    __secretCount = 0  # 私有变量
    publicCount = 0  # 公开变量
    def count(self):
        self.__secretCount += 1
        self.publicCount += 1
        print(self.__secretCount)
    def prt_private(self):
        print(self.__secretCount)
counter = JustCounter()
counter.count()
counter.count()
print(counter.publicCount)
counter.prt_private()

'''
类的专有方法
__init__: 构造函数，在生成对象时调用
__del__: 析构函数，释放对象时调用
__repr__: 打印，转换
__setitem__: 按照索引赋值
__getitem__: 按照索引获取值
__len__: 获取长度
__cmp__: 比较运算
__call__: 函数调用
__add__: 加运算
__sub__: 减运算
__mul__: 乘运算
__truediv__: 除运算
__mod__: 求余运算
__pow__: 乘方   
'''


'''运算符重载'''
class Vector:
    def __init__(self,a,b):
        self.a = a
        self.b = b
    def __str__(self):
        return 'Vector ({},{})'.format(self.a,self.b)
    def __add__(self,other):
        return Vector(self.a+other.a,self.b+other.b)
v1 = Vector(2,10)
v2 = Vector(5,-2)
print(v1+v2)