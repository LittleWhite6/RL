import datetime
# Python处理日期和时间的标准库
now = datetime.datetime.now()
# datetime是模块，模块中还包含一个datetime类.
# 仅导入datetime则必须引用全名datetime.datetime.
print(now)
print(type(now))

dt = datetime.datetime(2015, 4, 19, 12, 20)
print(dt)

# datetime转换为时间戳timestamp,浮点数，小数位表示毫秒数，无时区概念
dt.timestamp()
print(dt)

# timestamp转换为datetime
t = 1429417200.0
print(datetime.datetime.fromtimestamp(t))
# datetime是有时区的，上述转换是在timestamp和本地时间做转换
# 本地时间是指当前操作系统设定的时区，例如北京时区是8区，即UTC+8:00时区

# str转换为datetime
cday = datetime.strptime('2015-6-1 18:19:59', '%Y-%m-%d %H:%M:%S')
print(cday)

# datetime加减
datetime.datetime(2015, 5, 18, 16, 57, 3, 540997)
print(now+timedelta(hours=10))
print(now-timedelta(days=1))
print(now+timedelta(days=2, hours=12))
