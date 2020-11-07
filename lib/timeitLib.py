#性能度量
from timeit import Timer
print(Timer('t=a; a=b; b=t','a=1; b=2').timeit())
print(Timer('a,b=b,a', 'a=1; b=2').timeit())
#相对于timeti的细粒度， :mod:profile和pstats模块提供了针对更大代码块的时间度量工具