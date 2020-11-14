import tensorflow as tf
import numpy as np

# 定义一个矩阵a,表示需要被卷积的矩阵
a = np.array(np.arange(1, 1+20).reshape([1, 10, 2]), dtype=np.float32)  #np.arange()第一个参数为起点默认0;第二个参数为终点;第三个参数为步长默认1，支持小数。
#print(a)

# 卷积核，此处卷积核的数目为1
kernel = np.array(np.arange(1, 1+4), dtype=np.float32).reshape([2, 2, 1])
#print(kernel)

# 进行conv1d卷积
conv1d = tf.nn.conv1d(a, kernel, 1, 'VALID')

with tf.Session() as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 输出卷积值
    out=sess.run(conv1d)
    print(out)


'''tf.nn.conv1d(value,filters,stride,padding,use_cudnn_on_gpu=None,data_format=None,name=None)
参数：
    value：在注释中，value的格式为：[batch, in_width, in_channels]，batch为样本维，表示多少个样本，in_width为宽度维，表示样本的宽度，in_channels维通道维，表示样本有多少个通道。
    filters：filters：在注释中，filters的格式为：[filter_width, in_channels, out_channels]。按照value的第二种看法，filter_width可以看作每次与value进行卷积的行数，in_channels表示value一共有多少列（与value中的in_channels相对应）。out_channels表示输出通道，可以理解为一共有多少个卷积核，即卷积核的数目。
    stride：一个整数，表示步长，每次（向下）移动的距离。
    padding：'SAME'或'VALID'
    use_cudnn_on_gpu：可选的bool,默认为True。
    data_format：一个可选的string,可以是"NWC"和"NCW"；默认为"NWC",数据按[batch,in_width,in_channels]的顺序存储；"NCW"格式将数据存储为[batch, in_channels, in_width]。
    name：操作的名称(可选)。'''