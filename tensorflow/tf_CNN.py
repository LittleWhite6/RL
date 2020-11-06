#5个卷积部分的输出通道数
filter_depth=[64,128,256,512]
VGG_NMU_HIDDEN_1,VGG16_NUM_HIDDEN_2=4096,1000
def submodel(x,iters,k_width,in_depth,out_depth):
    for i in range(iters):