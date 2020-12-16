import torch
import torch.utils.data as Data
torch.manual_seed(1)

BATCH_SIZE = 5  #批训练的数据个数
#每次取一批数据训练，当剩下的数据个数不足一批时，取剩下的全部数据训练

x = torch.linspace(1, 10, 10)
y = torch.linspace(10, 1 ,10)

#先转换成torch能识别的Dataset
loader = Data.DataLoader(
    dataset = torch_dataset,    #torch TensorDataset
    batch_size = BATCH_SIZE,    #mini batch size
    shuffle = True, #要不要打乱数据
    num_workers=2   #多线程读数据
)

for epoch in range(3):  #训练所有数据
    for step, (batch_x,batch_y) in enumerate(loader):   #每一步loader释放一小批数据用来学习
        #train

        #打印数据
        print('Epoch:',epoch,'|Step:',step,'|batch x:',
            batch_x.numpy(),'|batch y:',batch_y.numpy())

