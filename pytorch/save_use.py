import torch
import torch.nn.functional as F

torch.manual_seed(1)    #reproducible

x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1))
    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    #训练
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    torch.save(net1, './Model/net.pkl') #保存整个网络
    torch.save(net1.state_dict(), './Model/net_params.pkl') #只保存网络中的参数（速度快，占内存少）

def restore_net():
    net2 = torch.load('./Model/net.pkl')
    prediction = net2(x)

def restore_params():
    #新建 net3
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    #将保存的参数复制到net3,需要俩个网络结构相同
    net3.load_state_dict(torch.load('./Model/net_params.pkl'))
    prediction = net3(x)

#保存网络net1
save()

#提取整个网络
restore_net()

#提取网络参数，复制到新网络
restore_params()