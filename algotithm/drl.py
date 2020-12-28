import numpy as np
import itertools
from hyper_parameter import *
from embed_net import *

class Action_Net(nn.Module):
    def __init__(self):
        super(Action_Net, self).__init__()
        self.fc1 = nn.Linear(5 + num_train_points, 10)
        self.fc1.weight.data.normal_(0, 0.1)    #initialization
        self.out = nn.Linear(10, action_num)
        self.out.weight.data.normal_(0, 0.1)    #initialization
    
    def forward(self, feature):
        feature = self.fc1(feature)
        feature = F.relu(feature)
        actions_values = self.out(feature)
        return actions_values

class MODEL(object):
    def __init__(self):
        self.embed_net = EMBED_NET(feature_size, 64, 0, Dropout, Alpha, N_heads)
        # nclass = 0, 推测原文中用于节点分类，故在当前实验中取消
        self.eval_net, self.target_net = Action_Net(),Action_Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, (num_train_points + 1 + 4) * 2 + 2))
        self.optimizer = torch.optim.Adam(itertools.chain(self.eval_net.parameters(), self.embed_net.parameters()), lr=LR)
        self.loss_func = nn.MSELoss()
    
    def choose_action(self, feature):
        if np.random.uniform() < EPSILON:
            action_values = self.eval_net.forward(feature)
            action_value = torch.max(action_values)
            # 寻找q_value最大的动作的索引
            for max_index in range(len(action_values)):
                if action_value == action_values[max_index]:
                    return max_index
        else:
            action = np.random.randint(0, action_num)
        return action
    
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        #如果记忆库满了，则覆盖旧数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self. learn_step_counter += 1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        observation_size = num_train_points + 4 + 1
        b_s = torch.FloatTensor(b_memory[:, :observation_size])
        b_a = torch.LongTensor(b_memory[:, observation_size:observation_size+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, observation_size+1:observation_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, -observation_size:])

        # 针对做过的动作b_a, 选择q_eval的值
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # shape = (batch, 1)
        q_next = self.target_net(b_s_).detach()
        # q_next 不进行反向传递误差, 所以 detach

        # 原来V2.0版本code中由q_next生成的q_target的shape=(32,32)与q_eval不匹配，现在改变代码为下面2行后形状匹配
        q_next = q_next.max(1)[0]
        q_next = q_next.view(-1, 1)
        
        q_target = b_r + DISCOUNT_FACTOR*q_next
        # shape = (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        #计算，更新eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()