EPSILON = 0.9
# epison-greedy 90%最优 10%随机

problem_seed = 0
# Global seed

num_train_points = 20
# Global 训练样本点数

max_episodes = 5000
#共进行多少次训练

max_rollout_num = 20000
#每次训练迭代多少次

action_num = 22
#动作个数

num_full_features = 68

DISCOUNT_FACTOR = 0.9
#奖励折扣因子lambda

LR = 0.01
#学习效率alpha < 1 , 表示当前误差有多少要被学习
#new_Q(s1,a1) = old_Q(s1,a1) + alpha*(lambda*maxQ(s2))
#决策部分等到更新完了再做

feature_size = 8
# feature = [node_number, coordinate_x, coordinate_y, node_capacity, route_number, node_in_route_position, vehicle_load(车辆到当前节点的负载)，total_route_demand]

fresh_time = 60
#fresh time for one move (1m)

Dropout = 0.6
#Dropout rate (1 - keep probability)

Alpha = 0.2
#GAT's Alpha for the LeakyReLu

N_heads = 8
# Multi-head attention nums

WEIGHT_DECAY = 5e-4

MEMORY_CAPACITY = 500
# 记忆库容量

TARGET_REPLACE_ITER = 100

BATCH_SIZE = 32