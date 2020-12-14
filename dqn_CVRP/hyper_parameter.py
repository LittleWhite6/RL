EPSILON = 0.9
# epison-greedy 90%最优 10%随机

problem_seed = 0
# Global seed

num_train_points = 20
# Global 训练样本点数

max_episodes = 5
#共进行多少次训练

max_rollout_num = 20000
#每次训练迭代多少次

action_num = 10
#动作个数

num_full_features = 68

discount_factor = 0.9
#奖励折扣因子lambda

learning_rate = 0.01
#学习效率alpha < 1 , 表示当前误差有多少要被学习
#new_Q(s1,a1) = old_Q(s1,a1) + alpha*(lambda*maxQ(s2))
#决策部分等到更新完了再做

feature_size = 5
# feature = [node_number, route_number, node_in_route_position, node_capacity, vehicle_load]

fresh_time = 60
#fresh time for one move (1m)