from train import *
import sys

action_times = np.zeros(shape=(action_num))
model = torch.load("./Model/100_episode_4500.pkl")
#print_net_parameters(model.embed_net)

f = open("resulut.txt", "a+")
if __name__ == "__main__":
    model.embed_net.eval()
    start = datetime.datetime.now()
    for episode in range(2000):
        problem = generate_problem()
        init_solution = construct_solution(problem)
        solution = copy.deepcopy(init_solution)
        best_solution = copy.deepcopy(init_solution)
        best_solutions = []
        no_change = 0
        problem_feature = generate_problem_features(problem)
        solution_feature = generate_solution_features(problem, init_solution)
        feature = torch.from_numpy(np.concatenate((problem_feature, solution_feature), axis=1))
        # 问题的初始特征feature: shape = (num_train_points + 1, 8)
        adjacency_M = generate_adjacency_matrix(problem ,init_solution)
        # 初始解决方案的邻接矩阵，shape = (num_train_points + 1, num_train_points + 1)

        feature = model.embed_net(feature, adjacency_M)
        # 解决方案特征值, shape = (64)
        state = generate_state()
        observation = torch.cat((state, feature))
        for rollout in tqdm(range(max_rollout_num), desc='episode{}_rollout'.format(episode)):
            action = model.choose_action(observation)
            action_times[action] += 1
            next_solution, label = env_step(rollout, problem, solution, action, no_change)
            next_solution.cost = next_solution.get_cost(problem)
            delta = solution.cost - next_solution.cost
            reward = 0
            if delta > 0:
                solution = copy.deepcopy(next_solution)
                solution_feature = generate_solution_features(problem, solution)
                feature = torch.from_numpy(np.concatenate((problem_feature, solution_feature), axis=1))
                feature = model.embed_net(feature, adjacency_M)
                reward += (1 * delta)
                no_change = 0
            else:
                reward += (1 * delta)
                no_change += 1
            min_delta = best_solution.cost - next_solution.cost
            if min_delta > 0:
                best_solution = copy.deepcopy(next_solution)
                #print("当前最优解为:{}".format(best_solution.cost))
                reward += (10 * min_delta)
            state = generate_state(state, action, reward, min_delta, delta)
            observation = torch.cat((state, feature))
                

        print(best_solution.cost)
        sys.stdout.flush()
        best_solutions.append(best_solution.cost)
    end = datetime.datetime.now()
    time = (end - start).total_seconds()
    print("算法总训练时间:{}秒".format(time), file=f)
    print("best_solution.cost: ",np.mean(best_solutions), file=f)
    print("算法总训练时间:{}秒".format(time))
    print("best_solution.cost: ",np.mean(best_solutions))
    f.close()