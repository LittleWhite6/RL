from train import *

model = torch.load("./Model/50_episode_1000.pkl")
#print_net_parameters(model.embed_net)

if __name__ == "__main__":
    model = MODEL()
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

        for rollout in tqdm(range(max_rollout_num), desc='episode{}_rollout'.format(episode)):
            observation = torch.cat((state, feature))
            action = model.choose_action(observation)
            next_solution, label = env_step(problem, solution, action, no_change)

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


            #输出结果
            if rollout % 500 == 0 and episode % 100 == 0:
                print("rollout num {}".format(rollout), end="\t")
                print("action:{}".format(action), end="\t")
                print("delta:{}".format(delta))
                print("solution_cost:{}".format(solution.cost))
                print("next_solution_cost:{}".format(next_solution.cost))
                print("best_solution_cost:{}".format(best_solution.cost))


            state = generate_state(state, action, reward, min_delta, delta)
            observation_ = torch.cat((state, feature))
            model.store_transition(observation.detach().numpy(), action, reward, observation_.detach().numpy())
            observation = observation_
            if no_change > 100:
                break
        best_solutions.append(best_solution.cost)
    end = datetime.datetime.now()
    time = (end - start).total_seconds()
    print("算法总训练时间:{}秒".format(time))
    print("best_solution.cost: ",np.mean(best_solutions))