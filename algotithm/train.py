import datetime
import time
from tqdm import tqdm

from ENV import *
from drl import *

def generate_problem_features(problem):
    features = np.zeros(shape=(num_train_points + 1, 4), dtype=np.float32)
    # features = [node_number, coordinate_x, coordinate_y, node_capacity]
    for node_number in range(num_train_points + 1):
            features[node_number][0] = node_number
            features[node_number][1] = problem.locations[node_number][0]
            features[node_number][2] = problem.locations[node_number][1]
            features[node_number][3] = problem.capacities[node_number]
    return features


def generate_solution_features(problem, solution):
    features = np.zeros(shape=(num_train_points + 1, 4), dtype=np.float32)
    # features = [route_number, node_in_route_position, vehicle_load，total_route_demand]
    for i in range(len(solution.path)):
        path_len = len(solution.path[i])
        path_nodes = []
        load = 0
        for j in range(1, path_len - 1):
            node = solution.path[i][j]
            path_nodes.append(node)
            load += problem.capacities[node]
            features[node][0] = i
            features[node][1] = j
            features[node][2] = load
        for node in path_nodes:
            features[node][3] = load
    return features


def generate_adjacency_matrix(problem ,solution, a_M = None):
    if a_M == None:
        a_M = torch.zeros(size=(num_train_points + 1, num_train_points + 1))
        for path in solution.path:
            for node_1, node_2 in zip(path[:len(path) - 1], path[1:len(path)]):
                a_M[node_1][node_2] = 1
                a_M[node_2][node_1] = 1
    return a_M


#生成每一次rollout的state
def generate_state(state=None, action=0, reward=0, delta_min=0, delta=0):
    state = torch.Tensor([action, reward, delta_min, delta])
    return state


# 打印网络参数函数
def print_net_parameters(net):
    for name, parameters in net.named_parameters():
        print(name, ':', parameters.size())


# 判断sparse or dense,是否将GAT改为SPGAT
def determine_graph_class(adjacency_M):
    count=0
    for i in range(adjacency_M.size()[0]):
        for j in range(adjacency_M.size()[1]):
            if adjacency_M[i][j]==1:
                count+=1
    if num_train_points * 10 > count:
        print("此图为sparse graph")
    else:
        print("此图为dense grapg")


# 验证解决方案是否满足节点及容量约束
def validate_solution(solution, problem):
    #capacity check
    for path_index in range(len(solution.path)):
        load = problem.capacities[0]
        route = solution.path[path_index]
        for node in range(1, len(route) - 1):
            node_index = route[node]
            load -= problem.capacities[node_index]
            if load < 0:
                print("当前解决方案不满足容量约束，问题在第{}个路径的{}节点".format(route, node_index))
                return False
    
    #node_number check
    nodes = np.zeros(num_train_points)
    for path_index in range(len(solution.path)):
        route = solution.path[path_index]
        for node in range(1, len(route) - 1):
            nodes[route[node] - 1] += 1
    if 0 in nodes:
        print("当前解决方案缺少节点")
        return False
    return True


def print_rollout_iteration(rollout, action, delta, solution_cost, next_solution_cost, best_solution_cost, episode):
    if rollout % 500 == 0 and episode % 50 == 0:
        print("rollout num {}".format(rollout), end="\t")
        print("action:{}".format(action), end="\t")
        print("delta:{}".format(delta))
        print("solution_cost:{}".format(solution_cost))
        print("next_solution_cost:{}".format(next_solution_cost))
        print("best_solution_cost:{}".format(best_solution_cost))


if __name__ == "__main__":
    model = MODEL()
    model.embed_net.train()
    start = datetime.datetime.now()
    for episode in range(max_episodes):
        problem = generate_problem()
        init_solution = construct_solution(problem)
        solution = copy.deepcopy(init_solution)
        best_solution = copy.deepcopy(init_solution)
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
        best_no_change = 0

        for rollout in tqdm(range(max_rollout_num), desc='episode{}_rollout'.format(episode)):
            observation = torch.cat((state, feature))
            action = model.choose_action(observation)
            next_solution, step_delta = env_step(rollout, problem, solution, action)
            next_solution.cost = next_solution.get_cost(problem)
            '''
            if not validate_solution(next_solution, problem):
                print("解决方案有问题")
                time.sleep(20)
                continue
            '''
            delta = solution.cost - next_solution.cost
            # 这步是否可以省略？ delta = step_delta
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
            else:
                best_no_change += 1
            #print_rollout_iteration(rollout, action, delta, solution.cost, next_solution.cost, best_solution.cost, episode)
            state = generate_state(state, action, reward, min_delta, delta)
            observation_ = torch.cat((state, feature))
            model.store_transition(observation.detach().numpy(), action, reward, observation_.detach().numpy())
            observation = observation_
            if model.memory_counter > MEMORY_CAPACITY:
                model.learn()

            # pertubation controller
            if no_change > 6:
                problem.record_solution(solution.path, solution.cost)
                if solution.cost / best_solution.cost < 1.01:
                    solution.path = pertubation_operator(problem, solution.path, rollout)
                else:
                    solution.path = pertubation_operator(problem, best_solution.path, rollout)
                solution.cost = solution.get_cost(problem)
                no_change = 0
            if best_no_change == 1000:
                break

        #每经历1000次采样保存模型
        if episode % 1000 == 0 and episode != 0:
            torch.save(model, './Model/{}_episode_{}.pkl'.format(num_train_points,episode))
        print(best_solution.cost)
    end = datetime.datetime.now()
    time = (end - start).total_seconds()
    print("算法总训练时间:{}秒".format(time))