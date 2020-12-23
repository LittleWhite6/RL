import tqdm

from ENV import *
from embed_net import *


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
    if state:
        state = [action, reward, delta_min, delta]
    else:
        state = [action, reward, delta_min, delta]
    return state


if __name__ == "__main__":
    for i in range(max_episodes):
        problem = generate_problem()
        init_solution = construct_solution(problem)
        best_solution = copy.deepcopy(init_solution)
        problem_feature = generate_problem_features(problem)
        solution_feature = generate_solution_features(problem, init_solution)

        feature = torch.from_numpy(np.concatenate((problem_feature, solution_feature), axis=1))
        # 问题的初始特征feature: shape = (num_train_points + 1, 8)
        adjacency_M = generate_adjacency_matrix(problem ,init_solution)
        # 初始解决方案的邻接矩阵，shape = (num_train_points + 1, num_train_points + 1)

        model = GAT_net(feature, adjacency_M, 8, 64)