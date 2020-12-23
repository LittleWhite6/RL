import tqdm

from ENV import *
from embed_net import *

def generate_solution_features(problem, solution):
    features = np.zeros(shape=(num_train_points, feature_size), dtype=np.float32)
    # features = [route_number, node_in_route_position, vehicle_load，total_route_demand]

    for i in range(1, num_train_points + 1):
        features[i - 1][0] = i
        features[i - 1][3] = problem.capacities[i]

    for i in range(len(solution.path)):
        path_len = len(solution.path[i])
        load = problem.capacities[0]
        for j in range(1, path_len - 1):
            node = solution.path[i][j]
            load -= problem.capacities[node]
            features[node - 1][1] = i
            features[node - 1][2] = j
            features[node - 1][4] = load

    return features


def generate_problem_features(problem):
    features = np.zeros(shape=(num_train_points, 4), dtype=np.float32)
    # features = [node_number, coordinate_x, coordinate_y, node_capacity]
    for node_number in range(num_train_points + 1):
        for features_index in range(4):



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
        problem_features = generate_problem_features(problem, init_solution)